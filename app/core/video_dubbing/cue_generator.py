from __future__ import annotations

import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.tts.base import BaseTTSEngine, TTSCancelled, TTSEngineError
from app.utils.ffmpeg_utils import (
    FFmpegCancelled,
    FFmpegError,
    FFmpegRunner,
    find_ffmpeg,
)

from .duration_fitter import DurationFitter, FittingResult
from .models import CueStatus, DubbingCue


ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]


class CueGenerationError(RuntimeError):
    pass


class CueGenerationCancelled(CueGenerationError):
    pass


@dataclass(frozen=True)
class CueGenerationConfig:
    sample_rate: int = 48000
    channels: int = 2
    trim_silence: bool = True
    silence_threshold_db: float = -40.0
    fade_ms: int = 8


class CueGenerator:
    """Generates per-cue raw WAV via a BaseTTSEngine, normalizes the technical
    format, measures duration, and applies the DurationFitter atempo step.

    The generator never edits the source video, never edits SRT timecodes, and
    never trims speech content — only technical leading/trailing silence.
    """

    def __init__(
        self,
        tts_engine: BaseTTSEngine,
        ffmpeg_path: str | Path,
        config: CueGenerationConfig | None = None,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.tts_engine = tts_engine
        self.ffmpeg_path = ffmpeg_path
        self.config = config or CueGenerationConfig()
        self.progress_callback = progress_callback or (lambda c, t, msg: None)
        self.log_callback = log_callback or (lambda msg: None)
        self._cancel_requested = threading.Event()
        self._ffmpeg_runner: FFmpegRunner | None = None

    def cancel(self) -> None:
        self._cancel_requested.set()
        try:
            self.tts_engine.cancel_current()
        except Exception:  # pragma: no cover - defensive
            pass
        runner = self._ffmpeg_runner
        if runner is not None:
            runner.cancel_current()

    def _check_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise CueGenerationCancelled("Cue generation cancelled.")

    def _runner(self) -> FFmpegRunner:
        if self._ffmpeg_runner is None:
            self._ffmpeg_runner = FFmpegRunner(find_ffmpeg(self.ffmpeg_path))
        return self._ffmpeg_runner

    def generate_raw(
        self,
        cue: DubbingCue,
        voice_config: dict[str, Any],
    ) -> DubbingCue:
        cue.error_message = None
        cue.status = CueStatus.RENDERING.value
        cue.attempt_count += 1
        text = cue.spoken_text.strip()
        if not text:
            cue.status = CueStatus.FAILED.value
            cue.error_message = "Empty cue text."
            raise CueGenerationError(f"Cue #{cue.sequence} has empty text.")
        self._check_cancelled()
        raw_output = cue.raw_audio_path
        if raw_output is None:
            raise CueGenerationError(
                f"Cue #{cue.sequence} has no raw audio path assigned."
            )
        raw_output.parent.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        try:
            # Inject the cue sequence so engines/voice configs can vary per cue.
            per_cue_config = dict(voice_config)
            per_cue_config["_dubbing_sequence"] = cue.sequence
            self.tts_engine.synthesize_to_wav(text, raw_output, per_cue_config)
        except TTSCancelled as exc:
            cue.status = CueStatus.PENDING.value
            raise CueGenerationCancelled(str(exc)) from exc
        except TTSEngineError as exc:
            cue.status = CueStatus.FAILED.value
            cue.error_message = str(exc)
            raise CueGenerationError(str(exc)) from exc
        self._check_cancelled()

        normalized = self._normalize_wav(raw_output)
        if normalized != raw_output:
            normalized.replace(raw_output)
        duration_ms = self._measure_wav_duration_ms(raw_output)
        cue.raw_duration_ms = duration_ms
        cue.is_stale = False
        cue.status = CueStatus.RENDERED.value
        self.log_callback(
            f"Cue #{cue.sequence}: raw WAV {duration_ms} ms "
            f"(synthesis+normalize {self._format_duration(time.perf_counter() - started)})."
        )
        return cue

    def apply_fitting(
        self,
        cue: DubbingCue,
        result: FittingResult,
    ) -> DubbingCue:
        if cue.raw_audio_path is None or not cue.raw_audio_path.is_file():
            raise CueGenerationError(
                f"Cue #{cue.sequence} has no raw WAV to fit."
            )
        fitted_path = cue.fitted_audio_path
        if fitted_path is None:
            raise CueGenerationError(
                f"Cue #{cue.sequence} has no fitted audio path assigned."
            )
        fitted_path.parent.mkdir(parents=True, exist_ok=True)
        self._check_cancelled()

        factor = result.applied_speed_factor
        if result.strategy.value == "none" or abs(factor - 1.0) <= 0.001:
            # No speed change: copy raw into fitted so downstream is deterministic.
            import shutil

            shutil.copy2(cue.raw_audio_path, fitted_path)
            cue.fitted_duration_ms = cue.raw_duration_ms
        else:
            filters = DurationFitter.build_atempo_chain(factor)
            arguments = [
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(cue.raw_audio_path),
                "-filter:a",
                ",".join(filters),
                "-ar",
                str(self.config.sample_rate),
                "-ac",
                str(self.config.channels),
                "-codec:a",
                "pcm_s16le",
                str(fitted_path),
            ]
            try:
                self._runner().run(arguments)
            except FFmpegCancelled as exc:
                raise CueGenerationCancelled(str(exc)) from exc
            except FFmpegError as exc:
                raise CueGenerationError(str(exc)) from exc
            cue.fitted_duration_ms = self._measure_wav_duration_ms(fitted_path)
        return cue

    def _normalize_wav(self, raw_output: Path) -> Path:
        normalized = raw_output.with_suffix(".norm.wav")
        filters: list[str] = []
        if self.config.trim_silence:
            threshold = f"{self.config.silence_threshold_db}dB"
            filters.append(
                "silenceremove=start_periods=1:start_silence=0.05:"
                f"start_threshold={threshold}"
            )
            filters.append("areverse")
            filters.append(
                "silenceremove=start_periods=1:start_silence=0.05:"
                f"start_threshold={threshold}"
            )
            filters.append("areverse")
        if self.config.fade_ms > 0:
            fade = self.config.fade_ms / 1000.0
            filters.append(f"afade=t=in:st=0:d={fade:.4f}")
        filters.append(
            "aresample="
            f"{self.config.sample_rate},aformat=sample_fmts=s16:"
            f"channel_layouts={'stereo' if self.config.channels == 2 else 'mono'}"
        )
        arguments = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(raw_output),
            "-filter:a",
            ",".join(filters),
            "-codec:a",
            "pcm_s16le",
            str(normalized),
        ]
        try:
            self._runner().run(arguments)
        except FFmpegCancelled as exc:
            raise CueGenerationCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise CueGenerationError(
                f"Could not normalize cue WAV: {exc}"
            ) from exc
        return normalized

    @staticmethod
    def _measure_wav_duration_ms(path: Path) -> int:
        try:
            with wave.open(str(path), "rb") as audio:
                frames = audio.getnframes()
                rate = audio.getframerate()
            if rate <= 0:
                return 0
            return int(round(frames / rate * 1000))
        except (wave.Error, OSError):
            return 0

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 1.0:
            return f"{seconds * 1000:.0f} ms"
        return f"{seconds:.2f} s"
