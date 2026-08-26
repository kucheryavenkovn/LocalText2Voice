from __future__ import annotations

import shutil
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
from .generation import GenerationCancelled
from .models import CueStatus, DubbingCue
from .wav_validator import WavArtifactValidator, atomic_replace


ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]


class CueGenerationError(RuntimeError):
    pass


class CueGenerationCancelled(GenerationCancelled):
    """Cancelled during a single-cue generation step."""


@dataclass(frozen=True)
class CueGenerationConfig:
    sample_rate: int = 48000
    channels: int = 2
    trim_silence: bool = True
    silence_threshold_db: float = -40.0
    fade_ms: int = 8
    compress_internal_pauses: bool = False
    internal_pause_keep_ms: int = 90
    timing_tolerance_ms: int = 20


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
        self._validator = WavArtifactValidator()
        self._active_part_path: Path | None = None

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

    def cleanup_partial_artifacts(self) -> None:
        part = self._active_part_path
        self._active_part_path = None
        if part is not None:
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
        # Also drop transient normalize/correction sidecars near active outputs.
        for pattern_suffix in (".norm.wav", ".corr.wav", ".fit.part"):
            if part is not None:
                sibling = part.with_name(part.name.replace(".part", "") + pattern_suffix)
                try:
                    sibling.unlink(missing_ok=True)
                except OSError:
                    pass

    def _runner(self) -> FFmpegRunner:
        if self._ffmpeg_runner is None:
            self._ffmpeg_runner = FFmpegRunner(find_ffmpeg(self.ffmpeg_path))
        return self._ffmpeg_runner

    def generate_raw(
        self,
        cue: DubbingCue,
        voice_config: dict[str, Any],
        *,
        apply_pause_compression: bool = False,
    ) -> DubbingCue:
        """Synthesize raw TTS WAV.

        Pause compression is intentionally NOT applied here by default so that
        changing compress settings only requires a refit, not a new TTS pass.
        Format normalize + optional edge silence trim still run on raw.
        """
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
        # Keep a real .wav suffix so FFmpeg/engine writers detect the container.
        part_path = raw_output.with_name(raw_output.stem + ".part.wav")
        self._active_part_path = part_path
        if part_path.exists():
            part_path.unlink(missing_ok=True)

        started = time.perf_counter()
        try:
            per_cue_config = dict(voice_config)
            per_cue_config["_dubbing_sequence"] = cue.sequence
            self.tts_engine.synthesize_to_wav(text, part_path, per_cue_config)
            self._check_cancelled()
            pre_validation = self._validator.validate(part_path)
            if not pre_validation.valid:
                raise CueGenerationError(
                    f"Cue #{cue.sequence} TTS produced invalid WAV: "
                    f"{pre_validation.error_message}"
                )
            # Silence trim can wipe very short/quiet OmniVoice clips to 0 frames.
            # Fall back to format-only normalize (no silenceremove) like older builds.
            normalized = self._normalize_wav(
                part_path,
                compress_internal_pauses=apply_pause_compression,
                trim_silence=self.config.trim_silence,
            )
            use_path = part_path
            if normalized != part_path:
                post = self._validator.validate(normalized)
                if post.valid:
                    atomic_replace(normalized, part_path)
                else:
                    try:
                        normalized.unlink(missing_ok=True)
                    except OSError:
                        pass
                    self.log_callback(
                        f"Cue #{cue.sequence}: silence trim emptied audio; "
                        "keeping format-only normalize."
                    )
                    fallback = self._normalize_wav(
                        part_path,
                        compress_internal_pauses=False,
                        trim_silence=False,
                    )
                    fb = self._validator.validate(fallback)
                    if fb.valid and fallback != part_path:
                        atomic_replace(fallback, part_path)
                    elif not self._validator.validate(part_path).valid:
                        raise CueGenerationError(
                            f"Cue #{cue.sequence} produced invalid WAV after normalize: "
                            f"{post.error_message}"
                        )
            validation = self._validator.validate(use_path)
            if not validation.valid:
                raise CueGenerationError(
                    f"Cue #{cue.sequence} produced invalid WAV: "
                    f"{validation.error_message}"
                )
            atomic_replace(part_path, raw_output)
            self._active_part_path = None
            cue.raw_duration_ms = validation.duration_ms
            cue.is_stale = False
            cue.status = CueStatus.RENDERED.value
            self.log_callback(
                f"Cue #{cue.sequence}: raw WAV {validation.duration_ms} ms "
                f"(synthesis+normalize {self._format_duration(time.perf_counter() - started)})."
            )
            return cue
        except CueGenerationCancelled:
            self.cleanup_partial_artifacts()
            if cue.status == CueStatus.RENDERING.value:
                cue.status = CueStatus.CANCELLED.value
                cue.error_message = "Cancelled."
            raise
        except TTSCancelled as exc:
            self.cleanup_partial_artifacts()
            cue.status = CueStatus.CANCELLED.value
            cue.error_message = "Cancelled."
            raise CueGenerationCancelled(str(exc)) from exc
        except TTSEngineError as exc:
            self.cleanup_partial_artifacts()
            cue.status = CueStatus.FAILED.value
            cue.error_message = str(exc)
            raise CueGenerationError(str(exc)) from exc
        except Exception:
            self.cleanup_partial_artifacts()
            raise

    def apply_fitting(
        self,
        cue: DubbingCue,
        result: FittingResult,
        *,
        compress_internal_pauses: bool | None = None,
        internal_pause_keep_ms: int | None = None,
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

        compress = (
            self.config.compress_internal_pauses
            if compress_internal_pauses is None
            else compress_internal_pauses
        )
        keep_ms = (
            self.config.internal_pause_keep_ms
            if internal_pause_keep_ms is None
            else internal_pause_keep_ms
        )

        factor = result.applied_speed_factor
        source_for_fit = cue.raw_audio_path
        temp_compressed: Path | None = None
        part_path = fitted_path.with_name(fitted_path.stem + ".part.wav")
        self._active_part_path = part_path
        try:
            if compress:
                temp_compressed = fitted_path.with_suffix(".prefit.wav")
                self._run_filters(
                    cue.raw_audio_path,
                    temp_compressed,
                    self._pause_compress_filters(keep_ms),
                )
                source_for_fit = temp_compressed

            if result.strategy.value == "none" or abs(factor - 1.0) <= 0.001:
                shutil.copy2(source_for_fit, part_path)
                validation = self._validator.validate(part_path)
                if not validation.valid:
                    raise CueGenerationError(
                        f"Cue #{cue.sequence} fitted copy invalid: "
                        f"{validation.error_message}"
                    )
                atomic_replace(part_path, fitted_path)
                cue.fitted_duration_ms = validation.duration_ms
            else:
                self._apply_atempo(source_for_fit, part_path, factor)
                validation = self._validator.validate(part_path)
                if not validation.valid:
                    raise CueGenerationError(
                        f"Cue #{cue.sequence} fitted WAV invalid: "
                        f"{validation.error_message}"
                    )
                cue.fitted_duration_ms = validation.duration_ms
                target_ms = result.target_duration_ms
                if (
                    target_ms
                    and cue.fitted_duration_ms
                    and abs(cue.fitted_duration_ms - target_ms)
                    > self.config.timing_tolerance_ms
                ):
                    fine = cue.fitted_duration_ms / target_ms
                    if fine > 1.0:
                        corrected = part_path.with_name(part_path.stem + ".corr.wav")
                        self._apply_atempo(part_path, corrected, fine)
                        atomic_replace(corrected, part_path)
                        validation = self._validator.validate(part_path)
                        if not validation.valid:
                            raise CueGenerationError(
                                f"Cue #{cue.sequence} corrected fit invalid: "
                                f"{validation.error_message}"
                            )
                        cue.fitted_duration_ms = validation.duration_ms
                atomic_replace(part_path, fitted_path)
            self._active_part_path = None
            return cue
        except CueGenerationCancelled:
            self.cleanup_partial_artifacts()
            if cue.status == CueStatus.RENDERING.value:
                cue.status = CueStatus.CANCELLED.value
            raise
        except Exception:
            self.cleanup_partial_artifacts()
            raise
        finally:
            if temp_compressed is not None:
                try:
                    temp_compressed.unlink(missing_ok=True)
                except OSError:
                    pass

    def _apply_atempo(
        self,
        source: Path,
        destination: Path,
        factor: float,
    ) -> None:
        filters = DurationFitter.build_atempo_chain(factor)
        self._run_filters(source, destination, filters)

    def _pause_compress_filters(self, keep_ms: int) -> list[str]:
        keep = max(0.02, keep_ms / 1000.0)
        threshold = f"{self.config.silence_threshold_db}dB"
        return [
            "silenceremove=stop_periods=-1:"
            f"stop_silence={keep:.3f}:stop_threshold={threshold}"
        ]

    def _normalize_wav(
        self,
        raw_output: Path,
        *,
        compress_internal_pauses: bool = False,
        trim_silence: bool | None = None,
    ) -> Path:
        normalized = raw_output.with_suffix(raw_output.suffix + ".norm.wav")
        do_trim = self.config.trim_silence if trim_silence is None else trim_silence
        filters: list[str] = []
        if do_trim:
            # Milder than before: keep more quiet speech so short clips survive.
            threshold = f"{min(self.config.silence_threshold_db, -45.0)}dB"
            filters.append(
                "silenceremove=start_periods=1:start_silence=0.02:"
                f"start_threshold={threshold}:start_duration=0.05"
            )
            filters.append("areverse")
            filters.append(
                "silenceremove=start_periods=1:start_silence=0.02:"
                f"start_threshold={threshold}:start_duration=0.05"
            )
            filters.append("areverse")
        if compress_internal_pauses:
            filters.extend(
                self._pause_compress_filters(self.config.internal_pause_keep_ms)
            )
        if self.config.fade_ms > 0 and do_trim:
            fade = min(self.config.fade_ms, 5) / 1000.0
            filters.append(f"afade=t=in:st=0:d={fade:.4f}")
        filters.append(
            "aresample="
            f"{self.config.sample_rate},aformat=sample_fmts=s16:"
            f"channel_layouts={'stereo' if self.config.channels == 2 else 'mono'}"
        )
        self._run_filters(raw_output, normalized, filters)
        return normalized

    def _run_filters(self, source: Path, destination: Path, filters: list[str]) -> None:
        arguments = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-filter:a",
            ",".join(filters),
            "-ar",
            str(self.config.sample_rate),
            "-ac",
            str(self.config.channels),
            "-codec:a",
            "pcm_s16le",
            str(destination),
        ]
        try:
            self._runner().run(arguments)
        except FFmpegCancelled as exc:
            raise CueGenerationCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise CueGenerationError(f"Could not process cue WAV: {exc}") from exc

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
