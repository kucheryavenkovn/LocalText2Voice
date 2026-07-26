from __future__ import annotations

import math
import threading
import wave
from array import array
from pathlib import Path
from typing import Callable

from app.utils.ffmpeg_utils import (
    FFmpegCancelled,
    FFmpegError,
    FFmpegRunner,
    find_ffmpeg,
)
from app.utils.ffprobe_utils import find_ffmpeg_sibling

from .models import (
    DubbingCue,
    DubbingProject,
    DuckingSettings,
    OriginalAudioMode,
)


ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]


class AudioMixerError(RuntimeError):
    pass


class AudioMixerCancelled(AudioMixerError):
    pass


MASK_SAMPLE_RATE = 1000  # 1 ms resolution; ramps are smooth at this rate.


class AudioMixer:
    """Builds the narration MP3 and the Dubbed Mix WAV.

    The Dubbed Mix combines the original audio (optionally ducked using a
    timecode-driven gain envelope) with the narration track. Ducking is driven
    by a deterministic gain mask built from cue timecodes, not by an envelope
    follower, so it stays exact regardless of content loudness.
    """

    def __init__(
        self,
        ffmpeg_path: str | Path,
        sample_rate: int = 48000,
        channels: int = 2,
        progress_callback: ProgressCallback | None = None,
        log_callback: LogCallback | None = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.progress_callback = progress_callback or (lambda c, t, msg: None)
        self.log_callback = log_callback or (lambda msg: None)
        self._cancel_requested = threading.Event()
        self._runner: FFmpegRunner | None = None

    def cancel(self) -> None:
        self._cancel_requested.set()
        runner = self._runner
        if runner is not None:
            runner.cancel_current()

    def _check_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise AudioMixerCancelled("Audio mix cancelled.")

    def _runner_instance(self) -> FFmpegRunner:
        if self._runner is None:
            self._runner = FFmpegRunner(find_ffmpeg(self.ffmpeg_path))
        return self._runner

    def _audio_format_filter(self) -> str:
        layout = "stereo" if self.channels == 2 else "mono"
        return (
            f"aresample={self.sample_rate},"
            f"aformat=sample_fmts=fltp:channel_layouts={layout}"
        )

    def narration_intervals(self, cues: list[DubbingCue]) -> list[tuple[int, int]]:
        intervals: list[tuple[int, int]] = []
        for cue in cues:
            if not cue.enabled:
                continue
            start = cue.effective_start_ms()
            end = start + max(
                cue.fitted_duration_ms or cue.duration_budget_ms,
                cue.duration_budget_ms,
            )
            if cue.fitted_duration_ms:
                end = start + cue.fitted_duration_ms
            end = min(end, cue.effective_end_ms() + max(0, cue.overflow_ms))
            if end > start:
                intervals.append((start, end))
        intervals.sort()
        merged: list[tuple[int, int]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                prev_start, prev_end = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    def build_duck_mask(
        self,
        cues: list[DubbingCue],
        duration_ms: int,
        settings: DuckingSettings,
        output_path: Path,
    ) -> Path:
        sample_count = max(1, math.ceil(duration_ms / 1000 * MASK_SAMPLE_RATE))
        samples = array("h", [0]) * sample_count
        intervals = self.narration_intervals(cues)
        attack_samples = max(1, int(round(settings.attack_ms / 1000 * MASK_SAMPLE_RATE)))
        release_samples = max(1, int(round(settings.release_ms / 1000 * MASK_SAMPLE_RATE)))
        full = 32767
        for start_ms, end_ms in intervals:
            start_idx = max(0, min(sample_count, round(start_ms / 1000 * MASK_SAMPLE_RATE)))
            end_idx = max(0, min(sample_count, round(end_ms / 1000 * MASK_SAMPLE_RATE)))
            if end_idx <= start_idx:
                continue
            samples[start_idx:end_idx] = array("h", [full]) * (end_idx - start_idx)
            ramp_start = max(0, start_idx - attack_samples)
            for idx in range(ramp_start, start_idx):
                ratio = (idx - ramp_start + 1) / (start_idx - ramp_start + 1)
                samples[idx] = int(full * ratio)
            ramp_end = min(sample_count, end_idx + release_samples)
            for idx in range(end_idx, ramp_end):
                remaining = ramp_end - end_idx
                ratio = 1.0 - (idx - end_idx + 1) / (remaining + 1)
                samples[idx] = max(0, int(full * ratio))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(MASK_SAMPLE_RATE)
            audio.writeframes(samples.tobytes())
        return output_path

    def extract_original_audio(
        self,
        video_path: Path,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arguments = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            str(self.channels),
            "-ar",
            str(self.sample_rate),
            "-codec:a",
            "pcm_s16le",
            str(output_path),
        ]
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise AudioMixerCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise AudioMixerError(f"Could not extract original audio: {exc}") from exc
        return output_path

    def encode_narration_mp3(
        self,
        narration_wav: Path,
        output_path: Path,
        bitrate: str = "192k",
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arguments = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(narration_wav),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(output_path),
        ]
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise AudioMixerCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise AudioMixerError(f"Could not encode narration MP3: {exc}") from exc
        return output_path

    def render_dubbed_mix(
        self,
        project: DubbingProject,
        narration_wav: Path,
        output_path: Path,
        temp_dir: Path | None = None,
    ) -> Path:
        if project.video_path is None or not Path(project.video_path).is_file():
            raise AudioMixerError("Project has no source video; cannot mix.")
        if not narration_wav.is_file():
            raise AudioMixerError("Narration track not found; build it first.")
        settings = project.settings.ducking
        duration_ms = project.duration_ms
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir = temp_dir or project.temp_dir()
        work_dir.mkdir(parents=True, exist_ok=True)

        self.progress_callback(0, 3, "Extracting original audio...")
        original_wav = work_dir / "original_audio.wav"
        self.extract_original_audio(Path(project.video_path), original_wav)
        self._check_cancelled()

        self.progress_callback(1, 3, "Building ducking envelope...")
        mask_path = work_dir / "duck_mask.wav"
        self.build_duck_mask(project.cues, duration_ms, settings, mask_path)
        self._check_cancelled()

        self.progress_callback(2, 3, "Mixing narration and original audio...")
        arguments = self._build_mix_arguments(
            original_wav=original_wav,
            narration_wav=narration_wav,
            mask_path=mask_path,
            settings=settings,
            duration_seconds=duration_ms / 1000.0,
        )
        arguments.append(str(output_path))
        try:
            self._runner_instance().run(arguments)
        except FFmpegCancelled as exc:
            raise AudioMixerCancelled(str(exc)) from exc
        except FFmpegError as exc:
            raise AudioMixerError(f"Could not render dubbed mix: {exc}") from exc

        self.progress_callback(3, 3, "Dubbed mix rendered.")
        self.log_callback(
            f"Dubbed mix rendered: {duration_ms} ms, mode={settings.mode.value}."
        )
        return output_path

    def _build_mix_arguments(
        self,
        original_wav: Path,
        narration_wav: Path,
        mask_path: Path,
        settings: DuckingSettings,
        duration_seconds: float,
    ) -> list[str]:
        fmt = self._audio_format_filter()
        narration_gain = max(0.0, settings.narration_volume_percent / 100.0)
        narration_db = 20 * math.log10(narration_gain) if narration_gain > 0 else -96.0
        original_outside_db = (
            20 * math.log10(max(0.0001, settings.original_outside_percent / 100.0))
        )
        original_during_db = (
            20 * math.log10(max(0.0001, settings.original_during_percent / 100.0))
        )
        arguments: list[str] = [
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(original_wav),
            "-i",
            str(narration_wav),
        ]
        filters: list[str] = []

        mode = settings.mode
        final_label: str

        if mode == OriginalAudioMode.NARRATION_ONLY or mode == OriginalAudioMode.REPLACE:
            # Mix contains only narration.
            filters.append(
                f"[1:a]{fmt},volume={narration_db:.2f}dB,"
                f"atrim=0:{duration_seconds:.3f},apad=whole_dur={duration_seconds:.3f},"
                f"atrim=0:{duration_seconds:.3f},asetpts=N/SR/TB[mix]"
            )
            final_label = "mix"
        elif mode == OriginalAudioMode.CONSTANT:
            filters.append(
                f"[0:a]{fmt},volume={original_during_db:.2f}dB,"
                f"atrim=0:{duration_seconds:.3f},asetpts=N/SR/TB[orig]"
            )
            filters.append(
                f"[1:a]{fmt},volume={narration_db:.2f}dB,"
                f"atrim=0:{duration_seconds:.3f},asetpts=N/SR/TB[narr]"
            )
            filters.append(
                "[orig][narr]amix=inputs=2:duration=first:normalize=0[mix]"
            )
            final_label = "mix"
        else:
            # dynamic_ducking
            arguments.extend(["-i", str(mask_path)])
            filters.append(
                f"[0:a]{fmt},volume={original_outside_db:.2f}dB,"
                f"atrim=0:{duration_seconds:.3f},asetpts=N/SR/TB[orig_base]"
            )
            filters.append(
                "[orig_base][2:a]"
                + self._ducking_filter(original_during_db)
                + "[ducked]"
            )
            filters.append(
                f"[1:a]{fmt},volume={narration_db:.2f}dB,"
                f"atrim=0:{duration_seconds:.3f},asetpts=N/SR/TB[narr]"
            )
            filters.append(
                "[ducked][narr]amix=inputs=2:duration=first:normalize=0[mix]"
            )
            final_label = "mix"

        if settings.normalize:
            filters.append(
                f"[{final_label}]loudnorm=I={settings.target_lufs:.1f}:LRA=11:"
                f"TP={settings.true_peak_db:.1f}[norm]"
            )
            final_label = "norm"

        filters.append(
            f"[{final_label}]alimiter=limit=0.95,"
            f"atrim=0:{duration_seconds:.3f},asetpts=N/SR/TB[out]"
        )
        arguments.extend(["-filter_complex", ";".join(filters), "-map", "[out]"])
        arguments.extend(
            [
                "-codec:a",
                "pcm_s16le",
                "-ar",
                str(self.sample_rate),
                "-ac",
                str(self.channels),
            ]
        )
        return arguments

    @staticmethod
    def _ducking_filter(original_during_db: float) -> str:
        duck_db = abs(original_during_db)
        ratio = 20.0
        if duck_db <= 0:
            return "anull"
        threshold_db = -duck_db / (1.0 - 1.0 / ratio)
        threshold = max(0.000976, min(1.0, 10 ** (threshold_db / 20)))
        return (
            "sidechaincompress="
            f"threshold={threshold:.6f}:ratio={ratio:.1f}:attack=0.01:release=0.01,"
            "aformat=sample_fmts=fltp"
        )


def resolve_ffmpeg(ffmpeg_path: str | Path) -> Path:
    return find_ffmpeg_sibling(ffmpeg_path)
