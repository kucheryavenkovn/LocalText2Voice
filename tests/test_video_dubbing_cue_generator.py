from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

from app.core.video_dubbing.cue_generator import (
    CueGenerator,
    CueGenerationConfig,
)
from app.core.video_dubbing.duration_fitter import DurationFitter
from app.core.video_dubbing.models import (
    CueStatus,
    DubbingCue,
    DubbingProjectSettings,
    FittingStrategy,
    SyncMode,
)
from app.tts.base import BaseTTSEngine

FFMPEG_EXE = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(
    not FFMPEG_EXE,
    reason="ffmpeg not available on PATH",
)


class FakeToneTTS(BaseTTSEngine):
    """Generates a deterministic tone WAV of a target duration using ffmpeg."""

    def __init__(self, ffmpeg_exe: str, duration_seconds: float) -> None:
        self.ffmpeg_exe = ffmpeg_exe
        self.duration_seconds = duration_seconds

    def validate(self, voice_config: dict[str, Any]) -> None:
        return None

    def synthesize_to_wav(
        self,
        text: str,
        output_wav: Path,
        voice_config: dict[str, Any],
    ) -> Path:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={self.duration_seconds:.3f}",
            "-ar",
            "22050",
            "-ac",
            "1",
            "-codec:a",
            "pcm_s16le",
            str(output_wav),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not output_wav.is_file():
            raise RuntimeError("fake tts failed")
        return output_wav

    def cancel_current(self) -> None:
        return None


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as audio:
        return int(round(audio.getnframes() / audio.getframerate() * 1000))


def _make_cue(sequence: int, start_ms: int, end_ms: int, cues_dir: Path) -> DubbingCue:
    return DubbingCue(
        cue_id=str(sequence),
        sequence=sequence,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_budget_ms=end_ms - start_ms,
        source_text=f"text {sequence}",
        spoken_text=f"text {sequence}",
        raw_audio_path=cues_dir / f"cue_{sequence:06d}_raw.wav",
        fitted_audio_path=cues_dir / f"cue_{sequence:06d}_fitted.wav",
    )


def test_generate_raw_measures_and_normalizes(tmp_path):
    tts = FakeToneTTS(FFMPEG_EXE, duration_seconds=1.5)
    gen = CueGenerator(
        tts,
        ffmpeg_path="ffmpeg/ffmpeg.exe",
        config=CueGenerationConfig(sample_rate=48000, channels=2),
    )
    cue = _make_cue(1, 0, 3000, tmp_path)
    gen.generate_raw(cue, voice_config={"engine": "fake"})
    assert cue.status == CueStatus.RENDERED.value
    assert cue.raw_duration_ms is not None
    assert 1400 <= cue.raw_duration_ms <= 1600
    with wave.open(str(cue.raw_audio_path), "rb") as audio:
        assert audio.getframerate() == 48000
        assert audio.getnchannels() == 2


def test_generate_raw_empty_text_fails(tmp_path):
    tts = FakeToneTTS(FFMPEG_EXE, duration_seconds=0.5)
    gen = CueGenerator(tts, ffmpeg_path="ffmpeg/ffmpeg.exe")
    cue = _make_cue(1, 0, 1000, tmp_path)
    cue.spoken_text = "   "
    with pytest.raises(Exception):
        gen.generate_raw(cue, voice_config={})
    assert cue.status == CueStatus.FAILED.value


def test_apply_fitting_short_cue_copies_raw(tmp_path):
    tts = FakeToneTTS(FFMPEG_EXE, duration_seconds=1.5)
    gen = CueGenerator(tts, ffmpeg_path="ffmpeg/ffmpeg.exe")
    settings = DubbingProjectSettings(max_speed_factor=1.35)
    fitter = DurationFitter(settings)
    cue = _make_cue(1, 0, 3000, tmp_path)
    gen.generate_raw(cue, voice_config={})
    result = fitter.evaluate(cue)
    assert result.strategy == FittingStrategy.NONE
    gen.apply_fitting(cue, result)
    assert cue.fitted_audio_path.is_file()
    assert cue.fitted_duration_ms == cue.raw_duration_ms


def test_apply_fitting_mild_speedup(tmp_path):
    tts = FakeToneTTS(FFMPEG_EXE, duration_seconds=4.1)
    gen = CueGenerator(tts, ffmpeg_path="ffmpeg/ffmpeg.exe")
    settings = DubbingProjectSettings(max_speed_factor=1.35)
    fitter = DurationFitter(settings)
    cue = _make_cue(1, 0, 3500, tmp_path)
    gen.generate_raw(cue, voice_config={})
    result = fitter.evaluate(cue)
    assert result.strategy == FittingStrategy.ATEMPO
    assert result.applied_speed_factor == pytest.approx(1.17, rel=2e-2)
    gen.apply_fitting(cue, result)
    assert cue.fitted_audio_path.is_file()
    fitted_ms = _wav_duration_ms(cue.fitted_audio_path)
    # 4.1s raw sped up by ~1.17 -> ~3.5s fitted
    assert 3300 <= fitted_ms <= 3700


def test_apply_fitting_strict_overflow_marks_shortening(tmp_path):
    tts = FakeToneTTS(FFMPEG_EXE, duration_seconds=3.8)
    gen = CueGenerator(tts, ffmpeg_path="ffmpeg/ffmpeg.exe")
    settings = DubbingProjectSettings(
        max_speed_factor=1.35, sync_mode=SyncMode.STRICT
    )
    fitter = DurationFitter(settings)
    cue = _make_cue(1, 0, 2000, tmp_path)
    gen.generate_raw(cue, voice_config={})
    result = fitter.evaluate(cue)
    assert result.status == CueStatus.NEEDS_SHORTENING.value
    assert result.overflow_ms > 0
    fitter.apply_result(cue, result)
    assert cue.overflow_ms > 0
    # Best-effort fit still produces a file but overflow remains.
    gen.apply_fitting(cue, result)
    assert cue.fitted_audio_path.is_file()
    assert cue.overflow_ms > 0
