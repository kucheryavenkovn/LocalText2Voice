from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from array import array

import pytest

from app.core.video_dubbing.audio_mixer import AudioMixer, MASK_SAMPLE_RATE
from app.core.video_dubbing.models import (
    DubbingCue,
    DubbingProject,
    DuckingSettings,
    OriginalAudioMode,
    VideoProbeInfo,
)

FFMPEG_EXE = shutil.which("ffmpeg")


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as audio:
        return int(round(audio.getnframes() / audio.getframerate() * 1000))


def _read_mask_samples(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as audio:
        frames = audio.readframes(audio.getnframes())
    samples = array("h")
    samples.frombytes(frames)
    return list(samples)


def _cue(start_ms, end_ms, fitted_ms, sequence=1):
    return DubbingCue(
        cue_id=str(sequence),
        sequence=sequence,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_budget_ms=end_ms - start_ms,
        source_text="x",
        spoken_text="x",
        fitted_duration_ms=fitted_ms,
        status="fitted",
    )


# ---------------------------------------------------------------------------
# Duck mask (deterministic, no ffmpeg)
# ---------------------------------------------------------------------------


def test_mask_outside_cue_full_zero_inside_high():
    mixer = AudioMixer("ffmpeg/ffmpeg.exe")
    cues = [_cue(2000, 4000, 2000)]
    out = Path("/tmp/mask_test.wav") if False else None
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "mask.wav"
    mixer.build_duck_mask(cues, 6000, DuckingSettings(), tmp)
    samples = _read_mask_samples(tmp)
    # outside narration: 0
    assert samples[0] == 0
    # inside narration: full
    mid = int((3000 / 1000) * MASK_SAMPLE_RATE)
    assert samples[mid] == 32767
    assert len(samples) == 6 * MASK_SAMPLE_RATE


def test_mask_has_attack_and_release_ramps():
    mixer = AudioMixer("ffmpeg/ffmpeg.exe")
    settings = DuckingSettings(attack_ms=200, release_ms=300)
    cues = [_cue(2000, 4000, 2000)]
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "mask.wav"
    mixer.build_duck_mask(cues, 6000, settings, tmp)
    samples = _read_mask_samples(tmp)
    # ramp up occurs before 2000ms boundary
    pre_attack_idx = int((1950 / 1000) * MASK_SAMPLE_RATE)
    assert 0 < samples[pre_attack_idx] < 32767
    # ramp down after 4000ms boundary
    post_release_idx = int((4150 / 1000) * MASK_SAMPLE_RATE)
    assert 0 < samples[post_release_idx] < 32767


def test_mask_merges_overlapping_intervals():
    mixer = AudioMixer("ffmpeg/ffmpeg.exe")
    cues = [_cue(2000, 4000, 2000, 1), _cue(3500, 5000, 1500, 2)]
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "mask.wav"
    mixer.build_duck_mask(cues, 6000, DuckingSettings(), tmp)
    intervals = mixer.narration_intervals(cues)
    assert intervals == [(2000, 5000)]


def test_mask_exact_length_matches_duration():
    mixer = AudioMixer("ffmpeg/ffmpeg.exe")
    cues = [_cue(1000, 2000, 1000)]
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "mask.wav"
    mixer.build_duck_mask(cues, 7500, DuckingSettings(), tmp)
    samples = _read_mask_samples(tmp)
    assert len(samples) == 7500 * MASK_SAMPLE_RATE // 1000


def test_mask_disabled_cue_excluded():
    mixer = AudioMixer("ffmpeg/ffmpeg.exe")
    cue = _cue(1000, 2000, 1000)
    cue.enabled = False
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "mask.wav"
    mixer.build_duck_mask([cue], 3000, DuckingSettings(), tmp)
    samples = _read_mask_samples(tmp)
    assert all(s == 0 for s in samples)


def test_narration_intervals_uses_fitted_duration():
    mixer = AudioMixer("ffmpeg/ffmpeg.exe")
    cue = _cue(1000, 3000, 1500)  # fitted shorter than budget
    intervals = mixer.narration_intervals([cue])
    assert intervals == [(1000, 2500)]


# ---------------------------------------------------------------------------
# Full mix (ffmpeg integration)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg not available")
def test_render_dubbed_mix_dynamic_ducking(tmp_path):
    # build a fake video with audio tone, fake narration wav
    video_path = tmp_path / "video.mp4"
    subprocess.run(
        [
            FFMPEG_EXE,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=6",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:d=6",
            "-shortest",
            "-c:a",
            "aac",
            "-c:v",
            "libx264",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    narration = tmp_path / "narration.wav"
    subprocess.run(
        [
            FFMPEG_EXE,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t",
            "6",
            str(narration),
        ],
        check=True,
        capture_output=True,
    )
    project = DubbingProject(
        project_id="m",
        project_dir=tmp_path / "proj",
        video_path=video_path,
        video_probe=VideoProbeInfo(duration_ms=6000),
    )
    project.ensure_directories()
    project.cues.append(_cue(2000, 4000, 2000))
    mixer = AudioMixer("ffmpeg/ffmpeg.exe")
    out = tmp_path / "mix.wav"
    mixer.render_dubbed_mix(project, narration, out)
    assert out.is_file()
    assert abs(_wav_duration_ms(out) - 6000) <= 40


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg not available")
def test_render_dubbed_mix_replace_mode_no_original(tmp_path):
    video_path = tmp_path / "video.mp4"
    subprocess.run(
        [
            FFMPEG_EXE,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=5",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:d=5",
            "-shortest",
            "-c:a",
            "aac",
            "-c:v",
            "libx264",
            str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    narration = tmp_path / "narration.wav"
    subprocess.run(
        [
            FFMPEG_EXE,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t",
            "5",
            str(narration),
        ],
        check=True,
        capture_output=True,
    )
    project = DubbingProject(
        project_id="m",
        project_dir=tmp_path / "proj",
        video_path=video_path,
        video_probe=VideoProbeInfo(duration_ms=5000),
    )
    project.ensure_directories()
    project.settings.ducking.mode = OriginalAudioMode.REPLACE
    project.cues.append(_cue(1000, 3000, 2000))
    mixer = AudioMixer("ffmpeg/ffmpeg.exe")
    out = tmp_path / "mix.wav"
    mixer.render_dubbed_mix(project, narration, out)
    assert abs(_wav_duration_ms(out) - 5000) <= 40


@pytest.mark.skipif(not FFMPEG_EXE, reason="ffmpeg not available")
def test_encode_narration_mp3(tmp_path):
    wav = tmp_path / "n.wav"
    subprocess.run(
        [
            FFMPEG_EXE,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            str(wav),
        ],
        check=True,
        capture_output=True,
    )
    mixer = AudioMixer("ffmpeg/ffmpeg.exe")
    out = tmp_path / "n.mp3"
    mixer.encode_narration_mp3(wav, out)
    assert out.is_file() and out.stat().st_size > 0
