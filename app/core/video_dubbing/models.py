from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CueStatus(str, Enum):
    PENDING = "pending"
    RENDERING = "rendering"
    RENDERED = "rendered"
    FITTED = "fitted"
    SPEED_UP = "speed_up"
    STRONG_SPEED_UP = "strong_speed_up"
    EXTREME_SPEED_REQUIRED = "extreme_speed_required"
    NEEDS_SHORTENING = "needs_text_shortening"
    FAILED = "failed"
    MISSING_AUDIO = "missing_audio"
    DISABLED = "disabled"
    STALE = "stale"


# Statuses that are considered "ready" and must not trigger TTS again.
READY_STATUSES = frozenset(
    {
        CueStatus.RENDERED.value,
        CueStatus.FITTED.value,
        CueStatus.SPEED_UP.value,
        CueStatus.STRONG_SPEED_UP.value,
    }
)

# Statuses that always block export regardless of policy.
BLOCKING_STATUSES = frozenset(
    {
        CueStatus.FAILED.value,
        CueStatus.MISSING_AUDIO.value,
        CueStatus.EXTREME_SPEED_REQUIRED.value,
    }
)


class FittingStrategy(str, Enum):
    NONE = "none"
    NATIVE_SPEED = "native_speed"
    ATEMPO = "atempo"
    NATIVE_THEN_ATEMPO = "native_then_atempo"
    BEST_EFFORT = "best_effort"
    CLIP = "clip"


class OriginalAudioMode(str, Enum):
    REPLACE = "replace_original"
    CONSTANT = "constant_overlay"
    DUCKING = "dynamic_ducking"
    NARRATION_ONLY = "narration_only"


class SyncMode(str, Enum):
    STRICT = "strict"
    BEST_EFFORT = "best_effort"


class Alignment(str, Enum):
    START = "start"
    CENTER = "center"
    END = "end"


class OutputContainer(str, Enum):
    MP4 = "mp4"
    MKV = "mkv"


@dataclass
class DubbingCue:
    cue_id: str
    sequence: int

    start_ms: int
    end_ms: int
    duration_budget_ms: int

    source_text: str
    spoken_text: str

    raw_audio_path: Path | None = None
    fitted_audio_path: Path | None = None

    raw_duration_ms: int | None = None
    fitted_duration_ms: int | None = None

    required_speed_factor: float | None = None
    applied_speed_factor: float = 1.0

    placement_offset_ms: int = 0
    overflow_ms: int = 0

    # New timing fields populated by the fitter.
    target_duration_ms: int | None = None
    safe_end_ms: int | None = None
    timing_diff_ms: int | None = None

    status: str = CueStatus.PENDING.value
    fitting_strategy: str | None = None
    warning_codes: list[str] = field(default_factory=list)
    error_message: str | None = None

    enabled: bool = True
    is_stale: bool = False

    attempt_count: int = 0
    native_speed_factor: float | None = None

    # Fingerprint of the inputs that produced the current raw audio. When it
    # matches the current inputs and the WAV is intact, the cue is reused
    # instead of being re-synthesized.
    generation_fingerprint: str = ""
    # True when the raw audio exists but its provenance (engine/voice/text used
    # at creation time) could not be verified — e.g. cues generated before
    # fingerprinting. Such cues may be re-fit from the existing raw WAV, but a
    # change of voice or text MUST trigger TTS, and a confirmed fingerprint is
    # only recorded after a real synthesis.
    legacy_audio_unverified: bool = False
    # Snapshot of the inputs observed when unverified legacy audio was first
    # migrated. It detects later edits without claiming those inputs produced
    # the raw WAV.
    legacy_observed_fingerprint: str = ""
    # Per-cue override of the hard speed limit (0 = use project default).
    hard_speed_override: float = 0.0
    # Per-cue "force fit even beyond hard limit" flag.
    force_fit: bool = False

    def mark_stale(self) -> None:
        self.is_stale = True
        if self.status not in {
            CueStatus.FAILED.value,
            CueStatus.EXTREME_SPEED_REQUIRED.value,
        }:
            self.status = CueStatus.STALE.value


@dataclass
class VideoProbeInfo:
    duration_ms: int
    video_codec: str = ""
    audio_codec: str = ""
    audio_tracks: int = 0
    sample_rate: int = 0
    channels: int = 0
    fps: float = 0.0
    container_format: str = ""
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "audio_tracks": self.audio_tracks,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "fps": self.fps,
            "container_format": self.container_format,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoProbeInfo:
        return cls(
            duration_ms=int(data.get("duration_ms", 0) or 0),
            video_codec=str(data.get("video_codec", "") or ""),
            audio_codec=str(data.get("audio_codec", "") or ""),
            audio_tracks=int(data.get("audio_tracks", 0) or 0),
            sample_rate=int(data.get("sample_rate", 0) or 0),
            channels=int(data.get("channels", 0) or 0),
            fps=float(data.get("fps", 0.0) or 0.0),
            container_format=str(data.get("container_format", "") or ""),
            width=int(data.get("width", 0) or 0),
            height=int(data.get("height", 0) or 0),
        )


@dataclass
class DuckingSettings:
    mode: OriginalAudioMode = OriginalAudioMode.DUCKING
    original_outside_percent: float = 100.0
    original_during_percent: float = 15.0
    narration_volume_percent: float = 100.0
    attack_ms: int = 100
    release_ms: int = 250
    target_lufs: float = -16.0
    true_peak_db: float = -1.0
    normalize: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "original_outside_percent": self.original_outside_percent,
            "original_during_percent": self.original_during_percent,
            "narration_volume_percent": self.narration_volume_percent,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "target_lufs": self.target_lufs,
            "true_peak_db": self.true_peak_db,
            "normalize": self.normalize,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DuckingSettings:
        mode_value = str(data.get("mode", OriginalAudioMode.DUCKING.value))
        try:
            mode = OriginalAudioMode(mode_value)
        except ValueError:
            mode = OriginalAudioMode.DUCKING
        return cls(
            mode=mode,
            original_outside_percent=float(
                data.get("original_outside_percent", 100.0)
            ),
            original_during_percent=float(
                data.get("original_during_percent", 15.0)
            ),
            narration_volume_percent=float(
                data.get("narration_volume_percent", 100.0)
            ),
            attack_ms=int(data.get("attack_ms", 100)),
            release_ms=int(data.get("release_ms", 250)),
            target_lufs=float(data.get("target_lufs", -16.0)),
            true_peak_db=float(data.get("true_peak_db", -1.0)),
            normalize=bool(data.get("normalize", False)),
        )


@dataclass
class PreviewSettings:
    pre_roll_ms: int = 1000
    post_roll_ms: int = 1000
    alignment: Alignment = Alignment.START
    crossover_mode: str = "block"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_roll_ms": self.pre_roll_ms,
            "post_roll_ms": self.post_roll_ms,
            "alignment": self.alignment.value,
            "crossover_mode": self.crossover_mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreviewSettings:
        alignment_value = str(data.get("alignment", Alignment.START.value))
        try:
            alignment = Alignment(alignment_value)
        except ValueError:
            alignment = Alignment.START
        return cls(
            pre_roll_ms=int(data.get("pre_roll_ms", 1000)),
            post_roll_ms=int(data.get("post_roll_ms", 1000)),
            alignment=alignment,
            crossover_mode=str(data.get("crossover_mode", "block")),
        )


@dataclass
class ExportSettings:
    container: OutputContainer = OutputContainer.MKV
    include_narration_only: bool = True
    embed_subtitles: bool = False
    video_codec_override: str = ""
    audio_bitrate: str = "192k"
    force_video_reencode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "container": self.container.value,
            "include_narration_only": self.include_narration_only,
            "embed_subtitles": self.embed_subtitles,
            "video_codec_override": self.video_codec_override,
            "audio_bitrate": self.audio_bitrate,
            "force_video_reencode": self.force_video_reencode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportSettings:
        container_value = str(data.get("container", OutputContainer.MKV.value))
        try:
            container = OutputContainer(container_value)
        except ValueError:
            container = OutputContainer.MKV
        return cls(
            container=container,
            include_narration_only=bool(data.get("include_narration_only", True)),
            embed_subtitles=bool(data.get("embed_subtitles", False)),
            video_codec_override=str(data.get("video_codec_override", "")),
            audio_bitrate=str(data.get("audio_bitrate", "192k")),
            force_video_reencode=bool(data.get("force_video_reencode", False)),
        )


@dataclass
class DubbingProjectSettings:
    language: str = ""
    tts_engine: str = "piper"
    voice: str = ""
    voice_config: dict[str, Any] = field(default_factory=dict)
    # Deprecated single limit kept for backward compatibility; mirrors
    # preferred_speed_limit when loading old manifests.
    max_speed_factor: float = 1.35
    preferred_speed_limit: float = 1.35
    hard_speed_limit: float = 2.50
    exact_timing: bool = True
    guard_gap_ms: int = 20
    compress_internal_pauses: bool = False
    internal_pause_keep_ms: int = 90
    sync_mode: SyncMode = SyncMode.STRICT
    ffmpeg_path: str = ""
    sample_rate: int = 48000
    channels: int = 2
    ducking: DuckingSettings = field(default_factory=DuckingSettings)
    preview: PreviewSettings = field(default_factory=PreviewSettings)
    export: ExportSettings = field(default_factory=ExportSettings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "tts_engine": self.tts_engine,
            "voice": self.voice,
            "voice_config": dict(self.voice_config),
            "max_speed_factor": self.max_speed_factor,
            "preferred_speed_limit": self.preferred_speed_limit,
            "hard_speed_limit": self.hard_speed_limit,
            "exact_timing": self.exact_timing,
            "guard_gap_ms": self.guard_gap_ms,
            "compress_internal_pauses": self.compress_internal_pauses,
            "internal_pause_keep_ms": self.internal_pause_keep_ms,
            "sync_mode": self.sync_mode.value,
            "ffmpeg_path": self.ffmpeg_path,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "ducking": self.ducking.to_dict(),
            "preview": self.preview.to_dict(),
            "export": self.export.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DubbingProjectSettings:
        sync_value = str(data.get("sync_mode", SyncMode.STRICT.value))
        try:
            sync_mode = SyncMode(sync_value)
        except ValueError:
            sync_mode = SyncMode.STRICT
        voice_config = data.get("voice_config", {})
        if not isinstance(voice_config, dict):
            voice_config = {}
        legacy_max = float(data.get("max_speed_factor", 1.35))
        preferred = float(data.get("preferred_speed_limit", legacy_max))
        hard = float(data.get("hard_speed_limit", max(2.5, preferred)))
        if hard < preferred:
            hard = preferred
        return cls(
            language=str(data.get("language", "")),
            tts_engine=str(data.get("tts_engine", "piper")),
            voice=str(data.get("voice", "")),
            voice_config=dict(voice_config),
            max_speed_factor=preferred,
            preferred_speed_limit=preferred,
            hard_speed_limit=hard,
            exact_timing=bool(data.get("exact_timing", True)),
            guard_gap_ms=int(data.get("guard_gap_ms", 20)),
            compress_internal_pauses=bool(data.get("compress_internal_pauses", False)),
            internal_pause_keep_ms=int(data.get("internal_pause_keep_ms", 90)),
            sync_mode=sync_mode,
            ffmpeg_path=str(data.get("ffmpeg_path", "")),
            sample_rate=int(data.get("sample_rate", 48000)),
            channels=int(data.get("channels", 2)),
            ducking=DuckingSettings.from_dict(data.get("ducking", {})),
            preview=PreviewSettings.from_dict(data.get("preview", {})),
            export=ExportSettings.from_dict(data.get("export", {})),
        )


@dataclass
class StaleFlags:
    cues: bool = True
    narration: bool = True
    mix: bool = True
    preview: bool = True
    video: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {
            "cues": self.cues,
            "narration": self.narration,
            "mix": self.mix,
            "preview": self.preview,
            "video": self.video,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StaleFlags:
        return cls(
            cues=bool(data.get("cues", True)),
            narration=bool(data.get("narration", True)),
            mix=bool(data.get("mix", True)),
            preview=bool(data.get("preview", True)),
            video=bool(data.get("video", True)),
        )


@dataclass
class DubbingProject:
    project_id: str
    project_dir: Path
    title: str = "Video Dubbing"
    video_path: Path | None = None
    video_probe: VideoProbeInfo | None = None
    srt_source_path: Path | None = None
    srt_source_text: str = ""
    settings: DubbingProjectSettings = field(default_factory=DubbingProjectSettings)
    cues: list[DubbingCue] = field(default_factory=list)
    stale: StaleFlags = field(default_factory=StaleFlags)
    created_at: str = ""
    updated_at: str = ""

    narration_wav: Path | None = None
    narration_mp3: Path | None = None
    dubbed_mix_wav: Path | None = None
    final_video_path: Path | None = None
    full_preview_path: Path | None = None
    cue_preview_cache: dict[str, Path] = field(default_factory=dict)

    # UI / playback restoration state.
    selected_sequence: int | None = None
    last_player_position_ms: int = 0
    schema_version: int = 2

    @property
    def duration_ms(self) -> int:
        if self.video_probe is not None:
            return self.video_probe.duration_ms
        return 0

    def source_dir(self) -> Path:
        return self.project_dir / "source"

    def cues_dir(self) -> Path:
        return self.project_dir / "cues"

    def preview_dir(self) -> Path:
        return self.project_dir / "preview"

    def render_dir(self) -> Path:
        return self.project_dir / "render"

    def reports_dir(self) -> Path:
        return self.project_dir / "reports"

    def temp_dir(self) -> Path:
        return self.project_dir / "temp"

    def cue_raw_path(self, sequence: int) -> Path:
        return self.cues_dir() / f"cue_{sequence:06d}_raw.wav"

    def cue_fitted_path(self, sequence: int) -> Path:
        return self.cues_dir() / f"cue_{sequence:06d}_fitted.wav"

    def cue_preview_path(self, sequence: int) -> Path:
        return self.preview_dir() / f"cue_{sequence:06d}_preview.mkv"

    def ensure_directories(self) -> None:
        for directory in (
            self.source_dir(),
            self.cues_dir(),
            self.preview_dir(),
            self.render_dir(),
            self.reports_dir(),
            self.temp_dir(),
        ):
            directory.mkdir(parents=True, exist_ok=True)
