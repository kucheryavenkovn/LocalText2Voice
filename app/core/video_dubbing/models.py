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
    ELASTIC_GROUP_FAILED = "elastic_group_failed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MISSING_AUDIO = "missing_audio"
    DISABLED = "disabled"
    STALE = "stale"


class CueTimingMode(str, Enum):
    STRICT = "strict"
    ELASTIC_GROUP = "elastic_group"


class TempoSmoothingMode(str, Enum):
    OFF = "off"
    SMOOTH = "smooth"
    COMMON_GROUP_FACTOR = "common_group_factor"


TEMPO_SMOOTHING_MODE_LABELS_RU = {
    TempoSmoothingMode.OFF.value: "Выключено",
    TempoSmoothingMode.SMOOTH.value: "Сглаживать соседние реплики",
    TempoSmoothingMode.COMMON_GROUP_FACTOR.value: "Единый темп для группы",
}


@dataclass
class TempoSmoothingSettings:
    mode: TempoSmoothingMode = TempoSmoothingMode.SMOOTH
    max_cues_per_group: int = 3
    speed_jump_threshold: float = 0.10
    max_neighbor_speed_delta: float = 0.08
    max_speed_factor: float = 1.35
    min_speed_factor: float = 1.0
    max_optional_speedup: float = 0.20
    include_fitting_cues: bool = True
    include_non_fitting_cues: bool = True
    prefer_natural_speed: bool = True
    preserve_locked_cues: bool = True
    auto_apply_after_generate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "max_cues_per_group": self.max_cues_per_group,
            "speed_jump_threshold": self.speed_jump_threshold,
            "max_neighbor_speed_delta": self.max_neighbor_speed_delta,
            "max_speed_factor": self.max_speed_factor,
            "min_speed_factor": self.min_speed_factor,
            "max_optional_speedup": self.max_optional_speedup,
            "include_fitting_cues": self.include_fitting_cues,
            "include_non_fitting_cues": self.include_non_fitting_cues,
            "prefer_natural_speed": self.prefer_natural_speed,
            "preserve_locked_cues": self.preserve_locked_cues,
            "auto_apply_after_generate": self.auto_apply_after_generate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TempoSmoothingSettings:
        data = data or {}
        mode_raw = str(data.get("mode", TempoSmoothingMode.SMOOTH.value) or TempoSmoothingMode.SMOOTH.value)
        try:
            mode = TempoSmoothingMode(mode_raw)
        except ValueError:
            mode = TempoSmoothingMode.SMOOTH
        max_cues = int(data.get("max_cues_per_group", 3) or 3)
        max_cues = max(2, min(5, max_cues))
        return cls(
            mode=mode,
            max_cues_per_group=max_cues,
            speed_jump_threshold=float(data.get("speed_jump_threshold", 0.10) or 0.10),
            max_neighbor_speed_delta=float(
                data.get("max_neighbor_speed_delta", 0.08) or 0.08
            ),
            max_speed_factor=float(data.get("max_speed_factor", 1.35) or 1.35),
            min_speed_factor=float(data.get("min_speed_factor", 1.0) or 1.0),
            max_optional_speedup=float(data.get("max_optional_speedup", 0.20) or 0.20),
            include_fitting_cues=bool(data.get("include_fitting_cues", True)),
            include_non_fitting_cues=bool(data.get("include_non_fitting_cues", True)),
            prefer_natural_speed=bool(data.get("prefer_natural_speed", True)),
            preserve_locked_cues=bool(data.get("preserve_locked_cues", True)),
            auto_apply_after_generate=bool(data.get("auto_apply_after_generate", True)),
        )


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


ORIGINAL_AUDIO_MODE_LABELS_RU = {
    OriginalAudioMode.REPLACE.value: "Заменить оригинальный звук",
    OriginalAudioMode.CONSTANT.value: "Постоянное наложение",
    OriginalAudioMode.DUCKING.value: "Динамическое приглушение",
    OriginalAudioMode.NARRATION_ONLY.value: "Только перевод",
}


@dataclass
class ElasticTimingSettings:
    enabled: bool = False
    max_cues_per_group: int = 3
    preserve_first_cue_start: bool = True
    shift_direction: str = "right_only"
    min_inter_cue_gap_ms: int = 120
    boundary_guard_ms: int = 100
    max_shift_per_cue_ms: int = 2000
    max_group_extension_ms: int = 5000
    use_internal_gaps: bool = True
    use_trailing_silence: bool = True
    common_speed_factor: bool = True
    max_common_speed_factor: float = 1.35
    prefer_smaller_group: bool = False
    group_locked: bool = False
    # Automatic neighbor tempo smoothing after TTS/refit. Works even when the
    # full elastic timing mode is off — only problematic cues borrow right time.
    auto_smooth_neighbors: bool = True
    auto_apply_after_generate: bool = True
    # Trigger smoothing when required speed exceeds this (defaults to preferred).
    smooth_speed_threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_cues_per_group": self.max_cues_per_group,
            "preserve_first_cue_start": self.preserve_first_cue_start,
            "shift_direction": self.shift_direction,
            "min_inter_cue_gap_ms": self.min_inter_cue_gap_ms,
            "boundary_guard_ms": self.boundary_guard_ms,
            "max_shift_per_cue_ms": self.max_shift_per_cue_ms,
            "max_group_extension_ms": self.max_group_extension_ms,
            "use_internal_gaps": self.use_internal_gaps,
            "use_trailing_silence": self.use_trailing_silence,
            "common_speed_factor": self.common_speed_factor,
            "max_common_speed_factor": self.max_common_speed_factor,
            "prefer_smaller_group": self.prefer_smaller_group,
            "group_locked": self.group_locked,
            "auto_smooth_neighbors": self.auto_smooth_neighbors,
            "auto_apply_after_generate": self.auto_apply_after_generate,
            "smooth_speed_threshold": self.smooth_speed_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ElasticTimingSettings:
        data = data or {}
        max_cues = int(data.get("max_cues_per_group", 3) or 3)
        max_cues = max(1, min(5, max_cues))
        return cls(
            enabled=bool(data.get("enabled", False)),
            max_cues_per_group=max_cues,
            preserve_first_cue_start=bool(data.get("preserve_first_cue_start", True)),
            shift_direction=str(data.get("shift_direction", "right_only") or "right_only"),
            min_inter_cue_gap_ms=int(data.get("min_inter_cue_gap_ms", 120) or 120),
            boundary_guard_ms=int(data.get("boundary_guard_ms", 100) or 100),
            max_shift_per_cue_ms=int(data.get("max_shift_per_cue_ms", 2000) or 2000),
            max_group_extension_ms=int(data.get("max_group_extension_ms", 5000) or 5000),
            use_internal_gaps=bool(data.get("use_internal_gaps", True)),
            use_trailing_silence=bool(data.get("use_trailing_silence", True)),
            common_speed_factor=bool(data.get("common_speed_factor", True)),
            max_common_speed_factor=float(data.get("max_common_speed_factor", 1.35) or 1.35),
            prefer_smaller_group=bool(data.get("prefer_smaller_group", False)),
            group_locked=bool(data.get("group_locked", False)),
            auto_smooth_neighbors=bool(data.get("auto_smooth_neighbors", True)),
            auto_apply_after_generate=bool(data.get("auto_apply_after_generate", True)),
            smooth_speed_threshold=float(data.get("smooth_speed_threshold", 0.0) or 0.0),
        )


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
    planned_speed_factor: float | None = None
    applied_speed_factor: float = 1.0
    smoothing_group_id: str | None = None
    smoothing_reason: str = ""

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
    # Fit-level fingerprint (raw identity + timing/fit settings).
    fit_fingerprint: str = ""
    fit_pipeline_version: int = 0
    raw_wav_hash: str = ""
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

    # Source SRT timing (immutable after import) and planned elastic timing.
    source_start_ms: int | None = None
    source_end_ms: int | None = None
    planned_start_ms: int | None = None
    planned_end_ms: int | None = None
    timing_group_id: str | None = None
    timing_group_position: int | None = None
    common_speed_factor: float | None = None
    start_shift_ms: int = 0
    end_shift_ms: int = 0
    borrowed_right_ms: int = 0
    timing_locked: bool = False

    def effective_start_ms(self) -> int:
        if self.planned_start_ms is not None:
            return self.planned_start_ms
        return self.start_ms

    def effective_end_ms(self) -> int:
        if self.planned_end_ms is not None:
            return self.planned_end_ms
        return self.end_ms

    def ensure_source_timing(self) -> None:
        if self.source_start_ms is None:
            self.source_start_ms = self.start_ms
        if self.source_end_ms is None:
            self.source_end_ms = self.end_ms

    def mark_stale(self) -> None:
        self.is_stale = True
        if self.status not in {
            CueStatus.FAILED.value,
            CueStatus.EXTREME_SPEED_REQUIRED.value,
            CueStatus.CANCELLED.value,
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
    autoplay_on_select: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_roll_ms": self.pre_roll_ms,
            "post_roll_ms": self.post_roll_ms,
            "alignment": self.alignment.value,
            "crossover_mode": self.crossover_mode,
            "autoplay_on_select": self.autoplay_on_select,
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
            autoplay_on_select=bool(data.get("autoplay_on_select", False)),
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
    cue_timing_mode: CueTimingMode = CueTimingMode.STRICT
    elastic_timing: ElasticTimingSettings = field(default_factory=ElasticTimingSettings)
    tempo_smoothing: TempoSmoothingSettings = field(default_factory=TempoSmoothingSettings)
    ffmpeg_path: str = ""
    sample_rate: int = 48000
    channels: int = 2
    ducking: DuckingSettings = field(default_factory=DuckingSettings)
    preview: PreviewSettings = field(default_factory=PreviewSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    # Default STOP preserves historical checkpoint/resume behaviour; UI may
    # switch to CONTINUE for batch "generate missing".
    cue_error_policy: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "tts_engine": self.tts_engine,
            "voice": self.voice,
            "voice_config": dict(self.voice_config),
            "max_speed_factor": self.max_speed_limit_compat(),
            "preferred_speed_limit": self.preferred_speed_limit,
            "hard_speed_limit": self.hard_speed_limit,
            "exact_timing": self.exact_timing,
            "guard_gap_ms": self.guard_gap_ms,
            "compress_internal_pauses": self.compress_internal_pauses,
            "internal_pause_keep_ms": self.internal_pause_keep_ms,
            "sync_mode": self.sync_mode.value,
            "cue_timing_mode": self.cue_timing_mode.value,
            "elastic_timing": self.elastic_timing.to_dict(),
            "tempo_smoothing": self.tempo_smoothing.to_dict(),
            "ffmpeg_path": self.ffmpeg_path,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "ducking": self.ducking.to_dict(),
            "preview": self.preview.to_dict(),
            "export": self.export.to_dict(),
            "cue_error_policy": self.cue_error_policy,
        }

    def max_speed_limit_compat(self) -> float:
        return self.preferred_speed_limit

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DubbingProjectSettings:
        sync_value = str(data.get("sync_mode", SyncMode.STRICT.value))
        try:
            sync_mode = SyncMode(sync_value)
        except ValueError:
            sync_mode = SyncMode.STRICT
        timing_value = str(data.get("cue_timing_mode", CueTimingMode.STRICT.value))
        try:
            cue_timing_mode = CueTimingMode(timing_value)
        except ValueError:
            cue_timing_mode = CueTimingMode.STRICT
        voice_config = data.get("voice_config", {})
        if not isinstance(voice_config, dict):
            voice_config = {}
        legacy_max = float(data.get("max_speed_factor", 1.35))
        preferred = float(data.get("preferred_speed_limit", legacy_max))
        hard = float(data.get("hard_speed_limit", max(2.5, preferred)))
        if hard < preferred:
            hard = preferred
        elastic = ElasticTimingSettings.from_dict(data.get("elastic_timing", {}))
        if cue_timing_mode == CueTimingMode.ELASTIC_GROUP:
            elastic.enabled = True
        elif "elastic_timing" not in data:
            elastic.enabled = False
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
            cue_timing_mode=cue_timing_mode,
            elastic_timing=elastic,
            tempo_smoothing=TempoSmoothingSettings.from_dict(
                data.get("tempo_smoothing", {})
            ),
            ffmpeg_path=str(data.get("ffmpeg_path", "")),
            sample_rate=int(data.get("sample_rate", 48000)),
            channels=int(data.get("channels", 2)),
            ducking=DuckingSettings.from_dict(data.get("ducking", {})),
            preview=PreviewSettings.from_dict(data.get("preview", {})),
            export=ExportSettings.from_dict(data.get("export", {})),
            cue_error_policy=str(data.get("cue_error_policy", "stop") or "stop"),
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
    schema_version: int = 3

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
