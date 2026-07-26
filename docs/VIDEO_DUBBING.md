# Video Dubbing (Озвучка видео)

Timed TTS dubbing for a video using an SRT script. Each SRT cue is synthesized
separately, measured, time-fitted to its `start`–`end` window, placed on the
video's absolute timeline, mixed with the original audio (with dynamic ducking),
and muxed into a final container with multiple audio tracks.

This is **not** lip-sync. The video track is never modified.

## Workflow

1. **Create project** — pick video, SRT, project folder, language, TTS engine,
   voice, max speed factor, sync mode, original-audio mode, ducking levels,
   and output container. The video duration (from FFprobe) fixes the length of
   the narration/mix tracks.
2. **Import SRT** — parsed into per-cue models with validation warnings.
3. **Generate (TTS)** — each cue is synthesized via the existing `BaseTTSEngine`,
   normalized to PCM, measured. This produces **raw** speech at natural length.
4. **Fit / Recalculate** — `required_speed_factor = raw_duration / budget`. FFmpeg
   `atempo` is applied to build **fitted** WAVs. Elastic groups and tempo
   smoothing run here too. **Right shifts do not re-run TTS.**
5. **Render narration** — a single track exactly as long as the video, with
   each fitted cue placed at its absolute (or planned) timecode and silence
   elsewhere. No cumulative drift.
6. **Render mix** — original audio + narration + dynamic ducking (driven by a
   timecode gain mask, not an envelope follower).
7. **Preview** — short per-cue preview clips and a full-project preview, both
   with the video stream copied / quickly re-encoded (never the heavy export
   path on every edit).
8. **Export** — final MP4/MKV with: Original track, Dubbed Mix (default),
   optional Narration Only, optional embedded subtitles. Video is copied
   (`-c:v copy`) when the codec is container-compatible.
9. **Optional review** — More → Cue review opens the shared Whisper review UI
   on cue WAVs (play + verify; audiobook rebuild disabled).

### Timing vs speed-up

| Layer | Changes | Needs TTS? | Button |
| --- | --- | --- | --- |
| Raw generation | voice, text, engine | Yes | Create speech / Force all |
| Fit / atempo | speed limits, pauses | No | Recalculate |
| Elastic groups | planned start/end, right shift | No | Recalculate |
| Tempo smoothing | planned speed factors only (strict) | No | Recalculate |
| Narration / mix / mux | placement + levels | No | Narration / Mix / Video |

### Timeline colors

| Color | Meaning |
| --- | --- |
| Green | Ready, little/no speed-up |
| Blue | Mild speed-up |
| Orange | Strong speed-up |
| Red | Failed / needs shortening / overflow |
| Yellow outline | Selected |
| Cyan dashed | Elastic/tempo group (and shift) |
| Dim gray underlay | Original SRT window |

## Architecture

All logic lives in `app/core/video_dubbing/` (UI-free, service-driven):

| Component | Responsibility |
| --- | --- |
| `models.py` | `DubbingCue`, `DubbingProject`, settings, enums, stale flags |
| `srt_parser.py` | Robust SRT parsing (BOM, CRLF, multiline, missing indices) + validation |
| `duration_fitter.py` | Per-cue speed/overflow decision (`atempo` chain, strict/best-effort) |
| `project_store.py` | SQLite + `dubbing_project.json` manifest, load/save/round-trip |
| `stale_state.py` | Granular invalidation (text vs ducking vs container changes) |
| `cue_generator.py` | TTS synthesis per cue + PCM normalization + fitting (cancel-aware) |
| `timeline_renderer.py` | Absolute-placement narration render, windowed for thousands of cues |
| `audio_mixer.py` | Ducking gain mask + Dubbed Mix + narration MP3 + limiter/loudnorm |
| `video_muxer.py` | Final container: Original + Dubbed Mix (default) + Narration Only + subs |
| `preview_renderer.py` | Fast cue preview + full preview (video stream copy) |
| `reports.py` | `timing_report.json` / `timing_report.csv` |
| `service.py` | `VideoDubbingService` — the only orchestration entry point |

Supporting modules:

- `app/utils/ffprobe_utils.py` — `find_ffprobe`, `probe_media`, video/audio
  stream parsing (FFprobe was not previously wrapped in this codebase).
- `app/workers/video_dubbing_worker.py` — `QThread` worker with Qt signals.
- `app/ui/video_dubbing_page.py` — the «Озвучка видео» page with an embedded
  `QMediaPlayer`/`QVideoWidget`/`QAudioOutput` player, cue table, and actions.
- `app/ui/cue_timeline_widget.py` — painted cue timeline + playhead.

The page is integrated into `MainWindow` at page-stack index 6 with a sidebar
entry and a View-menu item.

## Project layout on disk

```
<project_dir>/
    dubbing_project.json        # manifest (also mirrored in SQLite)
    source/subtitles.srt        # copy of imported SRT
    cues/cue_000001_raw.wav
    cues/cue_000001_fitted.wav
    preview/cue_000001_preview.mkv
    preview/full_preview.mkv
    render/narration.wav
    render/narration.mp3
    render/dubbed_mix.wav
    render/dubbed_video.mkv
    reports/timing_report.json
    reports/timing_report.csv
    temp/
```

## Final audio tracks

The exported video contains multiple audio tracks so any player can switch:

| Track | Title | Default | Content |
| --- | --- | --- | --- |
| 0 | Original | no | Unmodified source audio |
| 1 | Dubbed Mix | **yes** | Original (ducked) + narration |
| 2 (optional) | Narration Only | no | Translation voice only |

Track names are stored in both `title` (MKV) and `handler_name` (MP4) tags.

## Ducking

Default dynamic-ducking envelope, driven by cue timecodes:

- Original outside narration: **100 %**
- Original during narration: **15 %** (≈ −16.5 dB)
- Narration: **100 %**
- Attack: 100 ms, Release: 250 ms (linear ramps in the gain mask — click-free)

A 1 kHz-resolution gain mask WAV is built from cue intervals and applied via
`sidechaincompress`, which keeps it deterministic for thousands of cues without
a huge FFmpeg command line. An `alimiter` and optional `loudnorm`
(−16 LUFS / −1 dBTP) protect the final mix.

## Reusing existing infrastructure

- TTS: existing `BaseTTSEngine` / `create_tts_engine` / `voice_config`. No
  parallel TTS API.
- FFmpeg: existing `FFmpegRunner` / `find_ffmpeg`; FFprobe added alongside.
- Audiobook pipeline: **untouched**. Dubbing uses its own SQLite tables and
  manifest and never mixes with the audiobook model.

## Limitations

- No lip-sync, no video re-encoding by default, no speaker diarization.
- Strong speed-up (> 1.35 by default) degrades TTS quality; such cues are
  flagged, not silently accelerated to the required factor.
- MP4 stores track names in `handler_name`; some players show them differently.
- Codec/container incompatibilities are reported; MKV is the flexible default.
- Overlapping cues are reported; in `block` sync mode they block export.

## Tests

```
python -m pytest tests/test_video_dubbing_srt_fitter.py tests/test_video_dubbing_store.py \
  tests/test_video_dubbing_cue_generator.py tests/test_video_dubbing_timeline.py \
  tests/test_video_dubbing_audio_mixer.py tests/test_video_dubbing_muxer.py \
  tests/test_video_dubbing_preview.py tests/test_video_dubbing_service.py \
  tests/test_video_dubbing_ui.py -q
```

FFmpeg/FFprobe-dependent tests skip automatically when the tools are missing.
