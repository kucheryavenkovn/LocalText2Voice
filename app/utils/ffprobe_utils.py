from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ffmpeg_utils import FFmpegError
from .paths import resolve_app_path


def find_ffprobe(configured_ffmpeg_or_ffprobe: str | Path) -> Path:
    configured = resolve_app_path(configured_ffmpeg_or_ffprobe)
    if configured.is_file() and configured.name.lower().startswith("ffprobe"):
        return configured

    ffmpeg_match = shutil.which("ffmpeg")
    ffprobe_match = shutil.which("ffprobe")
    if configured.is_file() and configured.name.lower().startswith("ffmpeg"):
        sibling = configured.with_name("ffprobe.exe" if _is_windows() else "ffprobe")
        if sibling.is_file():
            return sibling
    if ffprobe_match:
        return Path(ffprobe_match)
    if ffmpeg_match:
        sibling = Path(ffmpeg_match).with_name(
            "ffprobe.exe" if _is_windows() else "ffprobe"
        )
        if sibling.is_file():
            return sibling
    raise FFmpegError(
        "FFprobe was not found. Place ffprobe.exe next to ffmpeg.exe, "
        "set ffmpeg_path in config.json, or add FFprobe to PATH."
    )


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")


def find_ffmpeg_sibling(configured_ffmpeg: str | Path) -> Path:
    configured = resolve_app_path(configured_ffmpeg)
    if configured.is_file():
        return configured
    ffmpeg_match = shutil.which("ffmpeg")
    if ffmpeg_match:
        return Path(ffmpeg_match)
    raise FFmpegError(
        "FFmpeg was not found. Place ffmpeg.exe in the ffmpeg folder, "
        "set ffmpeg_path in config.json, or add FFmpeg to PATH."
    )


@dataclass(frozen=True)
class ProbeStream:
    codec_type: str
    codec_name: str
    sample_rate: int
    channels: int
    width: int
    height: int
    avg_frame_rate: str

    def fps(self) -> float:
        num, _, den = self.avg_frame_rate.partition("/")
        try:
            numerator = float(num)
            denominator = float(den) if den else 1.0
        except ValueError:
            return 0.0
        if denominator <= 0:
            return 0.0
        return numerator / denominator


def probe_media(
    media_path: Path,
    ffmpeg_path: str | Path,
) -> dict[str, Any]:
    if not media_path.is_file():
        raise FFmpegError(f"Media file not found: {media_path}")
    ffprobe = find_ffprobe(ffmpeg_path)
    creation_flags = (
        subprocess.CREATE_NO_WINDOW
        if hasattr(subprocess, "CREATE_NO_WINDOW")
        else 0
    )
    process = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(media_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creation_flags,
        check=False,
    )
    if process.returncode != 0:
        error_text = process.stderr.decode("utf-8", errors="replace").strip()
        raise FFmpegError(
            f"FFprobe failed for {media_path.name}:\n{error_text}"
        )
    try:
        data = json.loads(process.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"FFprobe returned invalid JSON for {media_path}.") from exc
    if not isinstance(data, dict):
        raise FFmpegError(f"FFprobe returned invalid data for {media_path}.")
    return data


def parse_video_probe(data: dict[str, Any]) -> tuple[int, ProbeStream | None, list[ProbeStream]]:
    format_info = data.get("format", {}) if isinstance(data.get("format"), dict) else {}
    duration_text = str(format_info.get("duration", "0") or "0")
    try:
        duration_seconds = float(duration_text)
    except ValueError:
        duration_seconds = 0.0
    duration_ms = int(round(duration_seconds * 1000))

    raw_streams = data.get("streams", [])
    streams: list[ProbeStream] = []
    if isinstance(raw_streams, list):
        for raw in raw_streams:
            if not isinstance(raw, dict):
                continue
            streams.append(
                ProbeStream(
                    codec_type=str(raw.get("codec_type", "")),
                    codec_name=str(raw.get("codec_name", "")),
                    sample_rate=int(raw.get("sample_rate", 0) or 0),
                    channels=int(raw.get("channels", 0) or 0),
                    width=int(raw.get("width", 0) or 0),
                    height=int(raw.get("height", 0) or 0),
                    avg_frame_rate=str(raw.get("avg_frame_rate", "0/1") or "0/1"),
                )
            )
    video_stream = next((s for s in streams if s.codec_type == "video"), None)
    return duration_ms, video_stream, [s for s in streams if s.codec_type == "audio"]
