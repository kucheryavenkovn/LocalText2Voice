"""Resource snapshots taken at the boundaries of heavy operations.

Captures process RSS, system RAM, project disk free space, GPU name + VRAM,
CUDA availability and child process count. Every probe is defensive: a missing
``psutil``/``torch``/GPU must never crash the pipeline.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("video_dubbing.resources")

_psutil = None
try:
    import psutil as _psutil  # type: ignore
except Exception:  # pragma: no cover - psutil optional
    _psutil = None

_torch = None
_torch_cuda_checked = False
_torch_cuda_available = False


def _maybe_torch():
    global _torch, _torch_cuda_checked, _torch_cuda_available
    if _torch is not None or _torch_cuda_checked:
        return _torch
    _torch_cuda_checked = True
    try:
        import torch as _torch  # type: ignore
        _torch_cuda_available = bool(_torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch optional
        _torch = None
    return _torch


@dataclass
class ResourceLimit:
    rss_bytes: int | None = None
    available_ram_bytes: int | None = None
    disk_free_bytes: int | None = None
    gpu_name: str | None = None
    vram_allocated_bytes: int | None = None
    vram_reserved_bytes: int | None = None
    cuda_available: bool | None = None
    child_process_count: int = 0
    thread_count: int = 0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key == "extra":
                continue
            if value is None:
                continue
            if isinstance(value, int) and key.endswith("_bytes") and value > 0:
                data[key] = value
                data[key.replace("_bytes", "_mb")] = round(value / (1024 * 1024), 1)
            else:
                data[key] = value
        data.update(self.extra)
        return data


def snapshot_resources(
    project_dir: Path | None = None,
    *,
    label: str | None = None,
) -> ResourceLimit:
    """Collect a best-effort resource snapshot.

    ``project_dir`` is used to compute free disk space for the project volume.
    """
    snap = ResourceLimit()
    try:
        if _psutil is not None:
            proc = _psutil.Process(os.getpid())
            mem = proc.memory_info()
            snap.rss_bytes = int(getattr(mem, "rss", 0)) or None
            try:
                vm = _psutil.virtual_memory()
                snap.available_ram_bytes = int(vm.available) or None
            except Exception:  # pragma: no cover
                pass
            try:
                snap.child_process_count = len(proc.children(recursive=True))
            except Exception:  # pragma: no cover
                snap.child_process_count = 0
    except Exception as exc:  # pragma: no cover - never crash on diagnostics
        snap.error = f"psutil: {exc}"

    if project_dir is not None:
        try:
            usage = shutil.disk_usage(str(Path(project_dir)))
            snap.disk_free_bytes = int(usage.free) or None
        except Exception as exc:  # pragma: no cover
            snap.extra["disk_error"] = str(exc)

    # GPU / CUDA — never let this crash when CUDA is absent.
    try:
        torch = _maybe_torch()
        snap.cuda_available = _torch_cuda_available
        if torch is not None and _torch_cuda_available:
            try:
                snap.gpu_name = torch.cuda.get_device_name(0)
            except Exception:  # pragma: no cover
                snap.gpu_name = None
            try:
                snap.vram_allocated_bytes = int(torch.cuda.memory_allocated(0)) or None
                snap.vram_reserved_bytes = int(torch.cuda.memory_reserved(0)) or None
            except Exception:  # pragma: no cover
                pass
    except Exception as exc:  # pragma: no cover
        snap.extra["gpu_error"] = str(exc)

    try:
        snap.thread_count = threading.active_count()
    except Exception:  # pragma: no cover
        snap.thread_count = 0

    if label:
        snap.extra["label"] = label
    _log.debug("resource snapshot: %s", snap.to_dict())
    return snap


def log_resource_snapshot(
    project_dir: Path | None = None,
    *,
    label: str,
) -> dict[str, Any]:
    """Snapshot + emit + return as dict (for embedding in events)."""
    from .events import emit_event

    snap = snapshot_resources(project_dir, label=label)
    payload = snap.to_dict()
    emit_event("resource.snapshot", payload={"label": label, **payload})
    return payload


__all__ = ["ResourceLimit", "log_resource_snapshot", "snapshot_resources"]
