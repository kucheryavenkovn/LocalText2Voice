"""Video-dubbing MCP/HTTP control plane.

This package adds a programmatic control layer for the video-dubbing pipeline on
top of the existing persistent FastMCP/HTTP host in :mod:`app.server.http_app`.
It never creates a second MCP server and never imports PySide/Qt.

Layers (top -> bottom)::

    MCP tools / HTTP routes   (mcp_tools.py, http_routes.py)
        |
    VideoDubbingFacade        (facade.py)
        |
    DubbingJobManager         (job_manager.py)
        + revision / locks / snapshots / quality / simulation
        |
    VideoDubbingService       (app.core.video_dubbing.service)
        + DurationFitter / ElasticTimingPlanner / TempoSmoothingPlanner
        + TimelineRenderer / AudioMixer / PreviewRenderer / VideoMuxer
        + Observability
"""

from __future__ import annotations

from .errors import (
    DubbingError,
    DubbingFaultNotAvailable,
    DubbingNotFoundError,
    DubbingProjectBusyError,
    DubbingProjectChangedError,
    DubbingValidationError,
)
from .engine_provider import EngineLease, EngineLeaseProvider, StaticEngineLeaseProvider
from .facade import VideoDubbingFacade
from .job_manager import DubbingJobManager
from .job_models import DubbingJob, DubbingJobStatus
from .project_locks import ProjectLock, ProjectLockRegistry
from .revision_snapshots import RevisionStore, SnapshotStore

__all__ = [
    "DubbingError",
    "DubbingFaultNotAvailable",
    "DubbingNotFoundError",
    "DubbingProjectBusyError",
    "DubbingProjectChangedError",
    "DubbingValidationError",
    "DubbingJob",
    "DubbingJobStatus",
    "DubbingJobManager",
    "EngineLease",
    "EngineLeaseProvider",
    "StaticEngineLeaseProvider",
    "ProjectLock",
    "ProjectLockRegistry",
    "RevisionStore",
    "SnapshotStore",
    "VideoDubbingFacade",
]
