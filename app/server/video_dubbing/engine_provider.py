"""Engine lease provider.

The video-dubbing facade must not create a fresh TTS engine per cue/MCP call.
It leases the *shared*, persistent engine from the existing
:class:`LocalText2VoiceService` engine cache and never closes it after each
job. The provider is the single seam used by tests to inject fake engines.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from app.core.video_dubbing.models import DubbingProjectSettings
from app.tts.base import BaseTTSEngine


class _SupportsEngineCache(Protocol):
    keep_engines_alive: bool

    def _get_tts_engine(self, engine_id: str, log_callback: Any) -> BaseTTSEngine: ...

    def unload_engine(self, engine_id: str | None = None) -> dict[str, Any]: ...


@dataclass
class EngineLease:
    engine_id: str
    engine: BaseTTSEngine
    ownership: str = "shared"
    acquired_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    released: bool = False


class EngineLeaseProvider:
    """Acquire/release shared engines from the persistent host cache."""

    def __init__(self, ltv_service: _SupportsEngineCache | None = None) -> None:
        self._ltv = ltv_service
        self._lock = threading.RLock()
        self._leases: dict[str, int] = {}

    def acquire(
        self,
        settings: DubbingProjectSettings,
        *,
        log_callback: Any = None,
    ) -> EngineLease:
        engine_id = _normalise_engine_id(settings.tts_engine)
        log = log_callback or (lambda _msg: None)
        if self._ltv is None:
            raise RuntimeError(
                "No engine host available. Configure a LocalText2VoiceService "
                "or inject a StaticEngineLeaseProvider for tests."
            )
        engine = self._ltv._get_tts_engine(engine_id, log)
        with self._lock:
            self._leases[engine_id] = self._leases.get(engine_id, 0) + 1
        return EngineLease(engine_id=engine_id, engine=engine)

    def release(self, lease: EngineLease) -> None:
        if lease is None or lease.released:
            return
        lease.released = True
        # Shared engines are intentionally NOT closed here. They live in the
        # persistent host cache and are reused across jobs/projects.
        with self._lock:
            count = self._leases.get(lease.engine_id, 0)
            if count > 0:
                self._leases[lease.engine_id] = count - 1

    def unload(self, engine_id: str) -> dict[str, Any]:
        engine_id = _normalise_engine_id(engine_id)
        if self._ltv is None:
            return {"engine_id": engine_id, "loaded": False, "unloaded": False}
        return self._ltv.unload_engine(engine_id)

    def active_leases(self) -> dict[str, int]:
        with self._lock:
            return dict(self._leases)


class StaticEngineLeaseProvider(EngineLeaseProvider):
    """Test double: always returns a fixed engine instance.

    Used by integration tests so they never depend on a real/CUDA engine and
    never construct a full :class:`LocalText2VoiceService`.
    """

    def __init__(self, engine: BaseTTSEngine, engine_id: str | None = None) -> None:
        super().__init__(ltv_service=None)
        self._engine = engine
        self._engine_id = _normalise_engine_id(engine_id or getattr(engine, "engine_id", "static"))

    def acquire(
        self,
        settings: DubbingProjectSettings,
        *,
        log_callback: Any = None,
    ) -> EngineLease:
        with self._lock:
            self._leases[self._engine_id] = self._leases.get(self._engine_id, 0) + 1
        return EngineLease(
            engine_id=self._engine_id,
            engine=self._engine,
            ownership="static",
        )


def _normalise_engine_id(engine_id: str) -> str:
    engine_id = str(engine_id or "piper").strip()
    if engine_id == "kokoro_python":
        engine_id = "kokoro"
    return engine_id
