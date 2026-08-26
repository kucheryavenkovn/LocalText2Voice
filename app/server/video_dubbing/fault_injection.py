"""Gated fault injection for the resilience test suite.

OFF by default. When enabled it is restricted to:

* localhost listeners only (``allow_lan`` false);
* a separate test token (``diagnostic_test_token``);
* projects explicitly marked ``test_project=true`` in their settings.

Production MCP tools never call this path. Every injection is logged at
WARNING/ERROR so accidental use is visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.observability import emit_event

from .errors import DubbingFaultNotAvailable, DubbingValidationError

_log = logging.getLogger("video_dubbing.fault_injection")


ALLOWED_FAULTS = frozenset(
    {
        "fail_tts_at_cue",
        "hang_tts_at_cue",
        "kill_ffmpeg_at_stage",
        "cancel_at_checkpoint",
        "simulate_database_lock",
        "simulate_disk_full",
        "leave_partial_artifact",
        "force_engine_exit",
    }
)


@dataclass
class FaultSpec:
    fault: str
    sequence: int | None = None
    stage: str | None = None
    exception_type: str = "RuntimeError"
    message: str = "Injected fault"


class FaultInjector:
    def __init__(
        self,
        *,
        diagnostic_test_mode: bool = False,
        allow_lan: bool = False,
        test_token: str = "",
    ) -> None:
        self.diagnostic_test_mode = bool(diagnostic_test_mode)
        self.allow_lan = bool(allow_lan)
        self.test_token = str(test_token or "")
        self._active: dict[str, FaultSpec] = {}

    def available(self, *, initiator_token: str = "", project_is_test: bool = False) -> bool:
        if not self.diagnostic_test_mode:
            return False
        if self.allow_lan:
            return False
        if self.test_token and initiator_token and initiator_token != self.test_token:
            return False
        if not project_is_test:
            return False
        return True

    def ensure_available(self, *, initiator_token: str = "", project_is_test: bool = False) -> None:
        if not self.diagnostic_test_mode:
            raise DubbingFaultNotAvailable(
                "Fault injection is disabled (diagnostic_test_mode is off).",
            )
        if self.allow_lan:
            raise DubbingFaultNotAvailable(
                "Fault injection is unavailable when allow_lan is true.",
            )
        if not project_is_test:
            raise DubbingFaultNotAvailable(
                "Fault injection targets test_project=true projects only.",
            )
        if self.test_token and initiator_token and initiator_token != self.test_token:
            raise DubbingFaultNotAvailable(
                "Fault injection requires the dedicated test token.",
            )

    def arm(self, spec: FaultSpec) -> None:
        if spec.fault not in ALLOWED_FAULTS:
            raise DubbingValidationError(
                f"Unknown fault: {spec.fault}", code="unknown_fault"
            )
        self._active[spec.fault] = spec
        _log.error("Fault injection armed: %s", spec.fault)
        emit_event(
            "fault_injection.armed",
            payload={"fault": spec.fault, "sequence": spec.sequence, "stage": spec.stage},
            force_flush=True,
        )

    def disarm(self, fault: str) -> None:
        self._active.pop(fault, None)

    def active_faults(self) -> list[str]:
        return list(self._active.keys())

    def maybe_trigger(self, hook: str, context: dict[str, Any]) -> None:
        """Called by instrumented code at well-defined checkpoints.

        ``hook`` is a stable checkpoint name (e.g. ``tts.before_synthesize``,
        ``ffmpeg.before_run``). If an armed fault matches, it raises the
        configured exception deterministically.
        """
        for spec in list(self._active.values()):
            if _matches(spec, hook, context):
                self._active.pop(spec.fault, None)
                exc_type = _resolve_exception(spec.exception_type)
                raise exc_type(spec.message)


def _matches(spec: FaultSpec, hook: str, context: dict[str, Any]) -> bool:
    if spec.fault == "fail_tts_at_cue" and hook == "tts.before_synthesize":
        return context.get("sequence") == spec.sequence
    if spec.fault == "hang_tts_at_cue" and hook == "tts.before_synthesize":
        return context.get("sequence") == spec.sequence
    if spec.fault == "kill_ffmpeg_at_stage" and hook == "ffmpeg.before_run":
        return str(context.get("stage")) == str(spec.stage)
    if spec.fault == "cancel_at_checkpoint":
        return hook.startswith("checkpoint.")
    if spec.fault == "leave_partial_artifact" and hook == "artifact.before_finalize":
        return True
    if spec.fault == "force_engine_exit" and hook == "engine.before_use":
        return True
    return False


def _resolve_exception(name: str) -> type[Exception]:
    mapping: dict[str, type[Exception]] = {
        "RuntimeError": RuntimeError,
        "TTSEngineError": __import__(
            "app.tts.base", fromlist=["TTSEngineError"]
        ).TTSEngineError,
        "OSError": OSError,
        "VideoMuxError": __import__(
            "app.core.video_dubbing.video_muxer", fromlist=["VideoMuxError"]
        ).VideoMuxError,
    }
    return mapping.get(name, RuntimeError)


def make_default_injector(server_settings: dict[str, Any]) -> FaultInjector:
    return FaultInjector(
        diagnostic_test_mode=bool(server_settings.get("diagnostic_test_mode", False)),
        allow_lan=bool(server_settings.get("allow_lan", False)),
        test_token=str(server_settings.get("diagnostic_test_token", "") or ""),
    )
