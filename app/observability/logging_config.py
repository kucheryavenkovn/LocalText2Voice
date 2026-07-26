"""Application logging bootstrap.

* RotatingFileHandler: 10 MB, 10 backups, UTF-8, at
  ``logs/application/localtext2voice.log``;
* StreamHandler to stderr;
* crash handlers installed via :mod:`crash_reporter` (no unconditional
  ``faulthandler.dump_traceback_later``);
* a logging filter that injects the active diagnostic context into every
  record so log lines carry ``run_id``/``operation_id``/``cue_id``.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .context import current_context
from .crash_reporter import install_crash_handlers

_DEFAULT_LOG_DIR = Path("logs") / "application"
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "localtext2voice.log"
_DEFAULT_CRASH_FILE = Path("logs") / "crashes" / "crash_dump.log"

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 10

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class _ContextInjectingFilter(logging.Filter):
    """Append the active diagnostic context to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            ctx = current_context()
        except Exception:  # pragma: no cover
            ctx = {}
        record.diag_context = ctx  # type: ignore[attr-defined]
        # Render a compact context prefix into the message for readability.
        if ctx:
            parts = []
            for key in ("run_id", "operation_id", "cue_id", "engine_id", "stage"):
                value = ctx.get(key)
                if value not in (None, "", []):
                    parts.append(f"{key}={value}")
            record.context_prefix = " ".join(parts)  # type: ignore[attr-defined]
        else:
            record.context_prefix = ""  # type: ignore[attr-defined]
        return True


class _ContextFormatter(logging.Formatter):
    """Formatter that includes the context prefix when present."""

    def __init__(self) -> None:
        super().__init__(fmt=_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        prefix = getattr(record, "context_prefix", "")
        if prefix:
            return f"{base}  [{prefix}]"
        return base


def configure_observability_logging(
    log_file: Path | None = None,
    crash_file: Path | None = None,
    *,
    level: int = logging.INFO,
    stream: bool = True,
    install_crash: bool = True,
) -> tuple[Path, Path]:
    """Configure rotating application logging + crash handlers.

    Returns ``(log_file, crash_file)`` actually used. Safe to call once at
    startup; uses ``force=True`` so a re-configuration fully replaces handlers
    (matching the previous ``configure_logging`` contract).
    """
    log_path = Path(log_file) if log_file else _DEFAULT_LOG_FILE
    crash_path = Path(crash_file) if crash_file else _DEFAULT_CRASH_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    crash_path.parent.mkdir(parents=True, exist_ok=True)

    diag_filter = _ContextInjectingFilter()
    formatter = _ContextFormatter()

    handlers: list[logging.Handler] = []
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(diag_filter)
    handlers.append(file_handler)

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(diag_filter)
        handlers.append(stream_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    if install_crash:
        install_crash_handlers(crash_path)

    return log_path, crash_path


__all__ = ["configure_observability_logging"]
