from __future__ import annotations

import faulthandler
import logging
import sys
import threading
import traceback
from pathlib import Path
from typing import Any


_CRASH_FILE_NAME = "crash_dump.log"


def configure_logging(
    log_file: Path | None = None,
    crash_file: Path | None = None,
    install_excepthook: bool = True,
) -> None:
    """Configure logging and install crash/exception diagnostics.

    * Regular logs go to ``log_file`` (and stderr).
    * ``faulthandler`` dumps native crashes (segfaults, aborts) to
      ``crash_file`` and stderr, so a Qt/ffmpeg-level crash leaves a traceback
      instead of disappearing.
    * Uncaught Python exceptions are logged through the excepthook.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

    if crash_file is None and log_file is not None:
        crash_file = log_file.parent / _CRASH_FILE_NAME
    if crash_file is not None:
        crash_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            crash_handle = crash_file.open("a", encoding="utf-8", buffering=1)
        except OSError:
            crash_handle = None
    else:
        crash_handle = None

    # faulthandler: dump native crashes (SIGSEGV/SIGFPE/SIGABRT). The previous
    # unconditional ``dump_traceback_later(30)`` is intentionally removed: it
    # produced a spurious stack dump every 30 s even during normal long
    # operations. Stall detection is now opt-in via
    # app.observability.OperationWatchdog (armed only for active operations).
    try:
        faulthandler.enable(file=crash_handle if crash_handle else sys.__stderr__)
        for signame in ("SIGABRT", "SIGFPE", "SIGILL", "SIGSEGV", "SIGBUS"):
            signum = getattr(faulthandler, signame, None) or getattr(
                __import__("signal"), signame, None
            )
            if signum is not None:
                try:
                    faulthandler.register(
                        signum,
                        file=crash_handle if crash_handle else sys.__stderr__,
                        all_threads=True,
                        chain=False,
                    )
                except (ValueError, OSError):
                    pass
    except Exception:  # pragma: no cover - never let logging setup crash the app
        pass

    if install_excepthook:
        _install_excepthook(crash_handle)


def _install_excepthook(crash_handle: Any) -> None:
    previous_hook = sys.excepthook

    def excepthook(exc_type, exc_value, exc_tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logging.getLogger("crash").critical("Uncaught exception:\n%s", text)
        if crash_handle is not None:
            try:
                crash_handle.write(
                    "\n===== UNCAUGHT EXCEPTION =====\n" + text + "\n"
                )
                crash_handle.flush()
            except OSError:
                pass
        # Threading exceptions are not routed through sys.excepthook.
        try:
            threading.excepthook  # noqa: B018
        except AttributeError:
            pass

    def threading_excepthook(args) -> None:
        text = "".join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        )
        thread_name = getattr(args.thread, "name", "?")
        logging.getLogger("crash").critical(
            "Uncaught exception in thread %s:\n%s", thread_name, text
        )
        if crash_handle is not None:
            try:
                crash_handle.write(
                    f"\n===== UNCAUGHT EXCEPTION (thread {thread_name}) =====\n"
                    + text
                    + "\n"
                )
                crash_handle.flush()
            except OSError:
                pass

    sys.excepthook = excepthook
    try:
        threading.excepthook = threading_excepthook
    except AttributeError:  # pragma: no cover
        pass
    _ = previous_hook


def install_qt_message_handler(crash_file: Path | None = None) -> None:
    """Route Qt warnings/fatal messages into the log + crash file.

    Must be called after ``QApplication`` is created. Qt fatal messages
    (including media-backend failures) are recorded before the process dies.
    """
    try:
        from PySide6.QtCore import qInstallMessageHandler, Qt
    except Exception:  # pragma: no cover
        return

    crash_handle = None
    if crash_file is not None:
        crash_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            crash_handle = crash_file.open("a", encoding="utf-8", buffering=1)
        except OSError:
            crash_handle = None

    def handler(level, context, message) -> None:
        logger = logging.getLogger("qt")
        level_name = {
            Qt.QtMsgType.QtDebugMsg: "DEBUG",
            Qt.QtMsgType.QtInfoMsg: "INFO",
            Qt.QtMsgType.QtWarningMsg: "WARNING",
            Qt.QtMsgType.QtCriticalMsg: "CRITICAL",
            Qt.QtMsgType.QtFatalMsg: "FATAL",
        }.get(level, "WARNING")
        category = getattr(context, "category", "") or ""
        prefix = f"[{category}] " if category else ""
        text = f"Qt {level_name}: {prefix}{message}"
        logger.log(
            logging.DEBUG
            if level_name == "DEBUG"
            else logging.INFO
            if level_name == "INFO"
            else logging.WARNING
            if level_name == "WARNING"
            else logging.ERROR,
            text,
        )
        if crash_handle is not None and level_name in {"CRITICAL", "FATAL"}:
            try:
                crash_handle.write("\n===== QT FATAL/CRITICAL =====\n" + text + "\n")
                crash_handle.flush()
            except OSError:
                pass

    qInstallMessageHandler(handler)
