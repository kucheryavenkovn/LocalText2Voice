from __future__ import annotations

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.observability import configure_observability_logging, new_session_id, set_app_session_id
from app.ui.main_window import MainWindow
from app.utils.logging_utils import install_qt_message_handler
from app.utils.paths import application_root, resource_root


def main() -> int:
    log_dir = application_root() / "logs"
    set_app_session_id(new_session_id())
    configure_observability_logging(
        log_file=log_dir / "application" / "localtext2voice.log",
        crash_file=log_dir / "crashes" / "crash_dump.log",
    )
    application = QApplication(sys.argv)
    install_qt_message_handler(crash_file=log_dir / "crashes" / "crash_dump.log")
    application.setApplicationName("LocalText2Voice")
    application.setOrganizationName("AndromedaNova")
    application.setWindowIcon(
        QIcon(str(resource_root() / "assets" / "logotipo.png"))
    )
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
