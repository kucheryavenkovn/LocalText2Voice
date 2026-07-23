from __future__ import annotations

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow
from app.utils.logging_utils import configure_logging, install_qt_message_handler
from app.utils.paths import application_root, resource_root


def main() -> int:
    log_dir = application_root() / "logs"
    configure_logging(
        log_file=log_dir / "local_text_2_voice.log",
        crash_file=log_dir / "crash_dump.log",
    )
    application = QApplication(sys.argv)
    install_qt_message_handler(crash_file=log_dir / "crash_dump.log")
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
