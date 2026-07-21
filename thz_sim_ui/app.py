from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from thz_sim_ui.constants import APP_NAME, APP_ORG
from thz_sim_ui.main_window import MainWindow
from thz_sim_ui.services.backend import BackendService
from thz_sim_ui.styles.theme import APP_QSS


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setStyle('Fusion')
    app.setStyleSheet(APP_QSS)

    backend = BackendService()
    window = MainWindow(backend)
    window.showMaximized()
    return app.exec()
