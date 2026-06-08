
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from detector import config
from gui import MainWindow


def main() -> int:
    config.ensure_dirs()
    app = QApplication(sys.argv)
    app.setApplicationName("PPE Detection")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
