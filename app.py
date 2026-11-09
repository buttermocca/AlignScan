from __future__ import annotations
import os
import sys
from pathlib import Path
import cv2
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from alignscan.main_window import MainWindow
from alignscan.splash import Splash
from alignscan.asset_generator import ensure_assets


def resource_path(relative: str) -> str:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return str(base / relative)


def main():
    # Avoid oversubscribing CPU threads on mobile Ryzen laptops. Four workers
    # benchmarked well for scanner-size images while keeping the UI responsive.
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, min(4, os.cpu_count() or 2)))

    app = QApplication(sys.argv)
    app.setApplicationName("AlignScan")
    app.setOrganizationName("buttermocca")
    assets_dir = Path(resource_path("assets"))
    ensure_assets(assets_dir)
    icon_path = str(assets_dir / "app_icon.png")
    if Path(icon_path).exists():
        app.setWindowIcon(QIcon(icon_path))
    splash = Splash(str(assets_dir / "splash_mascot.gif"))
    splash.show()
    window = MainWindow()
    QTimer.singleShot(1500, lambda: (splash.close(), window.show()))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
