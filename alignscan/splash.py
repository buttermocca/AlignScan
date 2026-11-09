from __future__ import annotations
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QMovie
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QProgressBar


class Splash(QWidget):
    def __init__(self, asset_path: str):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setFixedSize(390, 360)
        self.setStyleSheet("QWidget{background:#fffaf7;border:2px solid #f7c9ca;border-radius:24px;}")
        layout = QVBoxLayout(self); layout.setContentsMargins(24, 16, 24, 22); layout.setSpacing(8)
        self.mascot = QLabel(); self.mascot.setAlignment(Qt.AlignCenter)
        self.movie = QMovie(asset_path); self.movie.setScaledSize(QSize(235, 235))
        self.mascot.setMovie(self.movie); self.movie.start()
        layout.addWidget(self.mascot, 1)
        title = QLabel("AlignScan")
        title.setAlignment(Qt.AlignCenter); title.setStyleSheet("color:#314b43;font-size:27px;font-weight:800;border:0")
        layout.addWidget(title)
        subtitle = QLabel("영수증과 카드를 반듯하게, 하나씩 ♡")
        subtitle.setAlignment(Qt.AlignCenter); subtitle.setStyleSheet("color:#80968c;font-size:12px;border:0")
        layout.addWidget(subtitle)
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.setTextVisible(False)
        self.progress.setFixedHeight(7); self.progress.setStyleSheet("QProgressBar{background:#f4e8e4;border:0;border-radius:3px} QProgressBar::chunk{background:#ff8c9b;border-radius:3px}")
        layout.addWidget(self.progress)
        self.center()

    def center(self):
        screen = self.screen().availableGeometry() if self.screen() else None
        if screen:
            self.move(screen.center().x() - self.width() // 2, screen.center().y() - self.height() // 2)
