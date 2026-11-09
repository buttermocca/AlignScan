from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QWheelEvent
from PySide6.QtWidgets import QScrollArea, QLabel


class _PreviewImageLabel(QLabel):
    wheel_zoom = Signal(int)

    def wheelEvent(self, event: QWheelEvent):
        self.wheel_zoom.emit(event.angleDelta().y())
        event.accept()


class ZoomPreview(QScrollArea):
    """Image preview that fits to the panel and zooms with the mouse wheel."""
    zoom_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setMinimumHeight(285)
        self.setStyleSheet("QScrollArea{background:#fffafd;border:1px solid #eadfeb;border-radius:14px;}")
        self.label = _PreviewImageLabel("영역을 선택하면 보정 미리보기가 표시됩니다\n마우스 휠로 확대·축소할 수 있어요 🔎")
        self.label.wheel_zoom.connect(self._zoom_by_delta)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("QLabel{background:#fffafd;color:#8791aa;padding:18px;font-size:10pt;}")
        self.label.setMinimumSize(1, 1)
        self.setWidget(self.label)
        self._source = QPixmap()
        self._base_scale = 1.0
        self._zoom = 1.0

    def set_message(self, message: str):
        self._source = QPixmap()
        self._zoom = 1.0
        self.label.setPixmap(QPixmap())
        self.label.setText(message)
        self.label.resize(self.viewport().size())

    def set_preview(self, pixmap: QPixmap, reset_zoom: bool = False):
        if pixmap.isNull():
            self.set_message("미리보기를 만들 수 없습니다.")
            return
        self._source = pixmap
        if reset_zoom:
            self._zoom = 1.0
        QTimer.singleShot(0, self._apply_fit)

    def _apply_fit(self):
        if self._source.isNull():
            return
        area = self.viewport().size()
        if area.width() < 20 or area.height() < 20:
            return
        self._base_scale = min((area.width() - 18) / self._source.width(),
                               (area.height() - 18) / self._source.height())
        self._base_scale = max(0.01, self._base_scale)
        self._render()

    def _render(self):
        if self._source.isNull():
            return
        factor = self._base_scale * self._zoom
        width = max(1, int(self._source.width() * factor))
        height = max(1, int(self._source.height() * factor))
        scaled = self._source.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setText("")
        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())
        self.zoom_changed.emit(int(round(self._zoom * 100)))

    def _zoom_by_delta(self, delta: int):
        if self._source.isNull() or not delta:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._zoom = min(5.0, max(0.35, self._zoom * factor))
        self._render()
        QTimer.singleShot(0, lambda: (
            self.horizontalScrollBar().setValue(max(0, (self.label.width() - self.viewport().width()) // 2)),
            self.verticalScrollBar().setValue(max(0, (self.label.height() - self.viewport().height()) // 2))
        ))

    def wheelEvent(self, event: QWheelEvent):
        self._zoom_by_delta(event.angleDelta().y())
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._source.isNull():
            QTimer.singleShot(0, self._apply_fit)
