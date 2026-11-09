from __future__ import annotations
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QPixmap, QColor, QPolygonF
from PySide6.QtWidgets import QWidget
import numpy as np


class ScanCanvas(QWidget):
    region_selected = Signal(int)
    polygon_created = Signal(object)
    polygon_changed = Signal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 420)
        self.setMouseTracking(True)
        self.pixmap = QPixmap()
        self.regions: list[np.ndarray] = []
        self.selected = -1
        self.add_mode = False
        self.draft: list[QPointF] = []
        self.drag_corner: tuple[int, int] | None = None
        self._draw_rect = QRectF()
        self._scale = 1.0
        self._offset = QPointF(0, 0)

    def set_image(self, pixmap: QPixmap):
        self.pixmap = pixmap
        self.update()

    def set_regions(self, regions: list[np.ndarray], selected: int = -1):
        self.regions = [np.asarray(p, dtype=np.float32).copy() for p in regions]
        self.selected = selected
        self.update()

    def set_add_mode(self, enabled: bool):
        self.add_mode = enabled
        self.draft.clear()
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def _transform(self):
        if self.pixmap.isNull():
            return
        iw, ih = self.pixmap.width(), self.pixmap.height()
        avail = self.rect().adjusted(22, 22, -22, -22)
        self._scale = min(avail.width() / iw, avail.height() / ih)
        dw, dh = iw * self._scale, ih * self._scale
        self._offset = QPointF((self.width() - dw) / 2, (self.height() - dh) / 2)
        self._draw_rect = QRectF(self._offset.x(), self._offset.y(), dw, dh)

    def image_to_widget(self, p):
        return QPointF(self._offset.x() + float(p[0]) * self._scale,
                       self._offset.y() + float(p[1]) * self._scale)

    def widget_to_image(self, p: QPointF):
        return np.array([(p.x() - self._offset.x()) / self._scale,
                         (p.y() - self._offset.y()) / self._scale], dtype=np.float32)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f6f8f7"))
        painter.setPen(QPen(QColor("#d7e5df"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(self.rect().adjusted(8, 8, -8, -8), 18, 18)
        if self.pixmap.isNull():
            painter.setPen(QColor("#7c9288"))
            painter.drawText(self.rect(), Qt.AlignCenter, "스캔 이미지를 열어주세요\n\n여러 장의 영수증·카드가 있는 이미지도\n한 번에 불러올 수 있어요 ♡")
            return
        self._transform()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(self._draw_rect, self.pixmap, QRectF(self.pixmap.rect()))
        for idx, poly in enumerate(self.regions):
            polygon = QPolygonF([self.image_to_widget(p) for p in poly])
            selected = idx == self.selected
            color = QColor("#ff7d91") if selected else QColor("#4bb99a")
            painter.setPen(QPen(color, 3 if selected else 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 26))
            painter.drawPolygon(polygon)
            if selected:
                for p in poly:
                    q = self.image_to_widget(p)
                    painter.setPen(QPen(Qt.white, 2)); painter.setBrush(QBrush(color))
                    painter.drawEllipse(q, 6, 6)
            center = polygon.boundingRect().center()
            painter.setPen(QPen(QColor("#ffffff"), 4))
            painter.drawText(center + QPointF(1, 1), str(idx + 1))
            painter.setPen(QPen(color, 1))
            painter.drawText(center, str(idx + 1))
        if self.draft:
            painter.setPen(QPen(QColor("#ff9d42"), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            for p in self.draft:
                painter.drawEllipse(p, 5, 5)
            if len(self.draft) > 1:
                painter.drawPolyline(QPolygonF(self.draft))

    def mousePressEvent(self, event):
        if self.pixmap.isNull() or event.button() != Qt.LeftButton:
            return
        if self.add_mode:
            self.draft.append(event.position())
            if len(self.draft) == 4:
                points = np.array([self.widget_to_image(p) for p in self.draft], dtype=np.float32)
                self.polygon_created.emit(points)
                self.draft.clear()
                self.add_mode = False
                self.setCursor(Qt.ArrowCursor)
            self.update(); return
        if self.selected >= 0 and self.selected < len(self.regions):
            for corner_idx, p in enumerate(self.regions[self.selected]):
                q = self.image_to_widget(p)
                if (q - event.position()).manhattanLength() < 16:
                    self.drag_corner = (self.selected, corner_idx)
                    return
        for idx in reversed(range(len(self.regions))):
            poly = QPolygonF([self.image_to_widget(p) for p in self.regions[idx]])
            if poly.containsPoint(event.position(), Qt.OddEvenFill):
                self.region_selected.emit(idx)
                return

    def mouseMoveEvent(self, event):
        if self.drag_corner is not None:
            idx, corner = self.drag_corner
            points = self.regions[idx].copy()
            points[corner] = self.widget_to_image(event.position())
            self.regions[idx] = points
            self.polygon_changed.emit(idx, points)
            self.update()

    def mouseReleaseEvent(self, event):
        self.drag_corner = None
