from __future__ import annotations

from pathlib import Path
import math
import numpy as np

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (QPainter, QPen, QBrush, QPixmap, QColor, QPolygonF,
                           QPainterPath, QWheelEvent, QMouseEvent)
from PySide6.QtWidgets import QWidget


class ScanCanvas(QWidget):
    region_selected = Signal(int)
    polygon_created = Signal(object)
    polygon_changed = Signal(int, object)
    image_dropped = Signal(str)
    ruler_created = Signal(object, object, str)
    zoom_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 420)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.pixmap = QPixmap()
        self.regions: list[np.ndarray] = []
        self.selected = -1

        self.add_mode = False
        self.draft: list[np.ndarray] = []
        self.drag_corner: tuple[int, int] | None = None

        self.ruler_mode: str | None = None
        self.ruler_line: tuple[np.ndarray, np.ndarray, str] | None = None
        self._ruler_start: np.ndarray | None = None
        self._ruler_current: np.ndarray | None = None

        self._zoom = 1.0
        self._fit_scale = 1.0
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._pan = QPointF(0, 0)
        self._pan_anchor: QPointF | None = None
        self._last_mouse = QPointF(-1000, -1000)
        self._magnifier_factor = 3.0

    @property
    def zoom_percent(self) -> int:
        return int(round(self._zoom * 100))

    def set_image(self, pixmap: QPixmap):
        self.pixmap = pixmap
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._transform()
        self.update()
        self.zoom_changed.emit(self.zoom_percent)

    def set_regions(self, regions: list[np.ndarray], selected: int = -1):
        self.regions = [np.asarray(p, dtype=np.float32).copy() for p in regions]
        self.selected = selected
        self.update()

    def set_add_mode(self, enabled: bool):
        self.add_mode = enabled
        self.ruler_mode = None
        self.draft.clear()
        self._ruler_start = None
        self._ruler_current = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def set_ruler_mode(self, orientation: str | None):
        if orientation not in (None, "horizontal", "vertical"):
            raise ValueError("룰러 방향은 horizontal 또는 vertical이어야 합니다.")
        self.add_mode = False
        self.draft.clear()
        self.ruler_mode = orientation
        self._ruler_start = None
        self._ruler_current = None
        self.setCursor(Qt.CrossCursor if orientation else Qt.ArrowCursor)
        self.update()

    def set_ruler_line(self, start, end, orientation: str):
        self.ruler_line = (
            np.asarray(start, dtype=np.float32).copy(),
            np.asarray(end, dtype=np.float32).copy(),
            orientation,
        )
        self.update()

    def clear_ruler_line(self):
        self.ruler_line = None
        self._ruler_start = None
        self._ruler_current = None
        self.update()

    def fit_to_window(self):
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._transform()
        self.zoom_changed.emit(self.zoom_percent)
        self.update()

    def _transform(self):
        if self.pixmap.isNull():
            return
        iw, ih = self.pixmap.width(), self.pixmap.height()
        avail = self.rect().adjusted(22, 22, -22, -22)
        self._fit_scale = max(0.001, min(avail.width() / iw, avail.height() / ih))
        self._scale = self._fit_scale * self._zoom
        dw, dh = iw * self._scale, ih * self._scale
        self._offset = QPointF(
            (self.width() - dw) / 2 + self._pan.x(),
            (self.height() - dh) / 2 + self._pan.y(),
        )

    def image_to_widget(self, point):
        self._transform()
        return QPointF(self._offset.x() + float(point[0]) * self._scale,
                       self._offset.y() + float(point[1]) * self._scale)

    def widget_to_image(self, point: QPointF):
        self._transform()
        if self._scale <= 0:
            return np.array([0, 0], dtype=np.float32)
        return np.array([(point.x() - self._offset.x()) / self._scale,
                         (point.y() - self._offset.y()) / self._scale], dtype=np.float32)

    def _zoom_at(self, point: QPointF, delta: int):
        if self.pixmap.isNull() or not delta:
            return
        self._transform()
        image_anchor = self.widget_to_image(point)
        factor = 1.18 if delta > 0 else 1 / 1.18
        self._zoom = min(12.0, max(0.10, self._zoom * factor))
        self._transform()
        self._pan = QPointF(
            point.x() - (self.width() - self.pixmap.width() * self._scale) / 2
            - float(image_anchor[0]) * self._scale,
            point.y() - (self.height() - self.pixmap.height() * self._scale) / 2
            - float(image_anchor[1]) * self._scale,
        )
        self._transform()
        self.zoom_changed.emit(self.zoom_percent)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f1f3ff"))
        painter.setPen(QPen(QColor("#dcdafa"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(self.rect().adjusted(8, 8, -8, -8), 18, 18)
        if self.pixmap.isNull():
            painter.setPen(QColor("#7773a8"))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "스캔 이미지를 여기에 끌어놓으세요 ♡\n\n"
                             "마우스 휠로 확대·축소하고 가운데 버튼으로 이동할 수 있어요\n"
                             "여러 장의 영수증·카드를 자동으로 찾아 반듯하게 정렬합니다")
            return

        self._transform()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        draw_rect = QRectF(self._offset.x(), self._offset.y(),
                           self.pixmap.width() * self._scale,
                           self.pixmap.height() * self._scale)
        painter.drawPixmap(draw_rect, self.pixmap, QRectF(self.pixmap.rect()))

        for idx, poly in enumerate(self.regions):
            polygon = QPolygonF([self.image_to_widget(point) for point in poly])
            selected = idx == self.selected
            color = QColor("#ff7396") if selected else QColor("#4eb99c")
            painter.setPen(QPen(color, 3 if selected else 2,
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 25))
            painter.drawPolygon(polygon)
            if selected:
                for point in poly:
                    q = self.image_to_widget(point)
                    painter.setPen(QPen(Qt.white, 2))
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(q, 7, 7)
            center = polygon.boundingRect().center()
            painter.setPen(QPen(QColor("#ffffff"), 4))
            painter.drawText(center + QPointF(1, 1), str(idx + 1))
            painter.setPen(QPen(color, 1))
            painter.drawText(center, str(idx + 1))

        if self.ruler_line is not None:
            start, end, orientation = self.ruler_line
            self._draw_ruler(painter, start, end, orientation, preview=False)
        if self._ruler_start is not None and self._ruler_current is not None:
            self._draw_ruler(painter, self._ruler_start, self._ruler_current,
                             self.ruler_mode or "horizontal", preview=True)

        if self.draft:
            painter.setPen(QPen(QColor("#8d83ed"), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            screen_points = [self.image_to_widget(point) for point in self.draft]
            for point in screen_points:
                painter.drawEllipse(point, 6, 6)
            if len(screen_points) > 1:
                painter.drawPolyline(QPolygonF(screen_points))
        if self.add_mode or self.drag_corner is not None:
            self._draw_magnifier(painter)

        # Canvas HUD remains legible over the scanned image.
        hud = QRectF(18, 16, 116, 30)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(44, 48, 89, 205))
        painter.drawRoundedRect(hud, 10, 10)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(hud, Qt.AlignCenter, f"  {self.zoom_percent}%  ·  휠 확대")

    def _draw_ruler(self, painter: QPainter, start, end, orientation: str, preview: bool):
        p1 = self.image_to_widget(start)
        p2 = self.image_to_widget(end)
        color = QColor("#17a8cb") if orientation == "horizontal" else QColor("#9275db")
        pen = QPen(color, 3 if not preview else 2, Qt.DashLine if preview else Qt.SolidLine)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(p1, 5, 5)
        painter.drawEllipse(p2, 5, 5)
        mid = (p1 + p2) / 2
        painter.setPen(QPen(color, 1))
        label = "수평 기준선" if orientation == "horizontal" else "수직 기준선"
        painter.drawText(mid + QPointF(8, -8), label)

    def _draw_magnifier(self, painter: QPainter):
        if self.pixmap.isNull():
            return
        self._transform()
        cursor = self._last_mouse
        image_point = self.widget_to_image(cursor)
        diameter = 154.0
        zoom = self._magnifier_factor
        source_side = diameter / max(zoom * self._scale, 0.001)
        image_w, image_h = self.pixmap.width(), self.pixmap.height()
        source_side = min(source_side, image_w, image_h)
        left = min(max(0.0, float(image_point[0]) - source_side / 2), image_w - source_side)
        top = min(max(0.0, float(image_point[1]) - source_side / 2), image_h - source_side)
        src = QRectF(left, top, source_side, source_side)

        mag_x = cursor.x() + 28
        mag_y = cursor.y() + 28
        if mag_x + diameter > self.width() - 10:
            mag_x = cursor.x() - diameter - 28
        if mag_y + diameter > self.height() - 10:
            mag_y = cursor.y() - diameter - 28
        mag_x = min(max(10.0, mag_x), self.width() - diameter - 10)
        mag_y = min(max(10.0, mag_y), self.height() - diameter - 10)
        target = QRectF(mag_x, mag_y, diameter, diameter)

        path = QPainterPath()
        path.addEllipse(target)
        painter.save()
        painter.setClipPath(path)
        painter.fillRect(target, QColor("#ffffff"))
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(target, self.pixmap, src)
        center = target.center()
        painter.setPen(QPen(QColor(255, 83, 131, 210), 1))
        painter.drawLine(QPointF(center.x() - 18, center.y()), QPointF(center.x() + 18, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - 18), QPointF(center.x(), center.y() + 18))
        painter.restore()
        painter.setPen(QPen(QColor("#ffffff"), 5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(target)
        painter.setPen(QPen(QColor("#726ed1"), 3))
        painter.drawEllipse(target)
        painter.setPen(QColor("#55548f"))
        painter.drawText(QRectF(mag_x - 3, mag_y + diameter + 2, diameter + 6, 20),
                         Qt.AlignCenter, "3× 확대 · 정밀 모서리 지정")

    def mousePressEvent(self, event: QMouseEvent):
        self._last_mouse = event.position()
        if self.pixmap.isNull():
            return
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_anchor = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return

        if self.ruler_mode:
            self._ruler_start = self.widget_to_image(event.position())
            self._ruler_current = self._ruler_start.copy()
            self.update()
            return

        if self.add_mode:
            self.draft.append(self.widget_to_image(event.position()))
            if len(self.draft) == 4:
                points = np.array(self.draft, dtype=np.float32)
                self.polygon_created.emit(points)
                self.draft.clear()
                self.add_mode = False
                self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        if 0 <= self.selected < len(self.regions):
            for corner_idx, point in enumerate(self.regions[self.selected]):
                q = self.image_to_widget(point)
                if (q - event.position()).manhattanLength() < 18:
                    self.drag_corner = (self.selected, corner_idx)
                    self.update()
                    return
        for idx in reversed(range(len(self.regions))):
            polygon = QPolygonF([self.image_to_widget(point) for point in self.regions[idx]])
            if polygon.containsPoint(event.position(), Qt.OddEvenFill):
                self.region_selected.emit(idx)
                return

    def mouseMoveEvent(self, event: QMouseEvent):
        self._last_mouse = event.position()
        if self._pan_anchor is not None:
            delta = event.position() - self._pan_anchor
            self._pan += delta
            self._pan_anchor = event.position()
            self._transform()
            self.update()
            return
        if self.ruler_mode and self._ruler_start is not None:
            self._ruler_current = self.widget_to_image(event.position())
            self.update()
            return
        if self.drag_corner is not None:
            idx, corner = self.drag_corner
            points = self.regions[idx].copy()
            points[corner] = self.widget_to_image(event.position())
            self.regions[idx] = points
            self.polygon_changed.emit(idx, points)
            self.update()
            return
        if self.add_mode:
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._pan_anchor = None
            self.setCursor(Qt.CrossCursor if (self.add_mode or self.ruler_mode) else Qt.ArrowCursor)
            return
        if event.button() == Qt.LeftButton and self.ruler_mode and self._ruler_start is not None:
            end = self.widget_to_image(event.position())
            start = self._ruler_start.copy()
            orientation = self.ruler_mode
            self._ruler_start = None
            self._ruler_current = None
            self.ruler_mode = None
            self.setCursor(Qt.ArrowCursor)
            if float(np.linalg.norm(end - start)) >= 5:
                self.ruler_created.emit(start, end, orientation)
            self.update()
            return
        if event.button() == Qt.LeftButton:
            self.drag_corner = None
            self.update()

    def wheelEvent(self, event: QWheelEvent):
        self._last_mouse = event.position()
        self._zoom_at(event.position(), event.angleDelta().y())
        event.accept()

    def leaveEvent(self, event):
        self._last_mouse = QPointF(-1000, -1000)
        self.update()
        super().leaveEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(
            url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in
            {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                    self.image_dropped.emit(path)
                    event.acceptProposedAction()
                    return
        event.ignore()
