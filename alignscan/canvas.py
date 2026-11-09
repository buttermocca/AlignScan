from __future__ import annotations

from pathlib import Path
import math
import numpy as np

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (QPainter, QPen, QBrush, QPixmap, QImage, QColor, QPolygonF,
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
        self._magnifier_factor = 4.5

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

    def _clamped_image_point(self, point: QPointF):
        result = self.widget_to_image(point)
        if not self.pixmap.isNull():
            result[0] = np.clip(result[0], 0, self.pixmap.width() - 1)
            result[1] = np.clip(result[1], 0, self.pixmap.height() - 1)
        return result

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
        painter.fillRect(self.rect(), QColor("#eff3ee"))
        painter.setPen(QPen(QColor("#dce6dc"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(self.rect().adjusted(8, 8, -8, -8), 18, 18)
        if self.pixmap.isNull():
            painter.setPen(QColor("#72877a"))
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
        painter.setBrush(QColor(89, 112, 97, 205))
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
        """Draw a cursor-centred loupe, preserving white space beyond image edges."""
        if self.pixmap.isNull():
            return
        self._transform()
        cursor = self._last_mouse
        image_point = self.widget_to_image(cursor)

        diameter = 188.0
        zoom = self._magnifier_factor
        source_side = diameter / max(zoom * self._scale, 0.001)
        source_side = max(24.0, source_side)

        # Keep the cursor's image coordinate exactly in the centre of the source
        # square. Do NOT clamp the source rectangle to the image bounds: doing so
        # shifts the receipt edge out of the lens when the cursor is near an edge.
        source_left = float(image_point[0]) - source_side / 2.0
        source_top = float(image_point[1]) - source_side / 2.0
        source_rect = QRectF(source_left, source_top, source_side, source_side)

        image = self.pixmap.toImage().convertToFormat(QImage.Format_RGB32)
        patch_size = max(2, int(round(source_side)))
        patch = QImage(patch_size, patch_size, QImage.Format_RGB32)
        patch.fill(QColor("#fffefa"))
        image_bounds = QRectF(0, 0, image.width(), image.height())
        intersection = source_rect.intersected(image_bounds)
        if not intersection.isEmpty():
            # Map the valid source intersection into the same location inside
            # the square patch. Areas outside the scan remain soft white, while
            # the crosshair stays locked on the actual cursor point.
            target_left = (intersection.left() - source_rect.left()) / source_side * patch_size
            target_top = (intersection.top() - source_rect.top()) / source_side * patch_size
            target_width = intersection.width() / source_side * patch_size
            target_height = intersection.height() / source_side * patch_size
            patch_painter = QPainter(patch)
            patch_painter.setRenderHint(QPainter.SmoothPixmapTransform)
            patch_painter.drawImage(
                QRectF(target_left, target_top, target_width, target_height),
                image, intersection)
            patch_painter.end()

        # Put the loupe opposite the mouse pointer. This is important for a
        # receipt that sits at the right edge of the scanner sheet: the circle
        # moves to the left instead of covering the edge or being clipped.
        gap = 22.0
        mag_x = cursor.x() - diameter - gap if cursor.x() > self.width() * 0.52 else cursor.x() + gap
        mag_y = cursor.y() - diameter - gap if cursor.y() > self.height() * 0.56 else cursor.y() + gap
        mag_x = min(max(10.0, mag_x), max(10.0, self.width() - diameter - 10.0))
        mag_y = min(max(10.0, mag_y), max(10.0, self.height() - diameter - 28.0))
        target = QRectF(mag_x, mag_y, diameter, diameter)

        clip = QPainterPath()
        clip.addEllipse(target)
        painter.save()
        painter.setClipPath(clip)
        painter.fillRect(target, QColor("#fffefa"))
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(target, patch, QRectF(patch.rect()))
        center = target.center()
        painter.setPen(QPen(QColor(239, 113, 143, 225), 1.5))
        painter.drawLine(QPointF(center.x() - 16, center.y()),
                         QPointF(center.x() + 16, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - 16),
                         QPointF(center.x(), center.y() + 16))
        painter.restore()

        painter.setPen(QPen(QColor("#ffffff"), 5))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(target)
        painter.setPen(QPen(QColor("#a9bda9"), 3))
        painter.drawEllipse(target)
        label_rect = QRectF(mag_x - 8, mag_y + diameter + 2, diameter + 16, 20)
        painter.setPen(QColor("#60776a"))
        painter.drawText(label_rect, Qt.AlignCenter,
                         f"{zoom:g}× 확대 · 모서리 정밀 확인")

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
            self._ruler_start = self._clamped_image_point(event.position())
            self._ruler_current = self._ruler_start.copy()
            self.update()
            return

        if self.add_mode:
            self.draft.append(self._clamped_image_point(event.position()))
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
            self._ruler_current = self._clamped_image_point(event.position())
            self.update()
            return
        if self.drag_corner is not None:
            idx, corner = self.drag_corner
            points = self.regions[idx].copy()
            points[corner] = self._clamped_image_point(event.position())
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
            end = self._clamped_image_point(event.position())
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
