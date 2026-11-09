from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class DocumentRegion:
    corners: np.ndarray
    label: str = "영역"

    def __post_init__(self) -> None:
        self.corners = order_points(np.asarray(self.corners, dtype=np.float32))


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)], pts[np.argmin(d)],
        pts[np.argmax(s)], pts[np.argmax(d)],
    ], dtype=np.float32)


def _bbox_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ax1, ay1 = np.min(a, axis=0); ax2, ay2 = np.max(a, axis=0)
    bx1, by1 = np.min(b, axis=0); bx2, by2 = np.max(b, axis=0)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    iou = inter / max(aa + ba - inter, 1.0)
    containment = inter / max(min(aa, ba), 1.0)
    return iou, containment


def detect_document_regions(image: np.ndarray, min_area_ratio: float = 0.012) -> list[DocumentRegion]:
    """Find several paper/card outlines, including faint edges on light backgrounds."""
    if image is None or image.size == 0:
        return []
    h0, w0 = image.shape[:2]
    scale = min(1.0, 1800.0 / max(h0, w0))
    work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    blur = cv2.GaussianBlur(gray, (3, 3), 0.8)
    image_area = float(h * w)
    candidates: list[tuple[float, np.ndarray]] = []

    # Low thresholds are intentional: scanned white paper can be only slightly
    # brighter than the scanner background. Several thresholds improve recall.
    for low, high in ((5, 15), (8, 24), (12, 36), (18, 54), (25, 75)):
        edges = cv2.Canny(blur, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * min_area_ratio or area > image_area * 0.94:
                continue
            peri = cv2.arcLength(contour, True)
            for epsilon in (0.012, 0.018, 0.025, 0.035):
                approx = cv2.approxPolyDP(contour, epsilon * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                pts = approx.reshape(4, 2).astype(np.float32)
                rect = cv2.minAreaRect(pts)
                rw, rh = rect[1]
                if min(rw, rh) < 30:
                    continue
                ratio = min(rw, rh) / max(rw, rh)
                if ratio < 0.18 or ratio > 0.96:
                    continue
                x, y = np.min(pts, axis=0); xx, yy = np.max(pts, axis=0)
                if x < 2 and y < 2 and xx > w - 3 and yy > h - 3:
                    continue
                score = area * (0.85 + 0.15 * min(ratio / 0.5, 1.0))
                candidates.append((score, pts))
                break

    if not candidates:
        for block, c in ((41, 5), (61, 7), (81, 9)):
            block = min(block, min(h, w) // 2 * 2 - 1)
            if block < 3:
                continue
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, block, c)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=2)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < image_area * min_area_ratio or area > image_area * 0.94:
                    continue
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    pts = approx.reshape(4, 2).astype(np.float32)
                    candidates.append((area, pts))

    candidates.sort(key=lambda item: item[0], reverse=True)
    kept: list[np.ndarray] = []
    for _, pts in candidates:
        duplicate = False
        for existing in kept:
            iou, containment = _bbox_overlap(pts, existing)
            if iou > 0.48 or containment > 0.78:
                duplicate = True
                break
        if duplicate:
            continue
        if scale < 1:
            pts = pts / scale
        kept.append(order_points(pts))
    kept.sort(key=lambda p: (float(np.min(p[:, 1])), float(np.min(p[:, 0]))))
    return [DocumentRegion(points, f"문서 {i + 1}") for i, points in enumerate(kept)]


def rectify_document(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    pts = order_points(corners)
    tl, tr, br, bl = pts
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    out_w = max(2, int(max(width_a, width_b)))
    out_h = max(2, int(max(height_a, height_b)))
    destination = np.array([[0, 0], [out_w - 1, 0],
                            [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, destination)
    return cv2.warpPerspective(image, matrix, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
