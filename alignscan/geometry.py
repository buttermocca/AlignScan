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


def detect_document_regions(image: np.ndarray, sensitivity: int = 62,
                             min_area_ratio: float | None = None) -> list[DocumentRegion]:
    """Detect multiple paper/card outlines. Higher sensitivity accepts fainter/smaller edges."""
    if image is None or image.size == 0:
        return []
    sensitivity = int(np.clip(sensitivity, 0, 100))
    h0, w0 = image.shape[:2]
    # Process at a bounded size so large scanner images remain responsive.
    scale = min(1.0, 1600.0 / max(h0, w0))
    work = cv2.resize(image, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    sigma = 0.75 + (100 - sensitivity) * 0.004
    blur = cv2.GaussianBlur(gray, (3, 3), sigma)
    low = int(round(3 + (100 - sensitivity) * 0.05))
    high = int(round(low * 3.0))
    kernel_size = 7 if sensitivity >= 55 else 9
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    h, w = gray.shape[:2]
    image_area = float(h * w)
    if min_area_ratio is None:
        min_area_ratio = 0.020 - sensitivity * 0.00016  # 2.0% -> 0.4%
    min_area_ratio = float(np.clip(min_area_ratio, 0.002, 0.05))
    candidates: list[tuple[float, np.ndarray]] = []

    # A small three-level edge sweep handles faint paper boundaries while
    # avoiding the much slower five-pass/four-approximation original detector.
    for offset in (0, 5, 10):
        edges = cv2.Canny(blur, max(2, low + offset), max(8, high + offset * 3), L2gradient=False)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * min_area_ratio or area > image_area * 0.94:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                rect = cv2.minAreaRect(contour)
                rw, rh = rect[1]
                if min(rw, rh) < 30 or max(rw, rh) <= 0:
                    continue
                rectangularity = area / max(rw * rh, 1.0)
                if rectangularity < 0.78:
                    continue
                pts = cv2.boxPoints(rect).astype(np.float32)
            else:
                pts = approx.reshape(4, 2).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            rw, rh = rect[1]
            if min(rw, rh) < 30 or max(rw, rh) <= 0:
                continue
            ratio = min(rw, rh) / max(rw, rh)
            if ratio < 0.16 or ratio > 0.98:
                continue
            x, y = np.min(pts, axis=0); xx, yy = np.max(pts, axis=0)
            if x < 2 and y < 2 and xx > w - 3 and yy > h - 3:
                continue
            score = area * (0.9 + 0.1 * min(ratio / 0.5, 1.0))
            candidates.append((score, pts))

    # Only invoke adaptive thresholding on failed edge passes.
    if not candidates and sensitivity >= 45:
        block = min(51, min(h, w) // 2 * 2 - 1)
        if block >= 3:
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, block, 7)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < image_area * min_area_ratio or area > image_area * 0.94:
                    continue
                rect = cv2.minAreaRect(contour)
                rw, rh = rect[1]
                if min(rw, rh) < 30 or max(rw, rh) <= 0:
                    continue
                if area / max(rw * rh, 1.0) < 0.8:
                    continue
                candidates.append((area, cv2.boxPoints(rect).astype(np.float32)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    kept: list[np.ndarray] = []
    for _, pts in candidates:
        duplicate = False
        for existing in kept:
            iou, containment = _bbox_overlap(pts, existing)
            if iou > 0.30 or containment > 0.62:
                duplicate = True
                break
        if duplicate:
            continue
        if scale < 1:
            pts = pts / scale
        kept.append(order_points(pts))
    kept.sort(key=lambda p: (float(np.min(p[:, 1])), float(np.min(p[:, 0]))))
    return [DocumentRegion(points, f"문서 {i + 1}") for i, points in enumerate(kept)]


def rectify_document(image: np.ndarray, corners: np.ndarray,
                     interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
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
    return cv2.warpPerspective(image, matrix, (out_w, out_h), flags=interpolation,
                               borderMode=cv2.BORDER_REPLICATE)
