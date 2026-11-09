from __future__ import annotations

import math
import cv2
import numpy as np


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine transform to any (..., 2) image-coordinate points."""
    pts = np.asarray(points, dtype=np.float32)
    shape = pts.shape
    flat = pts.reshape(-1, 2)
    ones = np.ones((len(flat), 1), dtype=np.float32)
    mapped = np.hstack([flat, ones]) @ np.asarray(matrix, dtype=np.float32).T
    return mapped.reshape(shape)


def rotate_expand(image: np.ndarray, angle_degrees: float,
                  interpolation: int = cv2.INTER_CUBIC) -> tuple[np.ndarray, np.ndarray]:
    """Rotate visually counter-clockwise by degrees, expanding canvas to avoid cropping."""
    if image is None or image.size == 0:
        raise ValueError("회전할 이미지가 없습니다.")
    h, w = image.shape[:2]
    angle = float(angle_degrees)
    if abs(angle) < 1e-7:
        return image.copy(), np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = max(1, int(math.ceil(h * sin + w * cos)))
    new_h = max(1, int(math.ceil(h * cos + w * sin)))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h),
                             flags=interpolation, borderMode=cv2.BORDER_REPLICATE)
    return rotated, matrix.astype(np.float32)


def rotate_quarter(image: np.ndarray, clockwise_degrees: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact 90-degree rotation. clockwise_degrees must be 90, 180 or 270."""
    h, w = image.shape[:2]
    angle = int(clockwise_degrees) % 360
    if angle == 90:
        result = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        matrix = np.array([[0, -1, h - 1], [1, 0, 0]], dtype=np.float32)
    elif angle == 180:
        result = cv2.rotate(image, cv2.ROTATE_180)
        matrix = np.array([[-1, 0, w - 1], [0, -1, h - 1]], dtype=np.float32)
    elif angle == 270:
        result = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        matrix = np.array([[0, 1, 0], [-1, 0, w - 1]], dtype=np.float32)
    elif angle == 0:
        result = image.copy()
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    else:
        raise ValueError("90, 180 또는 270도 회전만 지원합니다.")
    return result, matrix


def flip_image(image: np.ndarray, horizontal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    if horizontal:
        return cv2.flip(image, 1), np.array([[-1, 0, w - 1], [0, 1, 0]], dtype=np.float32)
    return cv2.flip(image, 0), np.array([[1, 0, 0], [0, -1, h - 1]], dtype=np.float32)


def ruler_correction_angle(start: np.ndarray, end: np.ndarray, orientation: str) -> float:
    """Return the OpenCV rotation angle that aligns a drawn ruler to H or V."""
    p1 = np.asarray(start, dtype=np.float32).reshape(2)
    p2 = np.asarray(end, dtype=np.float32).reshape(2)
    delta = p2 - p1
    if float(np.linalg.norm(delta)) < 2.0:
        raise ValueError("룰러 선을 조금 더 길게 그려주세요.")
    angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    if orientation == "horizontal":
        correction = angle
    elif orientation == "vertical":
        correction = angle - 90.0
        # A vertical rule is equivalent modulo 180 degrees.
        while correction <= -90:
            correction += 180
        while correction > 90:
            correction -= 180
    else:
        raise ValueError("룰러 방향은 horizontal 또는 vertical이어야 합니다.")
    # Keep the adjustment small; a ruler guides deskew, not page rotation.
    while correction <= -90:
        correction += 180
    while correction > 90:
        correction -= 180
    return float(correction)
