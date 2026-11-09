from __future__ import annotations
import cv2
import numpy as np


def enhance_image(image: np.ndarray, brightness: int = 0, contrast: int = 100,
                  saturation: int = 100, denoise: int = 0, dust: int = 0,
                  sharpness: int = 0, fast: bool = False) -> np.ndarray:
    """Apply brightness, contrast, saturation, cleanup and unsharp-mask sharpening."""
    if image is None or image.size == 0:
        return image
    out = image.copy()
    if denoise > 0 and fast:
        # A blended 3x3 Gaussian is intentionally used for the live preview;
        # the real export path below retains the stronger NLM filter.
        blur = cv2.GaussianBlur(out, (3, 3), 0.65)
        blend = min(0.72, denoise / 145.0)
        out = cv2.addWeighted(out, 1.0 - blend, blur, blend, 0)
    elif denoise > 0:
        strength = max(1, int(2 + denoise * 0.20))
        out = cv2.fastNlMeansDenoisingColored(out, None, strength, strength, 7, 21)
    if dust > 0 and fast:
        # Approximate isolated dust removal for a responsive preview. The
        # saved image still uses the conservative spot mask + inpaint below.
        median = cv2.medianBlur(out, 3)
        blend = min(0.8, dust / 100.0)
        out = cv2.addWeighted(out, 1.0 - blend, median, blend, 0)
    elif dust > 0:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        k = max(7, int(7 + dust * 0.10) | 1)
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        dark_spots = cv2.subtract(background, gray)
        threshold = max(16, 54 - int(dust * 0.32))
        _, mask = cv2.threshold(dark_spots, threshold, 255, cv2.THRESH_BINARY)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean_mask = np.zeros_like(mask)
        max_area = max(3, int(2 + dust * 0.75))
        for idx in range(1, count):
            x, y, w, h, area = stats[idx]
            if 1 <= area <= max_area and max(w, h) <= 14:
                clean_mask[labels == idx] = 255
        if cv2.countNonZero(clean_mask):
            out = cv2.inpaint(out, clean_mask, 2 + dust / 50.0, cv2.INPAINT_TELEA)
    alpha = max(0.0, contrast / 100.0)
    beta = float(brightness) * 1.35
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if saturation != 100:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (saturation / 100.0), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if sharpness > 0:
        amount = min(1.35, float(sharpness) / 75.0)
        blurred = cv2.GaussianBlur(out, (0, 0), 1.0 if fast else 1.25)
        out = cv2.addWeighted(out, 1.0 + amount, blurred, -amount, 0)
    return out
