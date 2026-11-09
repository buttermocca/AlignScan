import cv2
import numpy as np

from alignscan.geometry import detect_document_regions, rectify_document
from alignscan.imaging import enhance_image


def sample_scan():
    image = np.full((900, 700, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (80, 100), (299, 779), (120, 120, 120), 2)
    cv2.rectangle(image, (390, 160), (629, 699), (120, 120, 120), 2)
    return image


def test_detect_two_papers_and_rectify():
    image = sample_scan()
    found = detect_document_regions(image, sensitivity=62)
    assert len(found) == 2
    result = rectify_document(image, found[0].corners)
    assert result.shape[0] > 100 and result.shape[1] > 100


def test_sensitivity_setting_is_accepted():
    image = sample_scan()
    for sensitivity in (0, 25, 62, 90, 100):
        found = detect_document_regions(image, sensitivity=sensitivity)
        assert isinstance(found, list)


def test_sharpness_slider_changes_output():
    image = sample_scan()
    sharpened = enhance_image(image, sharpness=65)
    assert sharpened.shape == image.shape
    assert np.any(sharpened != image)


def test_fast_preview_and_full_export_paths():
    image = sample_scan()
    preview = enhance_image(image, denoise=45, dust=20, sharpness=30, fast=True)
    export = enhance_image(image, sharpness=30, fast=False)
    assert preview.shape == image.shape
    assert export.shape == image.shape
