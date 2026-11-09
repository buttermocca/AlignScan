import numpy as np
from alignscan.geometry import detect_document_regions, rectify_document


def test_detect_two_papers_and_rectify():
    image = np.full((900, 700, 3), 245, dtype=np.uint8)
    image[100:780, 80:300] = 255
    image[160:700, 390:630] = 252
    import cv2
    cv2.rectangle(image, (80, 100), (299, 779), (120, 120, 120), 2)
    cv2.rectangle(image, (390, 160), (629, 699), (120, 120, 120), 2)
    found = detect_document_regions(image)
    assert len(found) >= 2
    result = rectify_document(image, found[0].corners)
    assert result.shape[0] > 100 and result.shape[1] > 100
