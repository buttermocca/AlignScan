import numpy as np

from alignscan.ai_detector import OnnxDocumentDetector


def test_yolo_detect_output_channel_first_is_normalized():
    output = np.zeros((1, 5, 18900), dtype=np.float32)
    rows = OnnxDocumentDetector._rows_from_output(output)
    assert rows.shape == (18900, 5)


def test_yolo_detect_output_channel_last_is_preserved():
    output = np.zeros((1, 18900, 5), dtype=np.float32)
    rows = OnnxDocumentDetector._rows_from_output(output)
    assert rows.shape == (18900, 5)


def test_letterbox_matches_960_training_shape():
    image = np.zeros((1200, 600, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = OnnxDocumentDetector._letterbox(image, 960, 960)
    assert canvas.shape == (960, 960, 3)
    assert abs(scale - 0.8) < 1e-6
    assert pad_x == 240
    assert pad_y == 0
    assert np.all(canvas[:, :pad_x] == 114)


def test_overlapping_windows_cover_side_by_side_and_stacked_scan():
    windows = OnnxDocumentDetector._overlapping_windows(2527, 3369)
    assert len(windows) == 4
    names = [window[4] for window in windows]
    assert names.count("세로 분할") == 2
    assert names.count("가로 분할") == 2
    for x, y, width, height, _ in windows:
        assert x >= 0 and y >= 0
        assert width > 0 and height > 0
        assert x + width <= 2527
        assert y + height <= 3369


def test_small_image_does_not_create_unnecessary_tiles():
    windows = OnnxDocumentDetector._overlapping_windows(700, 900)
    assert windows == []
