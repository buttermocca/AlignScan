import numpy as np

from alignscan.ai_detector import OnnxDocumentDetector


def test_yolo_detect_output_channel_first_is_normalized():
    output = np.zeros((1, 5, 8400), dtype=np.float32)
    rows = OnnxDocumentDetector._rows_from_output(output)
    assert rows.shape == (8400, 5)


def test_yolo_detect_output_channel_last_is_preserved():
    output = np.zeros((1, 8400, 5), dtype=np.float32)
    rows = OnnxDocumentDetector._rows_from_output(output)
    assert rows.shape == (8400, 5)


def test_letterbox_matches_960_training_shape():
    image = np.zeros((1200, 600, 3), dtype=np.uint8)
    canvas, scale, pad_x, pad_y = OnnxDocumentDetector._letterbox(image, 960, 960)
    assert canvas.shape == (960, 960, 3)
    assert abs(scale - 0.8) < 1e-6
    assert pad_x == 240
    assert pad_y == 0
    assert np.all(canvas[:, :pad_x] == 114)
