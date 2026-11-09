import math
import cv2
import numpy as np

from alignscan.transforms import (
    rotate_expand, rotate_quarter, flip_image, transform_points,
    ruler_correction_angle,
)


def test_quarter_turn_transforms_keep_all_corners_inside():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    corners = np.array([[0, 0], [119, 0], [119, 79], [0, 79]], dtype=np.float32)
    for angle in (90, 180, 270):
        rotated, matrix = rotate_quarter(image, angle)
        mapped = transform_points(corners, matrix)
        h, w = rotated.shape[:2]
        assert np.all(mapped[:, 0] >= 0)
        assert np.all(mapped[:, 0] <= w - 1)
        assert np.all(mapped[:, 1] >= 0)
        assert np.all(mapped[:, 1] <= h - 1)


def test_flip_image_coordinates_match_pixels():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    point = np.array([[3, 7]], dtype=np.float32)
    for horizontal in (True, False):
        flipped, matrix = flip_image(image, horizontal)
        mapped = transform_points(point, matrix)[0]
        expected = [26, 7] if horizontal else [3, 12]
        assert np.allclose(mapped, expected)
        assert flipped.shape == image.shape


def test_ruler_deskew_angles_align_horizontal_and_vertical():
    horizontal_start = np.array([50.0, 60.0])
    horizontal_end = np.array([250.0, 80.0])
    horizontal_angle = ruler_correction_angle(horizontal_start, horizontal_end, "horizontal")
    assert horizontal_angle > 0
    rotated, matrix = rotate_expand(np.zeros((200, 300, 3), dtype=np.uint8), horizontal_angle)
    mapped = transform_points(np.array([horizontal_start, horizontal_end]), matrix)
    assert abs(float(mapped[1, 1] - mapped[0, 1])) < 1.0

    vertical_start = np.array([50.0, 30.0])
    vertical_end = np.array([70.0, 230.0])
    vertical_angle = ruler_correction_angle(vertical_start, vertical_end, "vertical")
    _, matrix = rotate_expand(np.zeros((300, 200, 3), dtype=np.uint8), vertical_angle)
    mapped = transform_points(np.array([vertical_start, vertical_end]), matrix)
    assert abs(float(mapped[1, 0] - mapped[0, 0])) < 1.0


def test_arbitrary_rotation_expands_canvas_and_preserves_point_bounds():
    image = np.zeros((100, 180, 3), dtype=np.uint8)
    points = np.array([[0, 0], [179, 0], [179, 99], [0, 99]], dtype=np.float32)
    rotated, matrix = rotate_expand(image, 17.5)
    mapped = transform_points(points, matrix)
    h, w = rotated.shape[:2]
    assert w > image.shape[1]
    assert h > image.shape[0]
    assert np.all(mapped[:, 0] >= -0.01)
    assert np.all(mapped[:, 0] <= w + 0.01)
    assert np.all(mapped[:, 1] >= -0.01)
    assert np.all(mapped[:, 1] <= h + 0.01)
