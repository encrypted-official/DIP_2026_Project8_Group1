import math
import numpy as np
import cv2

from src.angle_utils import axis_and_sign_to_heading, directed_angle_error
from src.heading_direction import (
    rotate_axis_to_horizontal, extract_direction_features,
    heading_sign_from_ground_truth, circular_heading_error, estimate_heading_sign,
)


def test_directed_heading_conversion_and_error():
    assert axis_and_sign_to_heading(35, 1) == 35
    assert axis_and_sign_to_heading(35, 0) == 215
    assert directed_angle_error(359, 1) == 2
    assert directed_angle_error(10, 190) == 180


def test_heading_sign_label():
    assert heading_sign_from_ground_truth(20, 20) == 1
    assert heading_sign_from_ground_truth(200, 20) == 0


def test_rotation_maps_positive_axis_to_right():
    for angle in (0, 30, 45, 90, 135):
        image = np.zeros((81, 101), dtype=np.uint8)
        theta = math.radians(angle)
        cx, cy = 50, 40
        x = int(round(cx + 20 * math.cos(theta)))
        y = int(round(cy + 20 * math.sin(theta)))
        image[y-2:y+3, x-2:x+3] = 255
        rotated = rotate_axis_to_horizontal(image, angle)
        ys, xs = np.where(rotated > 0)
        assert xs.mean() > (rotated.shape[1] - 1) / 2


def test_direction_features_are_finite():
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    cv2.rectangle(image, (8, 20), (88, 44), (120, 120, 120), -1)
    cv2.rectangle(image, (8, 20), (24, 44), (230, 230, 230), -1)
    features = extract_direction_features(image, 0)
    assert features
    assert all(np.isfinite(v) for v in features.values())


def test_circular_heading_error_wraps():
    assert circular_heading_error(2, 358) == 4


def test_rule_based_direction_returns_valid_sign_or_abstains():
    image = np.zeros((64, 120, 3), dtype=np.uint8)
    # A simple vehicle-like body with a narrower left terminal.
    cv2.rectangle(image, (12, 20), (108, 44), (180, 180, 180), -1)
    cv2.fillPoly(image, [np.array([[5, 32], [20, 20], [20, 44]], dtype=np.int32)], (180, 180, 180))
    sign, confidence, features = estimate_heading_sign(image, 0)
    assert sign in (0, 1, None)
    assert 0.0 <= confidence <= 1.0
    assert features
