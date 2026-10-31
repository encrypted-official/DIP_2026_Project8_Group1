from src.angle_utils import (
    heading_to_axis,
    heading_to_axis180,
    axis_error,
    heading_direction_label)

def test_legacy_heading_fold():
    assert heading_to_axis(42) == 42
    assert heading_to_axis(137) == 43
    assert heading_to_axis(225) == 45
    assert heading_to_axis(315) == 45

def test_axis_heading_uses_full_undirected_range():
    assert heading_to_axis180(135) == 135
    assert heading_to_axis180(225) == 45

def test_axis_error_wraparound():
    assert axis_error(179, 1) == 2
    assert axis_error(10, 190) == 0
    assert axis_error(135, 45) == 90
    assert axis_error(20, 25) == 5


def test_heading_direction_label():
    assert heading_direction_label(10, 10) == 1
    assert heading_direction_label(190, 10) == 0