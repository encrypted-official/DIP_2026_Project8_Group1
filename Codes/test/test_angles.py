from src.angle_utils import heading_to_axis, axis_error


def test_heading_fold():
    assert heading_to_axis(42) == 42
    assert heading_to_axis(137) == 43
    assert heading_to_axis(225) == 45
    assert heading_to_axis(315) == 45


def test_axis_error_wraparound():
    assert axis_error(179, 1) == 2
    assert axis_error(10, 190) == 0
    assert axis_error(20, 25) == 5
