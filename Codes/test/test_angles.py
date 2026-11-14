from src.angle_utils import heading_to_axis, axis_error, axis_and_sign_to_heading, directed_angle_error

def test_heading_to_undirected_axis():
    assert heading_to_axis(42) == 42
    assert heading_to_axis(137) == 137
    assert heading_to_axis(225) == 45
    assert heading_to_axis(315) == 135
    assert heading_to_axis(360) == 0

def test_axis_error_wraparound():
    assert axis_error(179, 1) == 2
    assert axis_error(10, 190) == 0
    assert axis_error(20, 25) == 5
    assert axis_error(135, 45) == 90

def test_axis_and_sign_to_directed_heading():
    assert axis_and_sign_to_heading(35, 1) == 35
    assert axis_and_sign_to_heading(35, 0) == 215

def test_directed_heading_error():
    assert directed_angle_error(359, 1) == 2
    assert directed_angle_error(10, 190) == 180