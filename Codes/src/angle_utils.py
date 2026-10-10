"""Small helpers for the angle convention used in this project."""


def heading_to_axis(heading_deg):
    """Fold a 0–360 degree front heading into an undirected 0–90 degree axis."""
    turn = float(heading_deg) % 180.0
    return min(turn, 180.0 - turn)


def wrap_axis(axis_deg):
    """Return an undirected axis angle in the range 0–180 degrees."""
    return float(axis_deg) % 180.0


def axis_error(pred_deg, true_deg):
    """Shortest angular distance between two undirected axes, in degrees."""
    if pred_deg is None or true_deg is None:
        return None

    diff = abs(wrap_axis(pred_deg) - wrap_axis(true_deg))
    return min(diff, 180.0 - diff)
