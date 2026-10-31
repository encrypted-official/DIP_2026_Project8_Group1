def heading_to_axis(heading_deg):
    """Legacy folded axis in [0, 90]; retained for old reports/tests."""
    r = float(heading_deg) % 180.0
    return min(r, 180.0 - r)

def heading_to_axis180(heading_deg):
    """Convert directed heading to an undirected axis in [0, 180)."""
    return float(heading_deg) % 180.0

def wrap_axis(axis_deg):
    """Normalize an undirected axis angle to [0, 180)."""
    return float(axis_deg) % 180.0

def axis_error(pred_deg, true_deg):
    """Shortest angular difference between undirected axes."""
    if pred_deg is None or true_deg is None:
        return None
    pred = wrap_axis(pred_deg)
    true = wrap_axis(true_deg)
    diff = abs(pred - true)
    return min(diff, 180.0 - diff)

def heading_direction_label(heading_deg, axis_deg):
    """Return 1 when heading points along axis_deg, 0 for the opposite end."""
    import math
    delta = math.radians(float(heading_deg) - float(axis_deg))
    return int(math.cos(delta) >= 0.0)