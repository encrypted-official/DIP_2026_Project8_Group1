def heading_to_axis(heading_deg):
    """Fold heading into the undirected axis range [0, 90]."""
    r = float(heading_deg) % 180.0
    return min(r, 180.0 - r)

def wrap_axis(axis_deg):
    """Normalize an axis angle to [0, 180)."""
    return float(axis_deg) % 180.0

def axis_error(pred_deg, true_deg):
    """Shortest angular difference between undirected axes."""
    if pred_deg is None or true_deg is None:
        return None

    pred = wrap_axis(pred_deg)
    true = wrap_axis(true_deg)
    diff = abs(pred - true)
    return min(diff, 180.0 - diff)