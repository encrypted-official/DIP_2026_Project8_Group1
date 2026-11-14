def heading_to_axis(heading_deg):
    """Convert a directed heading to its undirected axis in [0, 180).

    An axis is periodic over 180 degrees. Do not fold it to [0, 90]:
    45 and 135 degrees are different line orientations.
    """
    return float(heading_deg) % 180.0

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

def heading_direction_label(heading_deg, axis_deg):
    """Return 1 if the positive axis direction agrees with directed heading."""
    import math
    delta = math.radians(float(heading_deg) - float(axis_deg))
    return int(math.cos(delta) >= 0.0)



def directed_angle_error(pred_deg, true_deg):
    """Shortest circular difference between directed headings in [0, 360)."""
    if pred_deg is None or true_deg is None:
        return None
    import math
    if not (math.isfinite(float(pred_deg)) and math.isfinite(float(true_deg))):
        return None
    return abs((float(pred_deg) - float(true_deg) + 180.0) % 360.0 - 180.0)


def axis_and_sign_to_heading(axis_deg, heading_sign):
    """Convert an undirected [0,180) axis and sign into a [0,360) heading.

    heading_sign=1 means the positive axis vector (cos(axis), sin(axis))
    agrees with the directed heading; heading_sign=0 means the opposite end.
    Angles use image coordinates: x right, y down, matching atan2(y, x).
    """
    import math
    if axis_deg is None or heading_sign is None:
        return None
    if not math.isfinite(float(axis_deg)):
        return None
    sign = int(heading_sign)
    if sign not in (0, 1):
        raise ValueError("heading_sign must be 0 or 1")
    return (float(axis_deg) % 180.0 + (0.0 if sign == 1 else 180.0)) % 360.0