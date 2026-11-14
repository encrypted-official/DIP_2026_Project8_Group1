"""Deterministic, rule-based directed-heading estimation using OpenCV only.

The undirected axis is rotated horizontally. A foreground silhouette is
estimated with Otsu thresholding, then terminal widths are compared. The
narrower/tapered end is selected as the presumed front. This is a geometric
heuristic, not a learned model; symmetric or ambiguous crops remain unreliable.
"""
from __future__ import annotations
import math
from typing import Optional
import cv2
import numpy as np


def rotate_axis_to_horizontal(image: np.ndarray, axis_deg: float) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    h, w = image.shape[:2]
    theta = math.radians(float(axis_deg) % 180.0)
    nw = max(1, int(math.ceil(w * abs(math.cos(theta)) + h * abs(math.sin(theta)))))
    nh = max(1, int(math.ceil(h * abs(math.cos(theta)) + w * abs(math.sin(theta)))))
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(axis_deg) % 180.0, 1.0)
    matrix[0, 2] += (nw - 1) / 2.0 - center[0]
    matrix[1, 2] += (nh - 1) / 2.0 - center[1]
    return cv2.warpAffine(image, matrix, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _foreground_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, a = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = cv2.bitwise_not(a)
    h, w = gray.shape
    # Choose polarity whose largest component is closest to the crop centre
    # and occupies a plausible fraction of the image.
    def quality(mask):
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < max(8, int(h * w * 0.01)):
                continue
            cx, cy = centroids[i]
            dist = math.hypot((cx - w / 2) / max(w, 1), (cy - h / 2) / max(h, 1))
            frac = area / float(h * w)
            candidates.append((dist + abs(frac - 0.35), i, labels))
        if not candidates:
            return np.zeros_like(mask)
        _, idx, labels = min(candidates, key=lambda x: x[0])
        return np.uint8(labels == idx) * 255
    ma, mb = quality(a), quality(b)
    return ma if np.count_nonzero(ma) >= np.count_nonzero(mb) else mb


def extract_direction_features(image: np.ndarray, axis_deg: float,
                               end_fraction: float = 0.30,
                               centre_band_fraction: float = 0.72):
    """Return deterministic geometric/appearance measurements for both ends."""
    if image is None or image.size == 0 or axis_deg is None or not math.isfinite(float(axis_deg)):
        return None
    aligned = rotate_axis_to_horizontal(image, axis_deg)
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
    h, w = gray.shape
    if h < 5 or w < 12:
        return None
    y0 = max(0, int((h - h * centre_band_fraction) / 2))
    y1 = min(h, y0 + max(3, int(h * centre_band_fraction)))
    band = gray[y0:y1]
    mask = _foreground_mask(gray)[y0:y1]
    span = max(3, int(w * end_fraction))
    # Per-column foreground width, normalized by the central maximum width.
    widths = np.count_nonzero(mask > 0, axis=0).astype(float)
    middle = widths[int(w * .30):max(int(w * .70), int(w * .30) + 1)]
    scale = max(float(np.percentile(middle, 90)) if middle.size else 0.0, 1.0)
    left_taper = float(np.mean(widths[:span]) / scale)
    right_taper = float(np.mean(widths[-span:]) / scale)
    def appearance(part):
        edges = cv2.Canny(part, 50, 150)
        gx = cv2.Sobel(part, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(part, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(part)), float(np.std(part)), float(np.mean(edges > 0)), float(np.mean(cv2.magnitude(gx, gy)))
    left, right = appearance(band[:, :span]), appearance(band[:, -span:])
    return {
        "left_taper_ratio": left_taper, "right_taper_ratio": right_taper,
        "left_mean_intensity": left[0], "right_mean_intensity": right[0],
        "left_std_intensity": left[1], "right_std_intensity": right[1],
        "left_edge_density": left[2], "right_edge_density": right[2],
        "left_gradient_mean": left[3], "right_gradient_mean": right[3],
        "taper_difference_right_minus_left": right_taper - left_taper,
    }


def estimate_heading_sign(image: np.ndarray, axis_deg: float,
                          ambiguity_threshold: float = 0.08):
    """Choose the narrower terminal silhouette as presumed front.

    Returns (sign, confidence, features). sign=1 means positive axis/right end;
    sign=0 means negative axis/left end. Confidence is a normalized taper gap.
    """
    f = extract_direction_features(image, axis_deg)
    if not f:
        return None, 0.0, {}
    l, r = f["left_taper_ratio"], f["right_taper_ratio"]
    gap = abs(l - r) / max(l, r, 1e-6)
    if gap < ambiguity_threshold:
        return None, gap, f
    # The smaller terminal silhouette is treated as the front end.
    return (0 if l < r else 1), gap, f


def heading_sign_from_ground_truth(heading_deg: float, axis_deg: float) -> int:
    delta = math.radians(float(heading_deg) - float(axis_deg))
    return int(math.cos(delta) >= 0.0)


def heading_from_axis_and_sign(axis_deg: float, sign: int) -> float:
    if int(sign) not in (0, 1):
        raise ValueError("sign must be 0 or 1")
    return (float(axis_deg) % 180.0 + (0.0 if int(sign) else 180.0)) % 360.0


def circular_heading_error(pred_deg: float, true_deg: float) -> float:
    return abs((float(pred_deg) - float(true_deg) + 180.0) % 360.0 - 180.0)
