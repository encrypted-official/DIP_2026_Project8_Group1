"""Method C: Otsu foreground mask followed by image-moment axis estimation."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocess import clean_gray

def suppress_one_sided_shadow(gray, component):
    """Conservatively remove a dark, spatially one-sided appendage from a blob.

    This is an experimental heuristic, disabled by default. It only removes the
    darkest pixels when they form a minority and their centroid is displaced
    from the component centre. It is not a general shadow detector.
    """
    component = component.astype(bool)
    ys, xs = np.where(component)
    if len(xs) < 20:
        return component.astype(np.uint8) * 255

    values = gray[component].astype(np.float32)
    cutoff = float(np.percentile(values, 25))
    dark = component & (gray <= cutoff)
    dys, dxs = np.where(dark)

    dark_fraction = len(dxs) / max(1, len(xs))
    if len(dxs) < 5 or not (0.08 <= dark_fraction <= 0.40):
        return component.astype(np.uint8) * 255

    cx, cy = float(xs.mean()), float(ys.mean())
    dcx, dcy = float(dxs.mean()), float(dys.mean())
    h, w = gray.shape[:2]
    displacement = np.hypot((dcx - cx) / max(w, 1), (dcy - cy) / max(h, 1))

    if displacement < 0.12:
        return component.astype(np.uint8) * 255

    # Remove only the dark minority on the displaced side; retain a minimum blob.
    candidate = component & ~dark
    if int(candidate.sum()) < max(9, int(0.03 * gray.size)):
        return component.astype(np.uint8) * 255
    return candidate.astype(np.uint8) * 255

def estimate_axis_moments(image, blur_size=3, use_clahe=True, shadow_check=False):
    """Return undirected axis [0,180), or None when the mask is unreliable."""
    if image is None:
        return None

    gray = clean_gray(image, blur_size=blur_size, use_clahe=use_clahe)
    if gray is None:
        return None

    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return None

    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = cv2.bitwise_not(bright)
    centre = (w / 2.0, h / 2.0)
    candidates = []

    for raw in (bright, dark):
        mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < max(9, int(h * w * 0.03)) or area > int(h * w * 0.90):
                continue
            cx, cy = centroids[idx]
            dist = ((cx - centre[0]) / max(w, 1)) ** 2 + ((cy - centre[1]) / max(h, 1)) ** 2
            candidates.append((dist, -area, labels == idx))

    if not candidates:
        return None

    _, _, component = min(candidates, key=lambda item: (item[0], item[1]))

    if shadow_check:
        component = suppress_one_sided_shadow(gray, component)

    ys, xs = np.where(component)
    if len(xs) < 9:
        return None

    points = np.column_stack((xs, ys)).astype(np.float64)
    cov = np.cov(points, rowvar=False)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return None

    vals, vecs = np.linalg.eigh(cov)
    if vals[-1] <= 1e-9 or vals[-1] / max(vals[0], 1e-9) < 1.15:
        return None

    vx, vy = vecs[:, -1]
    return float(np.degrees(np.arctan2(vy, vx)) % 180.0)

def main():
    parser = argparse.ArgumentParser(description="Run Otsu + image-moment orientation estimation.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", default="results/moments_predictions.csv")
    parser.add_argument("--blur-size", type=int, default=3, choices=[1, 3, 5, 7])
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--shadow-check", action="store_true",
                        help="Enable experimental one-sided dark-blob suppression.")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    rows = []
    for _, row in meta.iterrows():
        image = cv2.imread(str(row["crop_path"]))
        angle = estimate_axis_moments(
            image, args.blur_size, not args.no_clahe, args.shadow_check
        )
        rows.append({
            "crop_id": row["crop_id"],
            "crop_path": row["crop_path"],
            "pred_axis_deg": angle,
            "method": "otsu_moments_shadow" if args.shadow_check else "otsu_moments",
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    valid = sum(pd.notna(row["pred_axis_deg"]) for row in rows)
    print(f"Saved {len(rows)} predictions ({valid} valid) to {out}")

if __name__ == "__main__":
    main()
