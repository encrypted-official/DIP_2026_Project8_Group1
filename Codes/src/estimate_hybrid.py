"""Hybrid orientation estimator: PCA near image axes, Hough for diagonal crops."""

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.angle_utils import axis_error
from src.estimate_pca import estimate_axis_pca
from src.estimate_edges import estimate_axis_edges

def distance_to_image_axis(angle):
    a = float(angle) % 180.0
    return min(a, abs(a-90.0), 180.0-a)

def estimate_axis_hybrid(image, switch_deg=15.0, pca_angle=None, hough_angle=None,
                         pca_params=None, hough_params=None):
    """Choose PCA when near horizontal/vertical, otherwise Hough.

    Angles are undirected axes in [0,180). The switch decision uses the PCA
    estimate only; it does not use ground truth.
    """
    pca_params = pca_params or {}
    hough_params = hough_params or {}
    if pca_angle is None:
        pca_angle = estimate_axis_pca(image, **pca_params)
    if pca_angle is None or not np.isfinite(pca_angle):
        if hough_angle is None:
            hough_angle = estimate_axis_edges(image, **hough_params)
        return hough_angle, "hough_fallback"
    if distance_to_image_axis(pca_angle) <= float(switch_deg):
        return float(pca_angle) % 180.0, "pca"
    if hough_angle is None:
        hough_angle = estimate_axis_edges(image, **hough_params)
    if hough_angle is None or not np.isfinite(hough_angle):
        return float(pca_angle) % 180.0, "pca_fallback"
    return float(hough_angle) % 180.0, "hough_diagonal"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", default="results/hybrid_predictions.csv")
    ap.add_argument("--switch-deg", type=float, default=15.0)
    ap.add_argument("--pca-blur", type=int, default=5)
    ap.add_argument("--pca-no-clahe", action="store_true")
    ap.add_argument("--pca-low", type=int, default=30)
    ap.add_argument("--pca-high", type=int, default=90)
    ap.add_argument("--hough-blur", type=int, default=1)
    ap.add_argument("--hough-low", type=int, default=30)
    ap.add_argument("--hough-high", type=int, default=90)
    ap.add_argument("--hough-no-clahe", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    meta = pd.read_csv(args.metadata)
    if args.limit > 0: meta = meta.head(args.limit)
    pca_cfg = dict(low_threshold=args.pca_low, high_threshold=args.pca_high,
                   blur_size=args.pca_blur, use_clahe=not args.pca_no_clahe)
    hough_cfg = dict(low_threshold=args.hough_low, high_threshold=args.hough_high,
                     blur_size=args.hough_blur, use_clahe=not args.hough_no_clahe)
    rows = []
    for _, r in meta.iterrows():
        im = cv2.imread(str(r["crop_path"]))
        p = estimate_axis_pca(im, **pca_cfg) if im is not None else None
        h = estimate_axis_edges(im, **hough_cfg) if im is not None else None
        pred, route = estimate_axis_hybrid(im, args.switch_deg, p, h) if im is not None else (None, "invalid_image")
        rows.append({"crop_id":r["crop_id"],"crop_path":r["crop_path"],
                     "pca_axis_deg":p,"hough_axis_deg":h,"pred_axis_deg":pred,
                     "route":route,"method":"pca_hough_hybrid"})
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(f"Saved {len(rows)} rows to {out}")

if __name__ == "__main__": main()