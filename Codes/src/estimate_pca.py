import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocess import edge_map

def estimate_axis_pca(image, low_threshold=50, high_threshold=150, blur_size=3, use_clahe=True):
    """Return an axis angle in 0–180 degrees, or None if there are too few edges."""
    edges = edge_map(image, low_threshold, high_threshold, blur_size, use_clahe)
    if edges is None:
        return None

    ys, xs = np.where(edges > 0)
    if len(xs) < 20:
        return None

    pts = np.column_stack((xs, ys)).astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)

    vals, vecs = np.linalg.eigh(cov)
    if not np.isfinite(vals).all() or vals[-1] <= 1e-9:
        return None

    # A very round edge cloud has no strong dominant direction.
    if vals[-1] / max(vals[0], 1e-9) < 1.05:
        return None

    vx, vy = vecs[:, -1]
    return float(np.degrees(np.arctan2(vy, vx)) % 180.0)

def main():
    parser = argparse.ArgumentParser(description="Run pca_edges orientation estimation on saved crops.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", default="results/pca_predictions.csv")
    parser.add_argument("--blur-size", type=int, default=3, choices=[1, 3, 5, 7])
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--low-threshold", type=int, default=50)
    parser.add_argument("--high-threshold", type=int, default=150)
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    preds = []
    for _, row in meta.iterrows():
        img = cv2.imread(str(row["crop_path"]))
        angle = estimate_axis_pca(
            img, args.low_threshold, args.high_threshold,
            args.blur_size, not args.no_clahe
        )
        preds.append({
            "crop_id": row["crop_id"],
            "crop_path": row["crop_path"],
            "pred_axis_deg": angle,
            "method": "pca_edges",
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_csv(out_path, index=False)
    valid = sum(pd.notna(row["pred_axis_deg"]) for row in preds)
    print(f"Saved {len(preds)} predictions ({valid} valid) to {out_path}")

if __name__ == "__main__":
    main()