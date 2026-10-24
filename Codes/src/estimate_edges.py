"""Estimate an undirected axis by combining Hough line directions."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocess import edge_map


def estimate_axis_edges(image, low_threshold=50, high_threshold=150, blur_size=3, use_clahe=True):
    """Use Hough line segments; longer lines get more weight in the average."""
    edges = edge_map(image, low_threshold, high_threshold, blur_size, use_clahe)
    if edges is None:
        return None

    min_len = max(8, int(min(image.shape[:2]) * 0.18))
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=15, minLineLength=min_len, maxLineGap=5
    )
    if lines is None:
        return None

    angles = []
    weights = []
    for line in lines.reshape(-1, 4):
        x0, y0, x1, y1 = [float(v) for v in line]
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        angle = np.arctan2(dy, dx)
        angles.append(angle)
        weights.append(length)

    if not angles:
        return None

    # Double angles so directions 10° and 190° count as the same axis.
    angles = np.asarray(angles)
    weights = np.asarray(weights)
    sin_sum = np.sum(weights * np.sin(2 * angles))
    cos_sum = np.sum(weights * np.cos(2 * angles))
    if np.hypot(sin_sum, cos_sum) < 1e-8:
        return None

    axis_rad = 0.5 * np.arctan2(sin_sum, cos_sum)
    return float(np.degrees(axis_rad) % 180.0)


def main():
    parser = argparse.ArgumentParser(description="Run Hough-line orientation estimation on crops.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", default="results/edge_predictions.csv")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    preds = []
    for _, row in meta.iterrows():
        img = cv2.imread(str(row["crop_path"]))
        angle = estimate_axis_edges(img)
        preds.append({
            "crop_id": row["crop_id"],
            "crop_path": row["crop_path"],
            "pred_axis_deg": angle,
            "method": "hough_edges",
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_csv(out_path, index=False)
    valid = sum(pd.notna(row["pred_axis_deg"]) for row in preds)
    print(f"Saved {len(preds)} predictions ({valid} valid) to {out_path}")


if __name__ == "__main__":
    main()
