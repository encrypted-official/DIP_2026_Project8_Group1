"""Build front/back direction features from intensity profiles along an estimated axis.
This creates a labelled feature CSV for exploratory analysis.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.angle_utils import heading_direction_label

def axis_profile_features(image, axis_deg):
    if image is None or axis_deg is None or pd.isna(axis_deg):
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    if min(h, w) < 5:
        return None

    theta = np.radians(float(axis_deg))
    ux, uy = np.cos(theta), np.sin(theta)
    vx, vy = -uy, ux
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    half_len = 0.42 * min(w, h)
    half_width = max(1.0, 0.08 * min(w, h))
    longitudinal = np.linspace(0.18, 0.42, 9)
    transverse = np.linspace(-half_width, half_width, 7)

    def sample(sign):
        values = []
        for t in longitudinal:
            for q in transverse:
                x = cx + sign * t * (2 * half_len) * ux + q * vx
                y = cy + sign * t * (2 * half_len) * uy + q * vy
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < w and 0 <= yi < h:
                    values.append(float(gray[yi, xi]))
        if not values:
            return None
        return np.asarray(values, dtype=np.float32)

    positive = sample(1)
    negative = sample(-1)
    if positive is None or negative is None:
        return None

    return {
        "positive_end_mean": float(positive.mean()),
        "negative_end_mean": float(negative.mean()),
        "positive_end_std": float(positive.std()),
        "negative_end_std": float(negative.std()),
        "end_mean_difference": float(positive.mean() - negative.mean()),
        "end_std_difference": float(positive.std() - negative.std()),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="results/week6/front_back_features.csv")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    pred = pd.read_csv(args.predictions)
    required_meta = {"crop_id", "crop_path", "heading_deg"}
    required_pred = {"crop_id", "pred_axis_deg"}
    if not required_meta.issubset(meta.columns):
        raise ValueError(f"Metadata must contain {sorted(required_meta)}")
    if not required_pred.issubset(pred.columns):
        raise ValueError(f"Predictions must contain {sorted(required_pred)}")

    joined = meta.merge(
        pred[["crop_id", "pred_axis_deg"]], on="crop_id",
        how="left", validate="one_to_one"
    )
    rows = []
    for _, row in joined.iterrows():
        image = cv2.imread(str(row["crop_path"]))
        features = axis_profile_features(image, row["pred_axis_deg"])
        if features is None or pd.isna(row["pred_axis_deg"]):
            continue
        rows.append({
            "crop_id": row["crop_id"],
            **features,
            "heading_sign": heading_direction_label(
                row["heading_deg"], row["pred_axis_deg"]
            ),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} front/back feature rows to {out}")
    print("heading_sign=1 means ground-truth heading points along the positive predicted-axis direction.")
