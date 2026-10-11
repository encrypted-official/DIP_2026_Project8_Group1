"""Save example crops with ground-truth and predicted axis lines."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def draw_axis(image, angle_deg, bgr, length):
    """Draw an axis through the crop centre. This line has no front/back arrow."""
    if angle_deg is None or pd.isna(angle_deg):
        return
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    rad = np.radians(float(angle_deg))
    dx = int(np.cos(rad) * length / 2)
    dy = int(np.sin(rad) * length / 2)
    cv2.line(image, (cx - dx, cy - dy), (cx + dx, cy + dy), bgr, 2)


def main():
    parser = argparse.ArgumentParser(description="Visualize ground truth and predicted axes.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="results/figures/examples")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    pred = pd.read_csv(args.predictions)
    joined = meta.merge(pred[["crop_id", "pred_axis_deg"]], on="crop_id", how="left")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for _, row in joined.head(args.limit).iterrows():
        image = cv2.imread(str(row["crop_path"]))
        if image is None:
            continue
        length = int(min(image.shape[:2]) * 0.75)
        draw_axis(image, row["gt_axis_deg"], (0, 220, 0), length)
        draw_axis(image, row["pred_axis_deg"], (0, 0, 255), length)
        title = f'{row["crop_id"]} | {row["class_name"]}'
        cv2.putText(image, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 1, cv2.LINE_AA)
        out_path = out_dir / f'{row["crop_id"]}.jpg'
        cv2.imwrite(str(out_path), image)
        saved += 1

    print(f"Saved {saved} example images to {out_dir}")
    print("Overlay colours: ground truth = green, prediction = red.")


if __name__ == "__main__":
    main()
