"""Draw predicted heading direction and undirected axis on vehicle crops."""
import argparse
from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd
from src.angle_utils import directed_angle_error


def _point(cx, cy, length, angle_deg):
    theta = math.radians(float(angle_deg))
    return (int(round(cx + length * math.cos(theta))),
            int(round(cy + length * math.sin(theta))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--predictions", required=True,
                    help="One method's *_directed_predictions.csv")
    ap.add_argument("--out", default="results/heading_visualizations")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata)
    pred = pd.read_csv(args.predictions)
    required = {"crop_id", "pred_axis_deg", "pred_heading_deg"}
    if not required.issubset(pred.columns):
        raise ValueError(f"Predictions missing columns: {sorted(required-set(pred.columns))}")
    joined = meta.merge(pred, on="crop_id", how="inner", suffixes=("", "_pred"))
    if args.limit > 0:
        joined = joined.head(args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    saved = 0
    for row in joined.itertuples(index=False):
        image = cv2.imread(str(row.crop_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        length = max(8, int(0.38 * min(w, h)))
        axis = getattr(row, "pred_axis_deg")
        heading = getattr(row, "pred_heading_deg")
        if pd.isna(axis) or pd.isna(heading):
            continue
        # Undirected axis is a double-ended line.
        p1 = _point(cx, cy, length, float(axis))
        p2 = _point(cx, cy, length, float(axis) + 180.0)
        cv2.line(image, p1, p2, (255, 200, 0), 2, cv2.LINE_AA)
        # Directed heading is a one-ended arrow.
        end = _point(cx, cy, length, float(heading))
        cv2.arrowedLine(image, (cx, cy), end, (0, 0, 255), 2,
                        cv2.LINE_AA, tipLength=0.25)
        gt = getattr(row, "heading_deg", np.nan)
        if pd.notna(gt):
            gt_end = _point(cx, cy, length, float(gt))
            cv2.arrowedLine(image, (cx, cy), gt_end, (0, 220, 0), 2,
                            cv2.LINE_AA, tipLength=0.25)
            err = directed_angle_error(float(heading), float(gt))
            label = f"Pred {float(heading):.1f} deg | GT {float(gt):.1f} deg | Err {err:.1f} deg"
        else:
            label = f"Pred heading {float(heading):.1f} deg"
        cv2.putText(image, label, (3, max(12, min(h - 4, 14))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, label, (3, max(12, min(h - 4, 14))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
        name = f"{row.crop_id}.jpg"
        cv2.imwrite(str(out / name), image)
        saved += 1
    print(f"Saved {saved} annotated crops to {out}")


if __name__ == "__main__":
    main()
