"""Compare predicted axis angles with the ground-truth axis angles."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.angle_utils import axis_error


def main():
    parser = argparse.ArgumentParser(description="Evaluate one estimator's predictions.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True, help="Output prefix, e.g. results/pca")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    pred = pd.read_csv(args.predictions)
    joined = meta.merge(pred[["crop_id", "pred_axis_deg", "method"]], on="crop_id", how="left")

    errs = []
    for _, row in joined.iterrows():
        if pd.isna(row["pred_axis_deg"]) or pd.isna(row["gt_axis_deg"]):
            errs.append(np.nan)
        else:
            errs.append(axis_error(float(row["pred_axis_deg"]), float(row["gt_axis_deg"])))
    joined["axis_error_deg"] = errs

    valid = joined.dropna(subset=["axis_error_deg"]).copy()
    if valid.empty:
        metrics = {
            "method": str(pred["method"].iloc[0]) if not pred.empty else "unknown",
            "total_crops": int(len(joined)),
            "valid_predictions": 0,
            "coverage_percent": 0.0,
            "mae_deg": None,
            "median_error_deg": None,
            "p90_error_deg": None,
            "within_5_deg_percent": None,
            "within_10_deg_percent": None,
            "within_15_deg_percent": None,
        }
    else:
        e = valid["axis_error_deg"].to_numpy(dtype=float)
        metrics = {
            "method": str(valid["method"].iloc[0]),
            "total_crops": int(len(joined)),
            "valid_predictions": int(len(valid)),
            "coverage_percent": round(100 * len(valid) / max(1, len(joined)), 2),
            "mae_deg": round(float(np.mean(e)), 3),
            "median_error_deg": round(float(np.median(e)), 3),
            "p90_error_deg": round(float(np.percentile(e, 90)), 3),
            "within_5_deg_percent": round(float(np.mean(e <= 5) * 100), 2),
            "within_10_deg_percent": round(float(np.mean(e <= 10) * 100), 2),
            "within_15_deg_percent": round(float(np.mean(e <= 15) * 100), 2),
        }

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out_prefix.with_name(out_prefix.name + "_per_crop.csv"), index=False)
    out_prefix.with_name(out_prefix.name + "_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # A small class table helps reveal which vehicle types are difficult.
    if not valid.empty:
        by_class = valid.groupby("class_name")["axis_error_deg"].agg(
            samples="count", mae_deg="mean", median_error_deg="median"
        ).reset_index()
        by_class[["mae_deg", "median_error_deg"]] = by_class[
            ["mae_deg", "median_error_deg"]
        ].round(3)
        by_class.to_csv(out_prefix.with_name(out_prefix.name + "_by_class.csv"), index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Detailed results saved with prefix: {out_prefix}")


if __name__ == "__main__":
    main()
