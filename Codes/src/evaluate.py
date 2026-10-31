"""Evaluate predicted undirected axes against heading_deg modulo 180."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.angle_utils import axis_error, heading_to_axis180

def main():
    parser = argparse.ArgumentParser(description="Evaluate one estimator's predictions.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True, help="Output prefix, e.g. results/pca")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    pred = pd.read_csv(args.predictions)

    required_meta = {"crop_id", "heading_deg"}
    required_pred = {"crop_id", "pred_axis_deg", "method"}
    if not required_meta.issubset(meta.columns):
        raise ValueError(f"Metadata must contain {sorted(required_meta)}")
    if not required_pred.issubset(pred.columns):
        raise ValueError(f"Predictions must contain {sorted(required_pred)}")

    meta = meta.copy()
    meta["gt_axis_deg_eval"] = meta["heading_deg"].map(heading_to_axis180)
    joined = meta.merge(
        pred[["crop_id", "pred_axis_deg", "method"]],
        on="crop_id", how="left", validate="one_to_one"
    )

    errors = []
    for _, row in joined.iterrows():
        if pd.isna(row["pred_axis_deg"]) or pd.isna(row["gt_axis_deg_eval"]):
            errors.append(np.nan)
        else:
            errors.append(axis_error(float(row["pred_axis_deg"]), float(row["gt_axis_deg_eval"])))
    joined["axis_error_deg"] = errors

    valid = joined.dropna(subset=["axis_error_deg"]).copy()
    values = valid["axis_error_deg"].to_numpy(dtype=float)
    method_name = str(pred["method"].iloc[0]) if not pred.empty else "unknown"

    metrics = {
        "method": method_name,
        "total_crops": int(len(joined)),
        "valid_predictions": int(len(valid)),
        "coverage_percent": round(100 * len(valid) / max(1, len(joined)), 2),
        "mae_deg": round(float(np.mean(values)), 3) if len(values) else None,
        "median_error_deg": round(float(np.median(values)), 3) if len(values) else None,
        "p90_error_deg": round(float(np.percentile(values, 90)), 3) if len(values) else None,
        "within_5_deg_percent": round(float(np.mean(values <= 5) * 100), 2) if len(values) else None,
        "within_10_deg_percent": round(float(np.mean(values <= 10) * 100), 2) if len(values) else None,
        "within_15_deg_percent": round(float(np.mean(values <= 15) * 100), 2) if len(values) else None,
    }

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out_prefix.with_name(out_prefix.name + "_per_crop.csv"), index=False)
    out_prefix.with_name(out_prefix.name + "_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    if not valid.empty and "class_name" in valid.columns:
        by_class = valid.groupby("class_name")["axis_error_deg"].agg(
            samples="count", mae_deg="mean", median_error_deg="median"
        ).reset_index()
        by_class[["mae_deg", "median_error_deg"]] = by_class[
            ["mae_deg", "median_error_deg"]
        ].round(3)
        by_class.to_csv(out_prefix.with_name(out_prefix.name + "_by_class.csv"), index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Detailed results saved with prefix: {out_prefix}")