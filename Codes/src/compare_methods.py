"""Compare estimators on their own valid crops and on a shared crop intersection."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.angle_utils import axis_error, heading_to_axis180

def summarize(name, errors, total):
    values = np.asarray([e for e in errors if e is not None and np.isfinite(e)], dtype=float)
    return {
        "comparison": name,
        "total_crops": int(total),
        "valid_predictions": int(len(values)),
        "coverage_percent": round(100 * len(values) / total, 2) if total else 0.0,
        "mae_deg": round(float(values.mean()), 3) if len(values) else None,
        "median_error_deg": round(float(np.median(values)), 3) if len(values) else None,
        "p90_error_deg": round(float(np.percentile(values, 90)), 3) if len(values) else None,
        "within_10_deg_percent": round(float(np.mean(values <= 10) * 100), 2) if len(values) else None,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--hough", required=True)
    parser.add_argument("--moments", required=True)
    parser.add_argument("--out", default="results/week6/method_comparison")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    if not {"crop_id", "heading_deg"}.issubset(meta.columns):
        raise ValueError("Metadata must contain crop_id and heading_deg.")
    meta = meta[["crop_id", "heading_deg"]].copy()
    meta["gt_axis_deg"] = meta["heading_deg"].map(heading_to_axis180)

    paths = {"PCA": args.pca, "Hough": args.hough, "Moments": args.moments}
    joined = meta.copy()
    pred_cols = []

    for name, path in paths.items():
        pred = pd.read_csv(path)
        if not {"crop_id", "pred_axis_deg"}.issubset(pred.columns):
            raise ValueError(f"{path} must contain crop_id and pred_axis_deg.")
        col = f"{name.lower()}_pred"
        joined = joined.merge(
            pred[["crop_id", "pred_axis_deg"]].rename(columns={"pred_axis_deg": col}),
            on="crop_id", how="left", validate="one_to_one"
        )
        pred_cols.append(col)

    errors_by_method = {}
    valid_by_method = {}
    for name in paths:
        col = f"{name.lower()}_pred"
        errors = []
        valid_mask = joined[col].notna() & joined["gt_axis_deg"].notna()
        for _, row in joined.loc[valid_mask].iterrows():
            errors.append(axis_error(float(row[col]), float(row["gt_axis_deg"])))
        errors_by_method[name] = errors
        valid_by_method[name] = valid_mask

    rows = [
        summarize(f"{name}_own_valid", errors_by_method[name], len(meta))
        for name in paths
    ]

    common_mask = joined[pred_cols].notna().all(axis=1) & joined["gt_axis_deg"].notna()
    for name in paths:
        col = f"{name.lower()}_pred"
        common_errors = [
            axis_error(float(row[col]), float(row["gt_axis_deg"]))
            for _, row in joined.loc[common_mask].iterrows()
        ]
        rows.append(summarize(f"{name}_common_crops", common_errors, int(common_mask.sum())))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out.with_name(out.name + "_summary.csv"), index=False)
    joined.to_csv(out.with_name(out.name + "_per_crop.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Common crops: {int(common_mask.sum())}/{len(meta)}")
    print(f"Saved comparison to {out.parent}")

if __name__ == "__main__":
    main()
