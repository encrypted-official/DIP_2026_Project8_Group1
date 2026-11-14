"""Create unified per-crop, method, axis-style, vehicle-class and route reports.

Example:
python -m src.final_evaluation --metadata crops/week6_1200/metadata.csv \
  --pred-dir results/final_sample --out results/final_evaluation

Prediction CSVs must contain crop_id and pred_axis_deg. Missing methods are skipped.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.angle_utils import axis_error

METHOD_FILES = {
    "pca_edges": "pca.csv",
    "hough_edges": "hough.csv",
    "otsu_moments": "moments.csv",
    "pca_hough_hybrid": "hybrid.csv",
    "pca_roi": "pca_roi.csv",
}

def axis_style(angle, tolerance=15.0):
    if pd.isna(angle):
        return "unknown"
    a = float(angle) % 180.0
    d = min(a, abs(a - 90.0), 180.0 - a)
    return "axis_aligned" if d <= tolerance else "diagonal"

def _metric_rows(df, group_cols):
    rows = []
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        valid = part.dropna(subset=["axis_error_deg"])
        e = valid["axis_error_deg"].to_numpy(float)
        row = dict(zip(group_cols, keys))
        row.update({
            "total": int(len(part)), "valid": int(len(valid)),
            "coverage_percent": round(100 * len(valid) / len(part), 2) if len(part) else 0.0,
            "mae_deg": round(float(np.mean(e)), 3) if len(e) else np.nan,
            "median_error_deg": round(float(np.median(e)), 3) if len(e) else np.nan,
            "p90_error_deg": round(float(np.percentile(e, 90)), 3) if len(e) else np.nan,
            "within_5_deg_percent": round(float(np.mean(e <= 5) * 100), 2) if len(e) else np.nan,
            "within_10_deg_percent": round(float(np.mean(e <= 10) * 100), 2) if len(e) else np.nan,
            "within_15_deg_percent": round(float(np.mean(e <= 15) * 100), 2) if len(e) else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", default="results/final_evaluation")
    ap.add_argument("--axis-tolerance", type=float, default=15.0)
    a = ap.parse_args()
    meta = pd.read_csv(a.metadata)
    required = {"crop_id", "gt_axis_deg"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Metadata missing columns: {sorted(required - set(meta.columns))}")
    if meta.crop_id.duplicated().any():
        raise ValueError("Metadata crop_id values must be unique")
    meta = meta.copy()
    # Prediction-derived columns are never trusted from metadata; avoid merge suffix collisions.
    meta = meta.drop(columns=[c for c in ("pred_axis_deg", "route", "method", "pred_axis_style", "axis_error_deg") if c in meta.columns])
    meta["gt_axis_style"] = meta.gt_axis_deg.map(lambda x: axis_style(x, a.axis_tolerance))
    if "class_name" not in meta: meta["class_name"] = "unknown"
    pred_dir = Path(a.pred_dir)
    frames = []
    for method, filename in METHOD_FILES.items():
        path = pred_dir / filename
        if not path.exists():
            print(f"SKIP missing prediction file: {path}")
            continue
        pred = pd.read_csv(path)
        if not {"crop_id", "pred_axis_deg"}.issubset(pred.columns):
            raise ValueError(f"{path} needs crop_id and pred_axis_deg columns")
        if pred.crop_id.duplicated().any():
            raise ValueError(f"Duplicate crop_id in {path}")
        keep = [c for c in ("crop_id", "pred_axis_deg", "route") if c in pred.columns]
        part = meta.merge(pred[keep], on="crop_id", how="left", validate="one_to_one")
        part["method"] = method
        part["pred_axis_style"] = part.pred_axis_deg.map(lambda x: axis_style(x, a.axis_tolerance))
        part["axis_error_deg"] = [axis_error(p, t) if pd.notna(p) and pd.notna(t) else np.nan
                                  for p, t in zip(part.pred_axis_deg, part.gt_axis_deg)]
        frames.append(part)
    if not frames:
        raise FileNotFoundError(f"No recognised prediction CSVs found in {pred_dir}")
    all_rows = pd.concat(frames, ignore_index=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out / "all_methods_per_crop.csv", index=False)
    tables = {
        "summary_by_method.csv": ["method"],
        "summary_by_method_axis_style.csv": ["method", "gt_axis_style"],
        "summary_by_method_vehicle_class.csv": ["method", "class_name"],
        "summary_by_method_axis_vehicle.csv": ["method", "gt_axis_style", "class_name"],
        "summary_by_predicted_axis_style.csv": ["method", "pred_axis_style"],
    }
    for filename, cols in tables.items():
        _metric_rows(all_rows, cols).to_csv(out / filename, index=False)
    usage = all_rows.groupby("method").agg(crops=("crop_id", "count"),
        valid_predictions=("pred_axis_deg", "count")).reset_index()
    usage["invalid_predictions"] = usage.crops - usage.valid_predictions
    usage.to_csv(out / "method_usage.csv", index=False)
    route_rows = all_rows[all_rows.method == "pca_hough_hybrid"].copy()
    if not route_rows.empty and "route" in route_rows:
        _metric_rows(route_rows, ["route", "gt_axis_style"]).to_csv(out / "hybrid_route_performance.csv", index=False)
        route_rows.groupby("route", dropna=False).size().rename("count").reset_index().to_csv(
            out / "hybrid_route_usage.csv", index=False)
    # Oracle is a retrospective lower bound, never a deployable method-selection result.
    wide = all_rows.pivot(index="crop_id", columns="method", values="axis_error_deg")
    wide["oracle_best_of_available_methods_deg"] = wide.min(axis=1, skipna=True)
    wide.reset_index().to_csv(out / "oracle_method_selection_upper_bound.csv", index=False)
    summary = {"metadata_rows": int(len(meta)), "methods_evaluated": sorted(all_rows.method.unique().tolist()),
        "axis_alignment_tolerance_deg": float(a.axis_tolerance),
        "vehicle_class_column": "class_name" if "class_name" in meta else None,
        "front_back_supervision": "not evaluated: no verified front/rear labels are assumed"}
    (out / "evaluation_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote unified evaluation to {out}")

if __name__ == "__main__": main()
