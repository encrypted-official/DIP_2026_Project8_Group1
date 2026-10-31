"""Run reproducible preprocessing ablations on a metadata CSV.

Grid: 3 blur sizes x 2 CLAHE settings x 3 Canny threshold pairs = 18 settings
for PCA and 18 for Hough, plus 3 blur sizes x 2 CLAHE settings = 6 for Otsu/moments.
Total: 42 method/configuration runs.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.estimate_pca import estimate_axis_pca
from src.estimate_edges import estimate_axis_edges
from src.estimate_moments import estimate_axis_moments
from src.angle_utils import axis_error

THRESHOLDS = [(30, 90), (50, 150), (80, 200)]
BLURS = [1, 3, 5]
CLAHE = [False, True]

def build_configurations():
    configs = []

    for method in ("pca_edges", "hough_edges"):
        for blur_size in [1, 3, 5]:
            for use_clahe in [False, True]:
                for low, high in [(30, 90), (50, 150), (80, 200)]:
                    configs.append({
                        "method": method,
                        "blur_size": blur_size,
                        "clahe": use_clahe,
                        "low_threshold": low,
                        "high_threshold": high,
                    })

    for blur_size in [1, 3, 5]:
        for use_clahe in [False, True]:
            configs.append({
                "method": "otsu_moments",
                "blur_size": blur_size,
                "clahe": use_clahe,
                "low_threshold": None,
                "high_threshold": None,
            })

    return configs

def metrics(errors, total, method, config):
    e = np.asarray([v for v in errors if v is not None and np.isfinite(v)], dtype=float)
    result = {"method": method, **config, "total_crops": int(total),
              "valid_predictions": int(len(e)),
              "coverage_percent": round(100 * len(e) / total, 2) if total else 0.0,
              "mae_deg": round(float(e.mean()), 3) if len(e) else None,
              "median_error_deg": round(float(np.median(e)), 3) if len(e) else None,
              "p90_error_deg": round(float(np.percentile(e, 90)), 3) if len(e) else None,
              "within_10_deg_percent": round(float(np.mean(e <= 10) * 100), 2) if len(e) else None}
    return result


def main():
    parser = argparse.ArgumentParser(description="Run PCA, Hough, and Otsu ablation grid.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", default="results/ablation")
    parser.add_argument("--limit", type=int, default=0, help="Optional crop limit; 0 uses all rows")
    args = parser.parse_args()
    meta = pd.read_csv(args.metadata)
    if args.limit > 0:
        meta = meta.head(args.limit).copy()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    detail_rows = []
    for blur in BLURS:
        for clahe in CLAHE:
            for method in ("pca_edges", "hough_edges"):
                for low, high in THRESHOLDS:
                    cfg = {"blur_size": blur, "clahe": clahe,
                           "low_threshold": low, "high_threshold": high}
                    errors = []
                    for _, row in meta.iterrows():
                        image = cv2.imread(str(row["crop_path"]))
                        if method == "pca_edges":
                            pred = estimate_axis_pca(image, low, high, blur, clahe)
                        else:
                            pred = estimate_axis_edges(image, low, high, blur, clahe)
                        truth = row.get("gt_axis_deg")
                        err = None if pred is None or pd.isna(truth) else axis_error(pred, float(truth))
                        errors.append(err)
                        detail_rows.append({"crop_id": row["crop_id"], "method": method,
                                            **cfg, "pred_axis_deg": pred, "error_deg": err})
                    all_results.append(metrics(errors, len(meta), method, cfg))
            # Otsu/moments has no Canny thresholds.
            cfg = {"blur_size": blur, "clahe": clahe,
                   "low_threshold": None, "high_threshold": None}
            errors = []
            for _, row in meta.iterrows():
                image = cv2.imread(str(row["crop_path"]))
                pred = estimate_axis_moments(image, blur, clahe)
                truth = row.get("gt_axis_deg")
                err = None if pred is None or pd.isna(truth) else axis_error(pred, float(truth))
                errors.append(err)
                detail_rows.append({"crop_id": row["crop_id"], "method": "otsu_moments",
                                    **cfg, "pred_axis_deg": pred, "error_deg": err})
            all_results.append(metrics(errors, len(meta), "otsu_moments", cfg))
    summary = pd.DataFrame(all_results)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(out_dir / "ablation_per_crop.csv", index=False)
    valid = summary.dropna(subset=["mae_deg"]).sort_values(["mae_deg", "coverage_percent"], ascending=[True, False])
    (out_dir / "best_configs.json").write_text(json.dumps(valid.head(10).to_dict("records"), indent=2), encoding="utf-8")
    print(f"Configurations: {len(summary)} (expected 42)")
    print(f"Crops per configuration: {len(meta)}")
    print(f"Saved results to {out_dir}")
    print("Best configurations by MAE (inspect coverage as well):")
    print(valid.head(10).to_string(index=False))

if __name__ == "__main__":
    main()

"""Run reproducible preprocessing ablations on a metadata CSV.

Grid: 3 blur sizes x 2 CLAHE settings x 3 Canny threshold pairs = 18 settings
for PCA and 18 for Hough, plus 3 blur sizes x 2 CLAHE settings = 6 for Otsu/moments.
Total: 42 method/configuration runs.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.estimate_pca import estimate_axis_pca
from src.estimate_edges import estimate_axis_edges
from src.estimate_moments import estimate_axis_moments
from src.angle_utils import axis_error

THRESHOLDS = [(30, 90), (50, 150), (80, 200)]
BLURS = [1, 3, 5]
CLAHE = [False, True]


def metrics(errors, total, method, config):
    e = np.asarray([v for v in errors if v is not None and np.isfinite(v)], dtype=float)
    result = {"method": method, **config, "total_crops": int(total),
              "valid_predictions": int(len(e)),
              "coverage_percent": round(100 * len(e) / total, 2) if total else 0.0,
              "mae_deg": round(float(e.mean()), 3) if len(e) else None,
              "median_error_deg": round(float(np.median(e)), 3) if len(e) else None,
              "p90_error_deg": round(float(np.percentile(e, 90)), 3) if len(e) else None,
              "within_10_deg_percent": round(float(np.mean(e <= 10) * 100), 2) if len(e) else None}
    return result


def main():
    parser = argparse.ArgumentParser(description="Run PCA, Hough, and Otsu ablation grid.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", default="results/ablation")
    parser.add_argument("--limit", type=int, default=0, help="Optional crop limit; 0 uses all rows")
    args = parser.parse_args()
    meta = pd.read_csv(args.metadata)
    if args.limit > 0:
        meta = meta.head(args.limit).copy()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    detail_rows = []
    for blur in BLURS:
        for clahe in CLAHE:
            for method in ("pca_edges", "hough_edges"):
                for low, high in THRESHOLDS:
                    cfg = {"blur_size": blur, "clahe": clahe,
                           "low_threshold": low, "high_threshold": high}
                    errors = []
                    for _, row in meta.iterrows():
                        image = cv2.imread(str(row["crop_path"]))
                        if method == "pca_edges":
                            pred = estimate_axis_pca(image, low, high, blur, clahe)
                        else:
                            pred = estimate_axis_edges(image, low, high, blur, clahe)
                        truth = row.get("gt_axis_deg")
                        err = None if pred is None or pd.isna(truth) else axis_error(pred, float(truth))
                        errors.append(err)
                        detail_rows.append({"crop_id": row["crop_id"], "method": method,
                                            **cfg, "pred_axis_deg": pred, "error_deg": err})
                    all_results.append(metrics(errors, len(meta), method, cfg))
            # Otsu/moments has no Canny thresholds.
            cfg = {"blur_size": blur, "clahe": clahe,
                   "low_threshold": None, "high_threshold": None}
            errors = []
            for _, row in meta.iterrows():
                image = cv2.imread(str(row["crop_path"]))
                pred = estimate_axis_moments(image, blur, clahe)
                truth = row.get("gt_axis_deg")
                err = None if pred is None or pd.isna(truth) else axis_error(pred, float(truth))
                errors.append(err)
                detail_rows.append({"crop_id": row["crop_id"], "method": "otsu_moments",
                                    **cfg, "pred_axis_deg": pred, "error_deg": err})
            all_results.append(metrics(errors, len(meta), "otsu_moments", cfg))
    summary = pd.DataFrame(all_results)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(out_dir / "ablation_per_crop.csv", index=False)
    valid = summary.dropna(subset=["mae_deg"]).sort_values(["mae_deg", "coverage_percent"], ascending=[True, False])
    (out_dir / "best_configs.json").write_text(json.dumps(valid.head(10).to_dict("records"), indent=2), encoding="utf-8")
    print(f"Configurations: {len(summary)} (expected 42)")
    print(f"Crops per configuration: {len(meta)}")
    print(f"Saved results to {out_dir}")
    print("Best configurations by MAE (inspect coverage as well):")
    print(valid.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
