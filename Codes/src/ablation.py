"""Run reproducible preprocessing ablations on a metadata CSV.

PCA: 18 configurations
Hough: 18 configurations
Otsu/moments: 6 configurations
Total: 42 configurations.
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
        for blur_size in BLURS:
            for use_clahe in CLAHE:
                for low, high in THRESHOLDS:
                    configs.append({
                        "method": method,
                        "blur_size": blur_size,
                        "clahe": use_clahe,
                        "low_threshold": low,
                        "high_threshold": high,
                    })

    for blur_size in BLURS:
        for use_clahe in CLAHE:
            configs.append({
                "method": "otsu_moments",
                "blur_size": blur_size,
                "clahe": use_clahe,
                "low_threshold": None,
                "high_threshold": None,
            })

    return configs

def metrics(errors, total, method, config):
    valid = [
        value for value in errors
        if value is not None and np.isfinite(value)
    ]

    e = np.asarray(valid, dtype=float)

    return {
        "method": method,
        **config,
        "total_crops": int(total),
        "valid_predictions": int(len(e)),
        "coverage_percent": round(100 * len(e) / total, 2) if total else 0.0,
        "mae_deg": round(float(e.mean()), 3) if len(e) else None,
        "median_error_deg": round(float(np.median(e)), 3) if len(e) else None,
        "p90_error_deg": round(float(np.percentile(e, 90)), 3) if len(e) else None,
        "within_10_deg_percent": round(float(np.mean(e <= 10) * 100), 2)
        if len(e) else None,
    }

def estimate(image, config):
    method = config["method"]
    blur = config["blur_size"]
    clahe = config["clahe"]

    if method == "pca_edges":
        return estimate_axis_pca(
            image,
            config["low_threshold"],
            config["high_threshold"],
            blur,
            clahe,
        )

    if method == "hough_edges":
        return estimate_axis_edges(
            image,
            config["low_threshold"],
            config["high_threshold"],
            blur,
            clahe,
        )

    if method == "otsu_moments":
        return estimate_axis_moments(image, blur, clahe)

    raise ValueError(f"Unknown method: {method}")

def main():
    parser = argparse.ArgumentParser(
        description="Run PCA, Hough, and Otsu ablation experiments."
    )

    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", default="results/ablation")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional crop limit; 0 uses all rows.",
    )

    args = parser.parse_args()

    metadata_path = Path(args.metadata)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    meta = pd.read_csv(metadata_path)

    required_columns = {"crop_path", "crop_id", "gt_axis_deg"}
    missing = required_columns - set(meta.columns)

    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    if args.limit > 0:
        meta = meta.head(args.limit).copy()

    if meta.empty:
        raise ValueError("Metadata contains no crops.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    configurations = build_configurations()

    all_results = []
    detail_rows = []

    print(f"Configurations: {len(configurations)}")
    print(f"Crops: {len(meta)}")

    for index, config in enumerate(configurations, start=1):
        method = config["method"]
        errors = []

        print(
            f"\n[{index}/{len(configurations)}] "
            f"{method} | blur={config['blur_size']} "
            f"| CLAHE={config['clahe']} "
            f"| thresholds=({config['low_threshold']}, "
            f"{config['high_threshold']})"
        )

        for _, row in meta.iterrows():
            image = cv2.imread(str(row["crop_path"]))

            pred = None

            if image is not None:
                try:
                    pred = estimate(image, config)
                except (ValueError, cv2.error, np.linalg.LinAlgError):
                    pred = None

            truth = row["gt_axis_deg"]

            if pred is None or pd.isna(truth):
                error = None
            else:
                error = axis_error(float(pred), float(truth))

            errors.append(error)

            detail_rows.append({
                "crop_id": row["crop_id"],
                **config,
                "pred_axis_deg": pred,
                "error_deg": error,
            })

        result = metrics(errors, len(meta), method, {
            key: value
            for key, value in config.items()
            if key != "method"
        })

        all_results.append(result)

        print(
            f"MAE: {result['mae_deg']} | "
            f"Coverage: {result['coverage_percent']}%"
        )

    summary = pd.DataFrame(all_results)

    summary.to_csv(
        out_dir / "ablation_summary.csv",
        index=False,
    )

    pd.DataFrame(detail_rows).to_csv(
        out_dir / "ablation_per_crop.csv",
        index=False,
    )

    valid = summary.dropna(subset=["mae_deg"]).sort_values(
        ["mae_deg", "coverage_percent"],
        ascending=[True, False],
    )

    best = valid.head(10).to_dict("records")

    (out_dir / "best_configs.json").write_text(
        json.dumps(best, indent=2),
        encoding="utf-8",
    )

    print("\nAblation completed.")
    print(f"Configurations: {len(summary)}")
    print(f"Crops per configuration: {len(meta)}")
    print(f"Saved results to: {out_dir}")
    print("\nBest configurations:")
    print(valid.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
