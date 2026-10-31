"""Week 6 ablation: PCA, Hough, Otsu/moments, including shadow-check toggle.

PCA: 18 configurations; Hough: 18; moments: 6 baseline + 6 shadow-check.
Total: 48 configurations.
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
from src.angle_utils import axis_error, heading_to_axis180

THRESHOLDS = [(30, 90), (50, 150), (80, 200)]
BLURS = [1, 3, 5]
CLAHE = [False, True]

def build_configurations():
    configs = []
    for method in ("pca_edges", "hough_edges"):
        for blur in BLURS:
            for clahe in CLAHE:
                for low, high in THRESHOLDS:
                    configs.append({
                        "method": method, "blur_size": blur, "clahe": clahe,
                        "low_threshold": low, "high_threshold": high,
                        "shadow_check": False,
                    })

    for shadow_check in (False, True):
        for blur in BLURS:
            for clahe in CLAHE:
                configs.append({
                    "method": "otsu_moments", "blur_size": blur, "clahe": clahe,
                    "low_threshold": None, "high_threshold": None,
                    "shadow_check": shadow_check,
                })
    return configs

def run_estimator(image, config):
    method = config["method"]
    blur = config["blur_size"]
    clahe = config["clahe"]

    if method == "pca_edges":
        return estimate_axis_pca(
            image, config["low_threshold"], config["high_threshold"], blur, clahe
        )
    if method == "hough_edges":
        return estimate_axis_edges(
            image, config["low_threshold"], config["high_threshold"], blur, clahe
        )
    return estimate_axis_moments(image, blur, clahe, config["shadow_check"])


def calculate_metrics(errors, total, config):
    values = np.asarray(
        [x for x in errors if x is not None and np.isfinite(x)], dtype=float
    )
    return {
        **config,
        "total_crops": int(total),
        "valid_predictions": int(len(values)),
        "coverage_percent": round(100 * len(values) / total, 2) if total else 0.0,
        "mae_deg": round(float(values.mean()), 3) if len(values) else None,
        "median_error_deg": round(float(np.median(values)), 3) if len(values) else None,
        "p90_error_deg": round(float(np.percentile(values, 90)), 3) if len(values) else None,
        "within_10_deg_percent": round(float(np.mean(values <= 10) * 100), 2)
        if len(values) else None,
    }

def main():
    parser = argparse.ArgumentParser(description="Run the Week 6 48-configuration ablation.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", default="results/week6/ablation")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    meta = pd.read_csv(metadata_path)
    required = {"crop_id", "crop_path", "heading_deg"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")
    if args.limit > 0:
        meta = meta.head(args.limit).copy()
    if meta.empty:
        raise ValueError("No crops to process.")

    meta["gt_axis_deg_eval"] = meta["heading_deg"].map(heading_to_axis180)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    configs = build_configurations()
    summaries, details = [], []

    print(f"Configurations: {len(configs)}")
    print(f"Crops: {len(meta)}")

    for i, config in enumerate(configs, 1):
        errors = []
        print(
            f"[{i}/{len(configs)}] {config['method']} blur={config['blur_size']} "
            f"CLAHE={config['clahe']} thresholds="
            f"{config['low_threshold']},{config['high_threshold']} "
            f"shadow={config['shadow_check']}"
        )

        for _, row in meta.iterrows():
            image = cv2.imread(str(row["crop_path"]))
            prediction = None
            if image is not None:
                try:
                    prediction = run_estimator(image, config)
                except (ValueError, cv2.error, np.linalg.LinAlgError):
                    prediction = None

            truth = row["gt_axis_deg_eval"]
            error = None if prediction is None or pd.isna(truth) else axis_error(
                float(prediction), float(truth)
            )
            errors.append(error)
            details.append({
                "crop_id": row["crop_id"],
                "method": config["method"],
                **{k: v for k, v in config.items() if k != "method"},
                "pred_axis_deg": prediction,
                "gt_axis_deg_eval": truth,
                "error_deg": error,
            })

        summaries.append(calculate_metrics(errors, len(meta), config))

    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "ablation_summary.csv", index=False)
    pd.DataFrame(details).to_csv(out / "ablation_per_crop.csv", index=False)

    valid = summary.dropna(subset=["mae_deg"]).sort_values(
        ["mae_deg", "coverage_percent"], ascending=[True, False]
    )
    (out / "best_configs.json").write_text(
        json.dumps(valid.head(10).to_dict("records"), indent=2),
        encoding="utf-8",
    )

    print(f"Completed {len(summary)} configurations over {len(meta)} crops.")
    print(f"Saved results to {out}")
    print(valid.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
