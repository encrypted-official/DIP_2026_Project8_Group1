"""Generate a report-ready Markdown summary from completed held-out evaluations.

Run after final_evaluation, run_heading_pipeline, and evaluate_ablation.
No metrics are fabricated: missing inputs are reported as unavailable.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def md_table(frame, columns=None, digits=3):
    if frame is None or frame.empty:
        return "_No results available._"
    f = frame.copy()
    if columns:
        f = f[[c for c in columns if c in f.columns]]
    for c in f.select_dtypes(include="number").columns:
        f[c] = f[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    headers = list(f.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"]
    for row in f.fillna("").astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--axis-dir", default="results/axis_evaluation")
    p.add_argument("--direction-dir", default="results/directed_heading")
    p.add_argument("--ablation-dir", default="results/ablation_test")
    p.add_argument("--splits", default="data/full_splits/split_manifest.json")
    p.add_argument("--out", default="results/final_report")
    a = p.parse_args()
    axis_dir, direction_dir, ablation_dir = map(Path, (a.axis_dir, a.direction_dir, a.ablation_dir))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    axis = read_csv(axis_dir / "summary_by_method.csv")
    axis_style = read_csv(axis_dir / "summary_by_method_axis_style.csv")
    classes = read_csv(axis_dir / "summary_by_method_vehicle_class.csv")
    routes = read_csv(axis_dir / "hybrid_route_performance.csv")
    direction = read_csv(direction_dir / "summary_by_method.csv")
    direction_classes = read_csv(direction_dir / "summary_by_method_vehicle_class.csv")
    direction_style = read_csv(direction_dir / "summary_by_method_axis_style.csv")
    ablation = read_csv(ablation_dir / "ablation_test_summary.csv")
    split = {}
    split_path = Path(a.splits)
    if split_path.exists():
        try: split = json.loads(split_path.read_text(encoding="utf-8"))
        except Exception: split = {}

    checks = []
    for name, frame, expected in [
        ("Axis evaluation", axis, {"method", "mae_deg", "coverage_percent"}),
        ("Directed-heading evaluation", direction, {"method", "heading_mae_deg", "heading_coverage_percent"}),
        ("Held-out ablation", ablation, {"method", "mae_deg", "coverage_percent"}),
    ]:
        missing = expected - set(frame.columns)
        checks.append({"check": name, "status": "PASS" if not frame.empty and not missing else "NOT RUN / INCOMPLETE",
                       "detail": "results found" if not frame.empty and not missing else ("missing columns: " + ", ".join(sorted(missing)) if not frame.empty else "result file not found")})
    if split:
        checks.append({"check":"Source-image split manifest", "status":"PASS" if split.get("split_unit")=="source_image" else "REVIEW",
                       "detail":f"train crops={split.get('rows_train','?')}; test crops={split.get('rows_test','?')}; train images={split.get('source_images_train','?')}; test images={split.get('source_images_test','?')}"})
    else:
        checks.append({"check":"Source-image split manifest", "status":"NOT RUN", "detail":"split manifest not found"})
    check_df = pd.DataFrame(checks); check_df.to_csv(out/"report_validation_checks.csv", index=False)

    best_axis = axis.sort_values(["mae_deg", "coverage_percent"], ascending=[True, False]).head(1) if not axis.empty else axis
    best_heading = direction.sort_values(["heading_mae_deg", "heading_coverage_percent"], ascending=[True, False]).head(1) if not direction.empty else direction
    best_ablation = ablation.sort_values(["mae_deg", "coverage_percent"], ascending=[True, False]).head(10) if not ablation.empty else ablation

    sections = [
        "# Final Experimental Results — DRASHTI-HaOBB",
        "",
        "> This report is generated from the result CSVs present when the script is run. Missing result sets are marked as not run; no values are inferred or filled in.",
        "",
        "## 1. Dataset and split summary",
        "",
        md_table(pd.DataFrame([split]) if split else pd.DataFrame(), ["rows_total","rows_train","rows_test","source_images_train","source_images_test","test_fraction_actual","split_unit","seed"]),
        "",
        "## 2. Evaluation integrity checks",
        "",
        md_table(check_df),
        "",
        "## 3. Undirected heading-axis comparison (0–180°)",
        "",
        "All methods are compared on the same held-out crop set. Lower MAE/median/P90 is better; higher coverage and within-threshold percentages are better.",
        "",
        md_table(axis, ["method","total","valid","coverage_percent","mae_deg","median_error_deg","p90_error_deg","within_5_deg_percent","within_10_deg_percent","within_15_deg_percent"]),
        "",
        "### Best axis method by held-out MAE",
        "",
        md_table(best_axis),
        "",
        "### Axis-style breakdown",
        "",
        md_table(axis_style),
        "",
        "### Vehicle-category breakdown",
        "",
        md_table(classes),
        "",
        "### Hybrid route analysis",
        "",
        md_table(routes),
        "",
        "## 4. Directed heading comparison (0–360°)",
        "",
        "Direction is estimated by the fixed conventional rule: Otsu foreground segmentation followed by terminal silhouette-width/taper comparison. The narrower end is presumed to be the front. Unresolved cases remain in the coverage denominator and are excluded from error statistics.",
        "",
        md_table(direction, ["method","test_crops","direction_resolved","heading_coverage_percent","heading_mae_deg","heading_median_error_deg","heading_p90_error_deg","within_5_deg_percent","within_10_deg_percent","within_15_deg_percent","ambiguous_or_invalid"]),
        "",
        "### Best directed-heading method by MAE",
        "",
        md_table(best_heading),
        "",
        "### Directed heading by axis style",
        "",
        md_table(direction_style),
        "",
        "### Directed heading by vehicle category",
        "",
        md_table(direction_classes),
        "",
        "## 5. Preprocessing ablation — held-out evaluation",
        "",
        "Configurations must be selected using training data only. This table contains the held-out evaluation of the selected configurations, not the training leaderboard.",
        "",
        md_table(best_ablation),
        "",
        "## 6. Interpretation and limitations",
        "",
        "- Axis error uses 180° periodicity; directed-heading error uses 360° circular distance. These metrics are not interchangeable.",
        "- The five axis estimators are conventional image-processing methods: PCA edges, Hough edges, Otsu/moments, PCA-Hough hybrid, and elliptical-ROI PCA.",
        "- The directed-end heuristic is deterministic and uses no training, fitted classifier, or learned parameters. Its narrower-end-is-front assumption can fail on symmetric, occluded, low-resolution, or poorly segmented vehicles.",
        "- Report both direction accuracy and direction-resolved coverage. A low MAE on a small resolved subset is not sufficient evidence of a reliable direction estimator.",
        "- Inspect the saved direction overlays against the DRASHTI-HaOBB annotations to verify angle convention, rotation, and front/rear interpretation.",
        "- State the exact numbers of source images and vehicle crops. Do not describe preliminary sample results as full-dataset results.",
        "",
        "## 7. Reproducibility",
        "",
        f"Generated from: `{axis_dir}`, `{direction_dir}`, `{ablation_dir}`, split manifest `{split_path}`.",
        ""
    ]
    report = "\n".join(sections)
    (out/"FINAL_REPORT_SUMMARY.md").write_text(report, encoding="utf-8")
    print(f"Report summary: {out/'FINAL_REPORT_SUMMARY.md'}")
    print(f"Validation checks: {out/'report_validation_checks.csv'}")
    print(check_df.to_string(index=False))

if __name__ == "__main__":
    main()
