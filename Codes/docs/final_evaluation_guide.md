# Final Report Evaluation and Comparison Guide

This workflow produces report-ready comparisons from DRASHTI-HaOBB annotations. It does not invent results: the Markdown report explicitly marks missing evaluations as not run.

## Evaluation checklist

1. Regenerate crop metadata with the corrected axis target `heading_deg % 180`.
2. Audit the selected dataset split and extracted crops.
3. Split by `source_image`, not by crop, so crops from one source frame cannot leak across train/test.
4. Run the same five axis methods on the same train/test metadata.
5. Evaluate the undirected axis using 180-degree periodic error.
6. Evaluate directed heading using 360-degree circular error and report direction-resolved coverage separately.
7. Run preprocessing ablation selection on training metadata only; evaluate selected configurations on held-out test metadata.
8. Inspect representative overlays, including correct predictions, large errors, and unresolved directions.
9. Generate the Markdown summary only after the required result CSVs exist.

## Dataset and metadata audit

```bat
python -m src.audit_dataset --root data/DRASHTI-HaOBB --split val --metadata crops/full_val/metadata.csv --out results/dataset_audit
```

The audit records paired images, missing image matches, unreadable images, annotation counts, image dimensions, and crop-size bands. Also inspect `metadata.csv` for unique crop IDs, valid paths, class labels, heading angles, and difficulty filtering.

## Source-image split

```bat
python -m src.make_experiment_splits --metadata crops/full_val/metadata.csv --out data/full_splits --test-size 0.20 --seed 42
```

Verify `split_manifest.json`; train/test source-image overlap must be zero.

## Axis estimation

Run all five estimators on both splits. Example for train; repeat with `test_metadata.csv` and output directory `results/axis_test`.

```bat
python -m src.estimate_pca --metadata data/full_splits/train_metadata.csv --out results/axis_train/pca.csv
python -m src.estimate_edges --metadata data/full_splits/train_metadata.csv --out results/axis_train/hough.csv
python -m src.estimate_moments --metadata data/full_splits/train_metadata.csv --out results/axis_train/moments.csv
python -m src.estimate_hybrid --metadata data/full_splits/train_metadata.csv --out results/axis_train/hybrid.csv
python -m src.estimate_pca_roi --metadata data/full_splits/train_metadata.csv --out results/axis_train/pca_roi.csv
```

Evaluate axis results:

```bat
python -m src.final_evaluation --metadata data/full_splits/test_metadata.csv --pred-dir results/axis_test --out results/axis_evaluation
```

## Directed heading

```bat
python -m src.run_heading_pipeline --train-metadata data/full_splits/train_metadata.csv --test-metadata data/full_splits/test_metadata.csv --train-pred-dir results/axis_train --test-pred-dir results/axis_test --out results/directed_heading
```

The direction stage uses Otsu foreground segmentation and a fixed terminal silhouette-width/taper rule. It does not train or fit a model. The narrower terminal is presumed to be the front; this is a heuristic, not guaranteed front/rear recognition. Report direction-resolved coverage, directed MAE, median, P90, within-5/10/15-degree percentages, and sign accuracy. Unresolved cases must remain in the coverage denominator.

## Ablation

Select configurations on the training subset only:

```bat
python -m src.ablation --metadata data/full_splits/ablation_train_sample.csv --out results/ablation_train
```

Evaluate selected configurations on held-out test metadata:

```bat
python -m src.evaluate_ablation --metadata data/full_splits/test_metadata.csv --configs results/ablation_train/best_configs.json --out results/ablation_test --top-k 5
```

## Report-ready summary

```bat
python -m src.generate_final_report --axis-dir results/axis_evaluation --direction-dir results/directed_heading --ablation-dir results/ablation_test --splits data/full_splits/split_manifest.json --out results/final_report
```

Main output: `results/final_report/FINAL_REPORT_SUMMARY.md`. Validation status: `report_validation_checks.csv`.

## Comparisons to include in the report

- **Axis methods:** compare PCA edges, Hough edges, Otsu/moments, PCA-Hough hybrid, and elliptical-ROI PCA on identical held-out crops.
- **Axis-style breakdown:** axis-aligned versus diagonal, using the stated 15-degree tolerance to the nearest horizontal or vertical orientation.
- **Vehicle-class breakdown:** include sample counts; avoid strong claims for rare classes.
- **Hybrid routes:** report which route was selected and route-conditioned errors.
- **Directed heading:** compare direction coverage and 360-degree heading error. Do not conflate direction-sign accuracy with total heading-angle accuracy.
- **Ablation:** compare held-out performance of configurations selected on training data only.
- **Qualitative checks:** inspect overlays and failure cases; record annotation convention and any unresolved ambiguity.

## Reporting safeguards

- The earlier 100-crop benchmark is preliminary and not a substitute for this held-out evaluation.
- Do not state full-dataset metrics until all commands finish successfully.
- Keep axis and directed-heading metrics separate because they use different angular periodicities.
- A low directed MAE with low direction coverage can be misleading; always report both.
- The narrower-end heuristic can fail on symmetrical body shapes, unusual vehicle types, occlusion, shadows, crop truncation, and segmentation errors.
- Confirm that the dataset's heading-angle convention matches image coordinates (x right, y down) before interpreting signs as front/rear.
