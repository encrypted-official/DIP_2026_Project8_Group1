# Week 5 Ablation Results

Record results after running `src/ablation.py`. Do not fill this table with expected or invented results.

| Method | Blur size | CLAHE | Canny thresholds | Valid predictions | Coverage (%) | MAE (deg) | Median error (deg) | P90 error (deg) | Within 10° (%) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| PCA | | | | | | | | | |
| Hough | | | | | | | | | |
| Moments/Otsu | | | N/A | | | | | | |

## Interpretation
- Identify the best configuration by error and coverage together.
- Compare median and P90 to distinguish typical performance from difficult cases.
- Note whether improvements persist across small crops, shadows, occlusions, and vehicle classes.
