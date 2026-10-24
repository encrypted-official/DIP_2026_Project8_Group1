# Week 5 Analysis Notes

## Dataset and evaluation checks
- Record the number of images, annotations, extracted crops, and valid predictions.
- Report coverage separately from angle error.
- Use undirected-axis error modulo 180 degrees; do not use plain absolute difference for wrap-around cases.
- Confirm the dataset heading convention (clockwise/counter-clockwise and image y-axis direction) against visualization overlays.

## Error analysis
For each method, inspect representative examples of:
- small vehicle crops;
- strong shadows or low contrast;
- partial occlusion;
- cluttered backgrounds;
- vehicles whose body axis is visually ambiguous.

## Reporting template
| Method | Valid / Total | Coverage (%) | MAE (deg) | Median (deg) | P90 (deg) | Within 10° (%) |
|---|---:|---:|---:|---:|---:|---:|
| PCA | | | | | | |
| Hough | | | | | | |
| Otsu/Moments | | | | | | |

Include overlay figures and describe failure cases. Do not claim full front/back heading when the estimator only predicts an undirected body axis.
