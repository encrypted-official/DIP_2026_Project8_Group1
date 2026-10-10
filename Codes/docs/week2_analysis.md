# Week 2 preliminary analysis

Do not fill in numbers until the code has been run. These are placeholders, not results.

## Experiment setup

- Dataset split: validation
- Images processed: ___
- Vehicle crops saved: ___
- Difficulty-1 objects included? Yes / No
- PCA preprocessing: grayscale + Gaussian blur + CLAHE + Canny
- Hough preprocessing: same edge map as PCA
- Target: undirected axis angle in 0–90 degrees

## Results

| Method | Valid predictions | Coverage (%) | MAE (°) | Median (°) | P90 (°) | Within 10° (%) |
| --- | --- | --- | --- | --- | --- | --- |
| PCA on edges | — | — | — | — | — | — |
| Hough lines | — | — | — | — | — | — |

## Observations

1. Which method had fewer invalid predictions?
2. Which method had the lower error on this small sample?
3. Which classes were hardest? Do not make a class-level claim with only one or two samples.
4. Did tiny crops, shadows, low contrast, or background lines cause visible errors?
5. What should be tested next?

## Limitations

This is a small preliminary experiment, not a final benchmark. The two methods are sensitive
to image structure and preprocessing. The axis estimate does not identify the vehicle front.
Do not generalize from the first 10 images to the whole dataset.
