# Final Project: Student-wise Ownership

The authoritative student responsibilities are the four `Student_Packages/Student_*/TASK.md` files. This document is retained as a navigation page; the old sample-only task plan is superseded.

- Student 1 — orientation estimators, hybrid routing and method-selection analysis.
- Student 2 — 42-configuration preprocessing ablation, ROI experiment and Otsu/moments comparison.
- Student 3 — conventional front/rear geometry analysis and rule-based heading-direction evaluation.
- Student 4 — grouped data split, unified evaluation, integration, tests and final evidence.

All students use the same source-image-grouped train/test split. No one tunes on the held-out test set. All five orientation estimators remain in the final comparison.


## Updated heading-direction ownership

The final integration must include `src.heading_direction`, `src.run_heading_pipeline`, and `src.visualize_heading_predictions`. A fixed rule-based taper detector is evaluated on held-out crops; no model is trained. Use the current `FINAL_PROJECT_PROTOCOL.md` as the authoritative run protocol.
