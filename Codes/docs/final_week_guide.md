# Final Project Runbook

This runbook supersedes the earlier sample-only final-week workflow. The authoritative objective, split policy, evaluation tables, and complete commands are in `FINAL_PROJECT_PROTOCOL.md`.

## Setup (Windows CMD)

```bat
cd /d "C:\path\to\DIP_2026_Project8_Group1\Codes"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m compileall src test
python -m pytest test -v
```

## Full final experiment

Use the source-image-grouped split, then evaluate every orientation method on the same held-out test crops. Keep all five methods even if some are weaker. Do not run any classifier; all final methods are conventional image processing. Follow `FINAL_PROJECT_PROTOCOL.md` for the exact command sequence.

## Sample results

`PRELIMINARY_100_CROP_RESULTS.md` documents the uploaded 100-crop benchmark. It is diagnostic only and must not be presented as the final held-out result.

## Interpretation constraints

- Axis angles are undirected and evaluated modulo 180°.
- Ground-truth axis style is derived from `gt_axis_deg`; default axis-aligned tolerance is 15° from horizontal/vertical.
- `class_name` is the vehicle-category annotation. Report it separately from orientation accuracy.
- The hybrid's `route` field identifies the method selected for each crop.
- Ground-truth direction labels are used only for evaluation. The rule presumes the narrower terminal silhouette is the front; validate this assumption against sample images.
- The oracle method-selection file is a retrospective upper bound, not a deployable estimator.


## Updated final objective: directed vehicle heading

The final objective includes both the undirected heading axis and the directed heading angle. Use `src.run_heading_pipeline` after generating train and test axis predictions for all five estimators. The pipeline uses Otsu foreground segmentation and a fixed terminal-silhouette taper rule to estimate direction. The narrower end is presumed to be the front; ambiguous cases are left unresolved. See `FINAL_PROJECT_PROTOCOL.md`.
