"""Reproducible AI-judge report pipeline.

Rebuilds the metrics table, statistical tests, and figures for the
personalized-judge study *from the per-prediction ground truth* — the
``outputs/runs/*fp_*/per_experiment/<judge>.csv`` files each judge run wrote,
one row per (rater x held-out car x dimension) with the model's prediction and
that rater's true rating. Nothing here re-queries a model; every number is a
deterministic function of those saved predictions plus the human ratings CSV.

Modules
-------
data       discover the fp_* runs, load predictions/cells, load human ratings
baselines  pop_mean and leave-one-out car_mean predictors (the honest baselines)
metrics    pooled metric suite (via evaluation.stats), within-rater rho,
           seed-averaging, and paired cluster-bootstrap dMAE tests
figures    the report figures (matplotlib, colourblind-safe)

Run it with ``python build_report.py`` (see that script / report_pipeline/README.md).
"""
