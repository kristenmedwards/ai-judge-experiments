# Reproducible AI-judge report

Rebuilds the judge-vs-human metrics, statistical tests, and figures **from the
saved per-prediction ground truth** — nothing here re-queries a model. Every
number is a deterministic function of two inputs:

1. the per-prediction CSVs each full-pool run wrote
   (`outputs/runs/*_fp_<judge>_s<seed>/per_experiment/<judge>.csv`), one row per
   `rater × held-out car × dimension` with the model's `predicted` and the
   rater's `true` rating; and
2. the merged human ratings CSV (only for the two naive baselines).

Metric functions are imported from the committed `evaluation/stats.py`, so a
judge's recomputed numbers reproduce `results.tsv` exactly (`--check` asserts
this: all six judge×seed MAEs match to < 1e-4).

## Run

```bash
cd ai_judge_experiments
python build_report.py \
  --runs-root outputs/runs \
  --human ../car_ratings_long_July8_July22_merged_with_color_and_Q23.csv \
  --check
```

Outputs land in `report_and_figures_reproduced/` (never clobbers the old
hand-made `report_and_figures/`):

- `metrics_by_dimension_seedavg.csv` — judges **and** baselines, per dimension +
  overall, seed-averaged over seeds 0/1.
- `statistical_tests_seedavg.csv` — paired ΔMAE for the key comparisons, each
  with a rater-cluster bootstrap 95% CI and two-sided p.
- `provenance.json` — exact input paths, options, population means, and the
  `results.tsv` reproduction check.
- `figures/*.png` — value ladder, MAE by dimension, ΔMAE forest, finalist
  per-dimension, and one panel per dimension.

## The baselines (the point of this rebuild)

- **`pop_mean`** — predict each dimension's population grand mean (a constant).
- **`car_mean`** — predict the mean rating *other* raters gave that car on that
  dimension, i.e. **leave-one-out** (`--car-mean-mode loo`, the default). The
  target rater's own score is excluded so the baseline can't peek at the answer;
  with ~3 raters/car, including it (the old report's behaviour, reproducible via
  `--car-mean-mode include_self`) understates the baseline's MAE by ~0.09.

Both are scored on the **identical held-out cells** as the judges, so all
comparisons are properly paired.

## Files

| file | role |
|---|---|
| `data.py` | discover `fp_*` runs, load predictions/cells, load human ratings |
| `baselines.py` | `pop_mean` and leave-one-out `car_mean` predictors |
| `metrics.py` | pooled metric suite (via `evaluation.stats`), within-rater ρ, seed-averaging, cluster-bootstrap ΔMAE |
| `figures.py` | the figures (matplotlib, Okabe-Ito colourblind-safe) |
| `../build_report.py` | orchestrator / CLI |
