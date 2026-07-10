# Evaluation pipeline

Compares AI-judge ratings to human ratings with the statistical battery from
the shared Colab analysis (`kme_copy_of_shared_statistical_analysis.py`):
quadratic weighted kappa, ICC(2,1), Spearman, MAE/RMSE, Bland–Altman, paired
TOST equivalence, and paired Wilcoxon tests between conditions. All statistics
are implemented on numpy/scipy in [stats.py](stats.py) (no pingouin/sklearn
needed) and unit-tested against published values in
[../tests/test_evaluation.py](../tests/test_evaluation.py).

## Workflow

**1. Generate predictions across context sizes** (talks to the VLM):

```bash
python -m car_judge.run_context_sweep \
  --data "../Car Aesthetics Ratings - Prolific_July 8, 2026_10.14.csv" \
  --image-root "../selected_2000_isometric_upload_chunks_renamed" \
  --base-url "$VLM_BASE_URL" --model "$VLM_MODEL" \
  --raters 5 --context-sizes 0 5 10 15 \
  --holdout-mode fixed --test-size 8 --repeats 3 \
  --out outputs/sweep_predictions.csv
```

Add `--dry-run` first to count requests before spending GPU time. Hold-out
schemes:

| mode        | context             | test set                        | use when |
|-------------|---------------------|---------------------------------|----------|
| `fixed`     | drawn from pool     | same `--test-size` cars for all context sizes | you want *paired* comparisons across context sizes (default) |
| `remainder` | first N of a shuffle| **all** remaining cars ("hold multiple out") | you want maximum test items per condition |
| `loo`       | drawn from the rest | each car once ("hold one out")  | small per-rater car counts; most requests |

`--repeats K` re-draws the context K times per (rater, size) so ICL gains can
be separated from context-sampling luck (reported as `mae_seed_sd`).

**2. Evaluate** (offline, no server):

```bash
# Does the judge reproduce EACH PERSON's ratings?
python -m evaluation.evaluate_predictions \
  --predictions outputs/sweep_predictions.csv \
  --reference rater \
  --data ".../Prolific.csv" \
  --out-dir evaluation/outputs/sweep_rater

# Does the judge reproduce the AVERAGE person? (needs co-rated cars — anchors)
python -m evaluation.evaluate_predictions \
  --predictions outputs/sweep_predictions.csv \
  --reference car_mean --exclude-self --min-raters 2 \
  --data ".../Prolific.csv" \
  --out-dir evaluation/outputs/sweep_consensus
```

Passing several `--predictions` files (one per model/prompt variant) adds
model-vs-model paired Wilcoxon comparisons, keyed by file stem.

## Outputs (in `--out-dir`)

- `metrics.csv` — one row per source × judge × holdout mode × context size ×
  dimension (plus a pooled `ALL` row): n, MAE, RMSE, exact, within-1,
  weighted kappa, ICC(2,1), Spearman rho/p, Bland–Altman (bias + limits of
  agreement), TOST p-values, and optional bootstrap CIs (`--bootstrap`).
- `comparisons.csv` — paired Wilcoxon on per-item |error| across context
  sizes and across sources, matched on (rater, car, dimension), with
  Holm-corrected p-values. Only fully valid in `fixed`/`loo` modes where test
  items are shared across conditions.
- `baseline.csv` — human–human agreement on multiply-rated (anchor) cars:
  mean pairwise kappa / ICC / Spearman / MAE across rater pairs. This is the
  ceiling: noisy with few anchors, so treat as a reference point.
- `interpretation.txt` — Colab-style decision criteria per condition
  (agreement ≥ 80% of human–human, MAE ≤ human–human, TOST equivalence
  within `--equivalence-margin`, default ±1.0 on the 1–6 scale).
- `plots/` — MAE / kappa / ICC / Spearman vs. context size (with the
  human–human baseline as a dashed line) and Bland–Altman per condition.

## Reading the numbers

- **TOST equivalence** tests *bias* (mean signed difference), not error
  size: a judge with large but symmetric errors can pass TOST while failing
  MAE. Both are reported; require both, as the interpretation does.
- **Weighted kappa** rounds ratings to integers first (same assumption the
  Colab makes) — relevant for `car_mean` references, which are fractional.
- **`reference = rater` vs `car_mean`** answer different questions
  (personalization vs consensus); with `--exclude-self` the consensus
  reference is independent of the rater the judge was personalized to.
