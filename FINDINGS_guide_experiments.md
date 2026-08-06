# Guide-conditioned judge experiments — findings (2026-07-26)

**Question.** Can the human-rating theory in `../rater_matching_guide.md` — converted
into per-person "taste cards" and a two-stage prediction recipe — beat the incumbent
judge (`b_persona` = N=12 random own-car exemplars + persona prompt, the e5_persona
winner of the earlier lever sweep)?

**Answer.** Mostly no, and the way it fails is informative. The guide's *procedure*
(two-stage: read shared styling first, then personalize) yields a small, real,
replicated MAE/RMSE gain (−0.0055 MAE at the full pool, 3× the empirical noise
floor) — but it is a **calibration** effect, not personalization: ICC / Spearman ρ /
within-rater ρ do not improve (they tick down ~0.002 on both seeds). The guide's
*facts* (the taste card alone) are a clean null. By the pre-registered rule — a win
must lower MAE/RMSE **and** raise ICC/ρ — **`b_persona` remains the recommended
judge** (`configs/recommended.json`); `g_card_twostage` is available as an optional
MAE-calibration variant.

## What was built (commit bb780c2, fix in 855b4a5)

- `car_judge/guide_note.py` — deterministic per-rater taste card programming the
  guide's §2–§6: personalized attribute weights (sex/age/income/driving/Big-Five/
  ownership), baseline & spread, endowment bumps as conditional instructions,
  perception shifts, legibility tier, and an exemplar-reconciliation footer.
  New JudgeConfig lever `include_guide`. No leakage; pure function of the profile.
- `prompt_variant="two_stage"` — the §7 recipe: STEP 1 estimate shared styling
  (with population attribute means and §5 body-style base rates), STEP 2 convert
  to the person's numbers. Configs in `configs/`.

## Search at --raters 150 (seeds 0, 1; paired per-rater Wilcoxon + cluster bootstrap)

| config vs b_persona | seed 0 dMAE | seed 1 dMAE | verdict |
|---|---|---|---|
| g_card (taste card) | −0.003 (p=.51) | −0.004 (p=.55) | null |
| g_twostage (recipe) | −0.014 (p=.002) | −0.007 (p=.09) | direction holds |
| g_card_twostage | −0.014 (CI excl. 0) | −0.011 (CI excl. 0) | finalist |

## Full-pool confirmation (672 eligible raters, 13,440 paired cells/run)

| run | MAE | ICC | ρ | RMSE | within-rater ρ |
|---|---|---|---|---|---|
| b_nocontext s0/s1 | 1.1609 / 1.1614 | 0.427 / 0.427 | 0.448 / 0.447 | 1.518 / 1.517 | 0.483 / 0.483 |
| b_persona s0/s1 | 0.8531 / 0.8504 | 0.664 / 0.666 | 0.663 / 0.665 | 1.238 / 1.234 | 0.594 / 0.591 |
| g_card_twostage s0/s1 | 0.8478 / 0.8448 | 0.661 / 0.664 | 0.661 / 0.663 | 1.224 / 1.218 | 0.592 / 0.591 |

Paired deltas vs b_persona:

- g_card_twostage: **−0.0054** [−0.0104, −0.0003] p=.050 (s0); **−0.0056**
  [−0.0103, −0.0009] p=.018 (s1). Real but small; ICC/ρ slightly DOWN both seeds.
- b_nocontext: **+0.308 / +0.311** — context is worth ~27% of MAE and lifts pooled
  ICC 0.43→0.66, ρ 0.45→0.66, within-rater ρ 0.48→0.59.

**Empirical noise floor** (identical config+split rerun, b_persona s0): macro
dMAE = 0.00003, run-level SE ≈ 0.0018 — even though 15.4% of individual
predictions flip run-to-run (server-side nondeterminism at temperature 0 with
seed set). The √N scaling from the 20-rater floor (~0.016) checks out.

## Per-dimension (full pool, seed-averaged MAE)

| config | sporty | luxurious | modern | rugged | preference |
|---|---|---|---|---|---|
| b_nocontext | 1.182 | 1.125 | 1.067 | 1.081 | 1.351 |
| b_persona | 0.893 | 0.872 | 0.854 | 0.698 | 0.942 |
| g_card_twostage | 0.891 | 0.868 | 0.852 | 0.696 | **0.924** |

The two-stage gain concentrates in **preference** (−0.018; also the only dimension
where ICC rises, 0.567→0.575) and is ~zero on rugged — the *opposite* of the
guide's prediction that the personal signal lives in rugged. With exemplars
present, rugged is already the best-predicted dimension (MAE 0.70, ICC 0.73).
Context's own gain is largest exactly where the guide said the signal lives:
rugged (−0.38) and preference (−0.41).

## Interpretation

1. **12 own-car exemplars already carry the person.** Everything the taste card
   encodes (leniency, spread, rugged tilt, ownership) is visible in 12 rated
   examples; restating it as priors adds nothing measurable (interpreted
   demographics fare no better than the raw dumps tried in the earlier sweep).
2. **Structure helps where facts don't.** Telling the model *how to think*
   (shared-styling-first, population anchors) improves calibration of the hard
   holistic dimension (preference); telling it *what to conclude* does not.
3. **The improvement is levels, not rankings.** MAE/RMSE down, ICC/ρ flat-to-down:
   population anchoring pulls predictions toward better-calibrated values without
   tracking person-specific variance any better. A judge selected for MAE alone
   would adopt it; the suite rule says don't.

## Honest nulls & caveats

- Taste card as static prior: null at n=150 (and its content is subsumed by the
  card+two-stage combo at the full pool).
- The 150-rater effect estimate (−0.014) shrank 3× at the full pool (−0.0055) —
  winner's-curse on finalist selection; full-pool confirmation was essential.
- Not tested (budget): per-dimension emphasis beyond the two-stage wording;
  confidence-aware damping as a separate lever; guide-conditioned *context
  selection*. The last is the most promising untried direction: use the card to
  pick *which* exemplars to show (e.g. span the rugged axis), attacking ranking
  rather than calibration.

## Runs & artifacts

17 live runs (1 smoke, 8 search-150, 8 full-pool incl. floor repeat and the
no-context anchors) — within the 30-run budget. Per-cell predictions +
`.metrics.json` for every run under `outputs/runs/<ts>_<tag>/per_experiment/`;
master log rows tagged `search150` / `confirm800` in `results.tsv`.
Harness fixes along the way: rater-eligibility crash at full pool (855b4a5),
`--max-retries` flag, crash-resilient resumable queue scripts
(`search_150.ps1`, `confirm_800.ps1`).
