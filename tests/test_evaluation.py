"""Offline tests for the evaluation pipeline — no VLM server required.

Run with: python -m pytest -q tests/test_evaluation.py
(or plain `python tests/test_evaluation.py`).
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import stats as jstats
from car_judge.data import Rater
from car_judge.run_context_sweep import iter_conditions


# --------------------------------------------------------------------------- #
# stats.py
# --------------------------------------------------------------------------- #
def test_kappa_perfect_and_brute_force():
    a = [1, 2, 3, 4, 5, 6, 3, 2]
    assert abs(jstats.quadratic_weighted_kappa(a, a) - 1.0) < 1e-12

    # cross-check against an independent brute-force implementation
    rng = np.random.RandomState(0)
    x = rng.randint(1, 7, 60)
    y = np.clip(x + rng.randint(-2, 3, 60), 1, 6)

    n_cat = 6
    obs = np.zeros((n_cat, n_cat))
    for xi, yi in zip(x, y):
        obs[xi - 1, yi - 1] += 1
    exp = np.outer(obs.sum(1), obs.sum(0)) / len(x)
    w = np.array([[(i - j) ** 2 for j in range(n_cat)] for i in range(n_cat)],
                 dtype=float) / (n_cat - 1) ** 2
    expected_kappa = 1 - (w * obs).sum() / (w * exp).sum()
    got = jstats.quadratic_weighted_kappa(x, y)
    assert abs(got - expected_kappa) < 1e-12


def test_icc2_1_shrout_fleiss():
    # Shrout & Fleiss (1979), Table 2: 6 targets x 4 judges. ICC(2,1) = 0.29.
    ratings = np.array([
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ], dtype=float)
    icc = jstats.icc2_1(ratings)
    assert abs(icc - 0.29) < 0.005, icc


def test_tost_paired():
    rng = np.random.RandomState(1)
    base = rng.randint(1, 7, 100).astype(float)
    # tiny noise, mean diff ~0 -> clearly equivalent within +/-1
    near = base + rng.normal(0, 0.2, 100)
    res = jstats.tost_paired(near, base, margin=1.0)
    assert res["equivalent"] == 1.0
    # constant offset of 2 -> clearly NOT equivalent within +/-1
    far = base + 2 + rng.normal(0, 0.2, 100)
    res = jstats.tost_paired(far, base, margin=1.0)
    assert res["equivalent"] == 0.0
    # zero-variance branch
    res = jstats.tost_paired(base, base, margin=1.0)
    assert res["equivalent"] == 1.0


def test_bland_altman_and_errors():
    ref = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    mod = ref + 1  # constant +1 model error
    ba = jstats.bland_altman(ref, mod)
    assert abs(ba["mean_diff"] - (-1.0)) < 1e-12
    assert abs(ba["sd_diff"]) < 1e-12
    assert abs(jstats.mae(ref, mod) - 1.0) < 1e-12
    assert abs(jstats.rmse(ref, mod) - 1.0) < 1e-12
    assert jstats.within_k(ref, mod, 1) == 1.0
    assert jstats.exact(ref, mod) == 0.0


def test_wilcoxon_guards():
    x = np.arange(10, dtype=float)
    stat, p = jstats.wilcoxon_paired(x, x)  # identical -> p = 1
    assert p == 1.0
    stat, p = jstats.wilcoxon_paired(x[:3], x[:3])  # too small -> nan
    assert np.isnan(p)


def test_summarize_pair_keys():
    rng = np.random.RandomState(2)
    ref = rng.randint(1, 7, 40)
    mod = np.clip(ref + rng.randint(-1, 2, 40), 1, 6)
    out = jstats.summarize_pair(ref, mod, bootstrap=True, n_boot=100)
    for key in ("n", "mae", "rmse", "weighted_kappa", "icc2_1", "spearman_rho",
                "tost_equivalent", "mae_ci_lo", "mae_ci_hi"):
        assert key in out, key
    assert out["mae_ci_lo"] <= out["mae"] <= out["mae_ci_hi"]


# --------------------------------------------------------------------------- #
# run_context_sweep.iter_conditions
# --------------------------------------------------------------------------- #
def _fake_rater(n_cars: int = 20, rid: str = "R1") -> Rater:
    dims = ["sporty", "luxurious", "modern", "rugged", "preference"]
    ratings = {f"car_{i}.png": {d: (i + k) % 6 + 1 for k, d in enumerate(dims)}
               for i in range(1, n_cars + 1)}
    return Rater(response_id=rid, ratings=ratings, anchor_images=[])


def test_conditions_fixed_mode():
    rater = _fake_rater()
    conds = list(iter_conditions(rater, "fixed", [0, 5, 10], repeats=2,
                                 test_size=6, split_seed=0, max_targets=None))
    # N=0 collapses to one repeat; N=5,10 get 2 each
    assert len(conds) == 1 + 2 + 2
    test_sets = {tuple(t) for _, _, _, t in conds}
    assert len(test_sets) == 1, "fixed mode must keep the test set constant"
    for n, _r, ctx, targets in conds:
        assert len(ctx) == n
        assert not set(ctx) & set(targets), "context leaked into test set"


def test_conditions_remainder_mode():
    rater = _fake_rater()
    conds = list(iter_conditions(rater, "remainder", [5, 15], repeats=2,
                                 test_size=0, split_seed=0, max_targets=None))
    for n, _r, ctx, targets in conds:
        assert len(ctx) == n
        assert len(targets) == 20 - n
        assert not set(ctx) & set(targets)
        assert set(ctx) | set(targets) == set(rater.ratings)


def test_conditions_loo_mode():
    rater = _fake_rater(n_cars=8)
    conds = list(iter_conditions(rater, "loo", [3], repeats=1,
                                 test_size=0, split_seed=0, max_targets=None))
    assert len(conds) == 8
    targets = [t for _n, _r, _c, (t,) in conds]
    assert sorted(targets) == sorted(rater.ratings.keys())
    for _n, _r, ctx, (target,) in conds:
        assert len(ctx) == 3 and target not in ctx


# --------------------------------------------------------------------------- #
# evaluate_predictions end-to-end on synthetic data
# --------------------------------------------------------------------------- #
_DIM_HEADER = {
    "sporty": "Sporty", "luxurious": "Luxurious", "modern": "Modern",
    "rugged": "Rugged", "preference": "How much do you like this car?",
}
DIMS = list(_DIM_HEADER)


def _write_synthetic_export(path: str, n_raters: int = 3, n_cars: int = 6) -> None:
    """Tiny Qualtrics-style export: every rater rates every car (co-rated)."""
    cars = [f"car_{i}.png" for i in range(1, n_cars + 1)]
    names, texts, ids = ["ResponseId"], ["Response ID"], ["{}"]
    cols = []
    q = 1
    for car in cars:
        for dim in DIMS:
            names.append(f"1_Q25_{q}")
            texts.append(f"Looking at the image, rate - {car} - {_DIM_HEADER[dim]}")
            ids.append("{}")
            cols.append((car, dim))
            q += 1
    rows = [names, texts, ids]
    rng = np.random.RandomState(7)
    for r in range(n_raters):
        row = [f"R{r + 1}"]
        for car, dim in cols:
            base = (hash((car, dim)) % 4) + 2  # car-specific "true" level 2..5
            row.append(str(int(np.clip(base + rng.randint(-1, 2), 1, 6))))
        rows.append(row)
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def _write_synthetic_predictions(path: str, data_csv: str) -> None:
    """Predictions = rater's own rating +1 (clipped): known MAE <= 1."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from car_judge.data import load_raters
    raters = load_raters(data_csv)
    fields = ["rater", "judge", "target_image", "car_name", "dimension",
              "predicted", "true", "abs_error", "n_context", "n_images",
              "enable_thinking"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rt in raters:
            for img, dims in rt.ratings.items():
                for dim, tv in dims.items():
                    pv = min(tv + 1, 6)
                    w.writerow({
                        "rater": rt.response_id, "judge": "in_context",
                        "target_image": img, "car_name": "", "dimension": dim,
                        "predicted": pv, "true": tv, "abs_error": abs(pv - tv),
                        "n_context": 5, "n_images": 6, "enable_thinking": False,
                    })


def test_evaluate_predictions_end_to_end():
    from evaluation import evaluate_predictions as ep
    import pandas as pd

    with tempfile.TemporaryDirectory() as d:
        data_csv = os.path.join(d, "export.csv")
        pred_csv = os.path.join(d, "predictions.csv")
        _write_synthetic_export(data_csv)
        _write_synthetic_predictions(pred_csv, data_csv)

        # --- reference = rater ---
        out1 = os.path.join(d, "out_rater")
        ep.main(["--predictions", pred_csv, "--reference", "rater",
                 "--data", data_csv, "--out-dir", out1])
        metrics = pd.read_csv(os.path.join(out1, "metrics.csv"))
        pooled = metrics[metrics["dimension"] == "ALL"].iloc[0]
        # predictions are true+1 clipped at 6 -> MAE in (0, 1]
        assert 0.0 < pooled["mae"] <= 1.0
        assert pooled["n"] == 3 * 6 * 5
        for f in ("comparisons.csv", "baseline.csv", "interpretation.txt"):
            assert os.path.exists(os.path.join(out1, f)), f
        assert os.path.isdir(os.path.join(out1, "plots"))

        baseline = pd.read_csv(os.path.join(out1, "baseline.csv"))
        assert "ALL" in set(baseline["dimension"])

        # --- reference = car mean, excluding self ---
        out2 = os.path.join(d, "out_mean")
        ep.main(["--predictions", pred_csv, "--reference", "car_mean",
                 "--exclude-self", "--min-raters", "2",
                 "--data", data_csv, "--out-dir", out2, "--no-plots"])
        metrics2 = pd.read_csv(os.path.join(out2, "metrics.csv"))
        assert len(metrics2[metrics2["dimension"] == "ALL"]) == 1
        assert metrics2["mae"].notna().all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"\nall {len(fns)} tests passed")
