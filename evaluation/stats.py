"""Statistical functions for judge-vs-human evaluation (numpy/scipy only).

These mirror the tests used in the shared Colab analysis
(``kme_copy_of_shared_statistical_analysis.py``) — quadratic weighted kappa,
ICC(2,1), Spearman, MAE/RMSE, Bland–Altman, and paired TOST equivalence —
implemented directly on numpy/scipy so the repo does not need pingouin or
scikit-learn.

All pairwise functions take two equal-length 1-D arrays (reference first,
model/prediction second) and ignore nothing: callers are expected to have
already dropped missing values (see ``evaluate_predictions._paired``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as st


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #
def quadratic_weighted_kappa(
    a: Sequence[float],
    b: Sequence[float],
    min_rating: int = 1,
    max_rating: int = 6,
) -> float:
    """Cohen's kappa with quadratic weights over an integer rating scale.

    Ratings are rounded to integers first (same assumption as the Colab
    script, which rounds before calling sklearn's ``cohen_kappa_score``).
    Returns nan if fewer than 2 pairs or if there is no rating variance to
    disagree over (degenerate expected matrix).
    """
    x = np.clip(np.round(np.asarray(a, dtype=float)), min_rating, max_rating).astype(int)
    y = np.clip(np.round(np.asarray(b, dtype=float)), min_rating, max_rating).astype(int)
    n = len(x)
    if n < 2:
        return float("nan")

    n_cat = max_rating - min_rating + 1
    observed = np.zeros((n_cat, n_cat), dtype=float)
    for xi, yi in zip(x - min_rating, y - min_rating):
        observed[xi, yi] += 1.0

    hist_x = observed.sum(axis=1)
    hist_y = observed.sum(axis=0)
    expected = np.outer(hist_x, hist_y) / n

    idx = np.arange(n_cat, dtype=float)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (n_cat - 1) ** 2

    denom = (weights * expected).sum()
    if denom <= 0:
        return float("nan")
    return float(1.0 - (weights * observed).sum() / denom)


def icc2_1(ratings: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater.

    ``ratings`` is an (n_targets, k_raters) matrix with no missing values.
    Classic Shrout & Fleiss (1979) formulation via the two-way ANOVA mean
    squares. Returns nan for degenerate inputs.
    """
    m = np.asarray(ratings, dtype=float)
    if m.ndim != 2:
        raise ValueError("ratings must be a 2-D (targets x raters) matrix")
    n, k = m.shape
    if n < 2 or k < 2:
        return float("nan")

    grand = m.mean()
    row_means = m.mean(axis=1)
    col_means = m.mean(axis=0)

    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((m - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols

    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))

    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if not math.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return float((msr - mse) / denom)


def spearman(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Spearman correlation (rho, p). nan/nan for constant or tiny samples."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan")
    res = st.spearmanr(x, y)
    return float(res.correlation), float(res.pvalue)


# --------------------------------------------------------------------------- #
# Error metrics
# --------------------------------------------------------------------------- #
def mae(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean(np.abs(x - y))) if len(x) else float("nan")


def rmse(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((x - y) ** 2))) if len(x) else float("nan")


def within_k(a: Sequence[float], b: Sequence[float], k: int = 1) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.mean(np.abs(x - y) <= k)) if len(x) else float("nan")


def exact(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.round(np.asarray(a, dtype=float)), np.round(np.asarray(b, dtype=float))
    return float(np.mean(x == y)) if len(x) else float("nan")


def bland_altman(reference: Sequence[float], model: Sequence[float]) -> Dict[str, float]:
    """Bland–Altman stats for reference - model differences."""
    ref = np.asarray(reference, dtype=float)
    mod = np.asarray(model, dtype=float)
    diff = ref - mod
    if len(diff) < 2:
        return {"mean_diff": float("nan"), "sd_diff": float("nan"),
                "upper_loa": float("nan"), "lower_loa": float("nan"),
                "outside_loa_pct": float("nan")}
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff
    outside = float(np.mean((diff > upper) | (diff < lower)) * 100.0)
    return {"mean_diff": mean_diff, "sd_diff": sd_diff,
            "upper_loa": upper, "lower_loa": lower, "outside_loa_pct": outside}


# --------------------------------------------------------------------------- #
# Equivalence & paired tests
# --------------------------------------------------------------------------- #
def tost_paired(
    x: Sequence[float],
    y: Sequence[float],
    margin: float,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired two one-sided tests for equivalence within +/- margin.

    Same math as the Colab ``ttost_paired``. ``equivalent`` is 1.0 when both
    one-sided tests reject at ``alpha`` (i.e. the mean difference is inside
    the margin), 0.0 otherwise, nan when undefined (n < 2 or zero variance).
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = len(d)
    out = {"t_lower": float("nan"), "p_lower": float("nan"),
           "t_upper": float("nan"), "p_upper": float("nan"),
           "mean_diff": float("nan"), "sd_diff": float("nan"),
           "n": float(n), "equivalent": float("nan")}
    if n < 2:
        return out
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1))
    out["mean_diff"], out["sd_diff"] = mean_d, sd_d
    if sd_d == 0:
        # No variance: equivalent iff the constant difference is in-bounds.
        out["equivalent"] = float(-margin < mean_d < margin)
        return out
    se = sd_d / math.sqrt(n)
    t_lower = (mean_d - (-margin)) / se
    t_upper = (mean_d - margin) / se
    p_lower = float(1 - st.t.cdf(t_lower, df=n - 1))
    p_upper = float(st.t.cdf(t_upper, df=n - 1))
    out.update(t_lower=float(t_lower), p_lower=p_lower,
               t_upper=float(t_upper), p_upper=p_upper,
               equivalent=float(p_lower < alpha and p_upper < alpha))
    return out


def wilcoxon_paired(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank (stat, p); nan/nan when undefined.

    Used to compare absolute errors of two judges / context sizes on the
    same items. All-zero differences mean the two are identical -> p = 1.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if len(d) < 5:
        return float("nan"), float("nan")
    if np.all(d == 0):
        return 0.0, 1.0
    res = st.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    reference: Sequence[float],
    model: Sequence[float],
    metric,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a paired metric(ref, model)."""
    ref = np.asarray(reference, dtype=float)
    mod = np.asarray(model, dtype=float)
    n = len(ref)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        v = metric(ref[idx], mod[idx])
        if isinstance(v, tuple):  # e.g. spearman returns (rho, p)
            v = v[0]
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo = (1 - ci) / 2 * 100
    return (float(np.percentile(vals, lo)),
            float(np.percentile(vals, 100 - lo)))


# --------------------------------------------------------------------------- #
# One-stop pairwise summary
# --------------------------------------------------------------------------- #
def summarize_pair(
    reference: Sequence[float],
    model: Sequence[float],
    equivalence_margin: float = 1.0,
    alpha: float = 0.05,
    min_rating: int = 1,
    max_rating: int = 6,
    bootstrap: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """All headline metrics for one (reference, model) rating vector pair."""
    ref = np.asarray(reference, dtype=float)
    mod = np.asarray(model, dtype=float)
    n = len(ref)

    rho, rho_p = spearman(ref, mod)
    ba = bland_altman(ref, mod)
    eq = tost_paired(mod, ref, margin=equivalence_margin, alpha=alpha)
    icc = (icc2_1(np.column_stack([ref, mod])) if n >= 2 else float("nan"))

    out = {
        "n": float(n),
        "mae": mae(ref, mod),
        "rmse": rmse(ref, mod),
        "exact": exact(ref, mod),
        "within1": within_k(ref, mod, 1),
        "weighted_kappa": quadratic_weighted_kappa(ref, mod, min_rating, max_rating),
        "icc2_1": icc,
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "mean_diff": ba["mean_diff"],
        "loa_lower": ba["lower_loa"],
        "loa_upper": ba["upper_loa"],
        "outside_loa_pct": ba["outside_loa_pct"],
        "tost_p_lower": eq["p_lower"],
        "tost_p_upper": eq["p_upper"],
        "tost_equivalent": eq["equivalent"],
    }
    if bootstrap:
        out["mae_ci_lo"], out["mae_ci_hi"] = bootstrap_ci(
            ref, mod, mae, n_boot=n_boot, seed=seed)
        out["kappa_ci_lo"], out["kappa_ci_hi"] = bootstrap_ci(
            ref, mod,
            lambda a, b: quadratic_weighted_kappa(a, b, min_rating, max_rating),
            n_boot=n_boot, seed=seed)
    return out
