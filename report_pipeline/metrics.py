"""Metric computation — thin wrappers over the committed evaluation.stats module.

Every metric here is computed the SAME way run_autojudge/results.tsv computed it,
so recomputing a judge's numbers from its saved predictions reproduces results.tsv
exactly (see build_report --check). Metrics are POOLED over the cells in scope
(overall = all cells; per-dimension = that dimension's cells), except
within_rater_rho which is the mean of each rater's own Spearman.
"""
from __future__ import annotations

import sys, os
from typing import Dict, List

import numpy as np
import pandas as pd

# import the repo's canonical stats (ICC(2,1), Spearman, weighted kappa, MAE, ...)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation import stats as st  # noqa: E402

from .data import DIMS  # noqa: E402


def pooled_suite(true, pred) -> Dict[str, float]:
    t = np.asarray(true, float); p = np.asarray(pred, float)
    # A constant predictor (e.g. pop_mean within one dimension) has no variance,
    # so ranking/agreement metrics are undefined -- report NaN rather than a
    # meaningless value. Error metrics (MAE/RMSE/within1) stay valid.
    const = len(p) > 0 and float(np.ptp(p)) == 0.0
    rho = float("nan") if const else st.spearman(t, p)[0]
    icc = float("nan") if const else st.icc2_1(np.column_stack([t, p]))
    wk  = float("nan") if const else st.quadratic_weighted_kappa(t, p)
    return {
        "n": int(len(t)),
        "mae": st.mae(t, p),
        "rmse": st.rmse(t, p),
        "within1": st.within_k(t, p, 1),
        "icc": icc,
        "rho": rho,
        "wkappa": wk,
    }


def within_rater_rho(df: pd.DataFrame) -> float:
    """Mean over raters of each rater's Spearman(true, predicted) on their own
    cells — the personalization-quality signal pooled correlation can't see."""
    vals: List[float] = []
    for _, g in df.groupby("rater"):
        rho, _ = st.spearman(g["true"].to_numpy(float), g["predicted"].to_numpy(float))
        if np.isfinite(rho):
            vals.append(rho)
    return float(np.mean(vals)) if vals else float("nan")


def metrics_for_judge(df: pd.DataFrame, judge: str) -> List[dict]:
    """One row per scope (overall + each dimension) for a single judge/seed frame."""
    rows = []
    scopes = [("overall", df)] + [(d, df[df["dimension"] == d]) for d in DIMS]
    for scope, sub in scopes:
        m = pooled_suite(sub["true"], sub["predicted"])
        m.update(judge=judge, scope=scope, within_rater_rho=within_rater_rho(sub))
        rows.append(m)
    return rows


def seed_average(per_seed: List[pd.DataFrame]) -> pd.DataFrame:
    """Average the numeric metric columns across seeds, keyed by (judge, scope)."""
    allrows = pd.concat(per_seed, ignore_index=True)
    num = [c for c in allrows.columns if c not in ("judge", "scope", "n")]
    agg = (allrows.groupby(["judge", "scope"], sort=False)[num].mean().reset_index())
    # carry n (identical across seeds) for reference
    ncol = allrows.groupby(["judge", "scope"], sort=False)["n"].first().reset_index()["n"]
    agg["n"] = ncol
    return agg


# --------------------------------------------------------------------------- #
# Paired comparison: dMAE with a cluster (rater-level) bootstrap CI
# --------------------------------------------------------------------------- #
def paired_dmae(dfA: pd.DataFrame, dfB: pd.DataFrame,
                n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """dMAE = mean(|errA| - |errB|) over the identical cells of A and B, with a
    percentile CI from resampling RATERS (the independent unit), pooled over
    whatever seeds are present. Negative => A more accurate than B.
    """
    key = ["rater", "target_image", "dimension"]
    a = dfA[key + ["true", "predicted"]].rename(columns={"predicted": "pA"})
    b = dfB[key + ["predicted"]].rename(columns={"predicted": "pB"})
    m = a.merge(b, on=key, how="inner")
    if not len(m):
        return {"dmae": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p": float("nan"), "n_cells": 0}
    m["diff"] = (m["pA"] - m["true"]).abs() - (m["pB"] - m["true"]).abs()
    dmae = float(m["diff"].mean())

    # cluster bootstrap over raters
    per_rater = m.groupby("rater")["diff"].agg(["sum", "count"])
    sums = per_rater["sum"].to_numpy(float)
    cnts = per_rater["count"].to_numpy(float)
    R = len(sums)
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.randint(0, R, size=R)
        boots[i] = sums[idx].sum() / cnts[idx].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p: how often the sign flips vs the point estimate
    if dmae < 0:
        p = 2.0 * float(np.mean(boots >= 0))
    else:
        p = 2.0 * float(np.mean(boots <= 0))
    return {"dmae": dmae, "ci_lo": float(lo), "ci_hi": float(hi),
            "p": min(1.0, p), "n_cells": int(len(m))}
