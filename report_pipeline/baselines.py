"""Naive reference predictors, scored on the SAME held-out cells as the judges.

pop_mean   Predict, for every cell, the population grand mean of that dimension
           (a per-dimension constant). "Just predict the average sporty score."

car_mean   Predict the mean rating that OTHER raters gave this specific car on
           this dimension — i.e. leave-one-out consensus. Excluding the target
           rater's own score is deliberate: including it lets the baseline peek
           at the answer (with ~3 raters/car the self term is ~1/3 of the mean),
           which flatters the baseline. Cars with no other rater fall back to
           pop_mean for that dimension.

Both are deterministic functions of the human ratings; no model is queried.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .data import HUMAN_METRIC_COLS


def population_means(H: pd.DataFrame) -> Dict[str, float]:
    """Grand mean of each dimension over all human ratings."""
    return {dim: float(H[col].dropna().mean())
            for col, dim in HUMAN_METRIC_COLS.items()}


def car_rating_pools(H: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Per dimension: a DataFrame indexed by image with the sum & count of all
    human ratings of that car (used to form the leave-one-out mean fast)."""
    pools: Dict[str, pd.DataFrame] = {}
    for col, dim in HUMAN_METRIC_COLS.items():
        g = H.dropna(subset=[col]).groupby("img")[col].agg(sum="sum", count="count")
        pools[dim] = g
    return pools


def predict_pop_mean(cells: pd.DataFrame, pop: Dict[str, float]) -> np.ndarray:
    return cells["dimension"].map(pop).to_numpy(dtype=float)


def predict_car_mean_loo(cells: pd.DataFrame, pools: Dict[str, pd.DataFrame],
                         pop: Dict[str, float], include_self: bool = False
                         ) -> Tuple[np.ndarray, int]:
    """Leave-one-out (default) or include-self per-car mean for each cell.

    Returns (predictions, n_fallback) where n_fallback counts cells whose car
    had no other rater (LOO undefined) and fell back to the population mean.
    """
    preds = np.empty(len(cells), dtype=float)
    n_fallback = 0
    # vectorized-ish per dimension
    for dim in cells["dimension"].unique():
        pool = pools[dim]
        mask = (cells["dimension"] == dim).to_numpy()
        sub = cells.loc[mask]
        s = pool["sum"].reindex(sub["target_image"]).to_numpy(dtype=float)
        n = pool["count"].reindex(sub["target_image"]).to_numpy(dtype=float)
        tv = sub["true"].to_numpy(dtype=float)
        if include_self:
            with np.errstate(invalid="ignore", divide="ignore"):
                vals = s / n
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                vals = (s - tv) / (n - 1.0)      # remove the target's own rating
        # fallback to population mean where undefined (car unseen or no co-raters)
        bad = ~np.isfinite(vals)
        n_fallback += int(bad.sum())
        vals[bad] = pop[dim]
        preds[mask] = vals
    return preds, n_fallback


def baseline_frame(cells: pd.DataFrame, name: str, preds: np.ndarray) -> pd.DataFrame:
    """Wrap baseline predictions as a judge-style prediction frame so the metrics
    code treats baselines and judges identically."""
    out = cells.copy()
    out["judge"] = name
    out["predicted"] = preds.astype(float)
    return out
