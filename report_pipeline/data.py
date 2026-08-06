"""Discover ground-truth runs and load predictions + human ratings.

The ground truth is the set of per-prediction CSVs written by run_autojudge for
the full-pool confirmation runs, living under
``<runs_root>/<timestamp>_fp_<judge>_s<seed>/per_experiment/<judge>.csv``.
Each has columns: rater, judge, target_image, dimension, predicted, true, ...
"""
from __future__ import annotations

import os
import re
import glob
from typing import Dict, List, Tuple

import pandas as pd

# canonical rating dimensions, in display order
DIMS: List[str] = ["sporty", "luxurious", "modern", "rugged", "preference"]

# human long-csv column -> canonical dimension key
HUMAN_METRIC_COLS: Dict[str, str] = {
    "sporty": "sporty", "luxurious": "luxurious", "modern": "modern",
    "rugged": "rugged", "overall_preference": "preference",
}

# parse "..._fp_<judge>_s<seed>[_repeat]" from a run-dir name
_RUN_RX = re.compile(r"_fp_(?P<judge>.+?)_s(?P<seed>\d+)(?P<suffix>_[a-z0-9]+)?$")


def discover_runs(runs_root: str, include_repeats: bool = False
                  ) -> Dict[Tuple[str, int], str]:
    """Map (judge_name, seed) -> path of its per-prediction CSV.

    Scans ``runs_root`` for full-pool (``_fp_``) run directories and picks the
    single CSV under each ``per_experiment/``. ``_repeat`` runs (the noise-floor
    reruns) are skipped unless ``include_repeats`` is set; if both a canonical
    and a repeat exist for the same (judge, seed) the canonical wins.
    """
    found: Dict[Tuple[str, int], str] = {}
    for d in sorted(glob.glob(os.path.join(runs_root, "*_fp_*"))):
        if not os.path.isdir(d):
            continue
        m = _RUN_RX.search(os.path.basename(d))
        if not m:
            continue
        is_repeat = bool(m.group("suffix"))
        if is_repeat and not include_repeats:
            continue
        judge, seed = m.group("judge"), int(m.group("seed"))
        csvs = glob.glob(os.path.join(d, "per_experiment", "*.csv"))
        if not csvs:
            continue
        key = (judge, seed)
        # canonical (no suffix) beats a repeat if both are present
        if key in found and is_repeat:
            continue
        found[key] = csvs[0]
    if not found:
        raise FileNotFoundError(
            f"No full-pool prediction CSVs found under {runs_root!r}. "
            "Expected <runs_root>/*_fp_<judge>_s<seed>/per_experiment/<judge>.csv")
    return found


def load_predictions(path: str) -> pd.DataFrame:
    """Load one per-prediction CSV; keep graded rows (both predicted & true present)."""
    df = pd.read_csv(path)
    need = {"rater", "target_image", "dimension", "predicted", "true"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    df = df.dropna(subset=["predicted", "true"]).copy()
    df["predicted"] = df["predicted"].astype(float)
    df["true"] = df["true"].astype(float)
    return df


def cells_of(df: pd.DataFrame) -> pd.DataFrame:
    """The evaluation cells (identical across judges for a given seed): the
    (rater, car, dimension, true-rating) grid the baselines are also scored on."""
    return df[["rater", "target_image", "dimension", "true"]].reset_index(drop=True)


def load_human_ratings(human_csv: str) -> pd.DataFrame:
    """Load the merged long human ratings and add a normalized image key ('car_N.png')."""
    H = pd.read_csv(human_csv, low_memory=False)
    miss = [c for c in HUMAN_METRIC_COLS if c not in H.columns]
    if miss:
        raise ValueError(f"human csv missing metric columns: {miss}")
    H["img"] = (H["car"].astype(str)
                .apply(lambda s: s if s.endswith(".png") else f"{s}.png"))
    return H
