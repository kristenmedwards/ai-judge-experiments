"""Evaluate AI-judge predictions against human ratings.

Consumes the predictions CSV written by ``car_judge.run_experiment`` or
``car_judge.run_context_sweep`` and runs the statistical battery from the
shared Colab analysis (weighted kappa, ICC(2,1), Spearman, MAE/RMSE,
Bland–Altman, paired TOST equivalence, paired Wilcoxon between conditions),
grouped by judge x context size x dimension.

Reference modes (``--reference``):

rater     Compare each prediction to THAT rater's own rating of the held-out
          car (the ``true`` column). This is the personalization question:
          "does the judge reproduce this person?"

car_mean  Compare each prediction to the average human rating of the car
          across all raters who rated it (requires ``--data``, the raw
          Qualtrics export). Cars rated by fewer than ``--min-raters`` people
          are dropped — with the current design that is mostly the anchor
          cars. ``--exclude-self`` removes the context rater from the mean so
          the reference is independent of the person the judge was
          personalized to. This is the consensus question: "does the judge
          reproduce the average person?"

When ``--data`` is given, a human–human baseline is also computed from cars
rated by 2+ raters (the anchors): mean pairwise kappa / Spearman / MAE across
rater pairs. Judge metrics are then interpreted against that baseline the
same way the Colab script checks AI-vs-expert against expert-vs-expert
(agreement >= 80% of human-human, MAE <= human-human, TOST equivalence).

Examples
--------
    python -m evaluation.evaluate_predictions \
      --predictions outputs/sweep_predictions.csv \
      --reference rater \
      --data "../Car Aesthetics Ratings - Prolific_July 8, 2026_10.14.csv" \
      --out-dir evaluation/outputs/sweep_rater

    # consensus reference on the anchor cars, self excluded
    python -m evaluation.evaluate_predictions \
      --predictions outputs/sweep_predictions.csv \
      --reference car_mean --exclude-self --min-raters 3 \
      --data ".../Prolific.csv" \
      --out-dir evaluation/outputs/sweep_consensus

Passing several prediction CSVs (e.g. one per model or prompt variant)
evaluates each under its file stem as ``source`` and adds source-vs-source
paired Wilcoxon comparisons — the Colab's "Model A vs Model B" test.

Outputs in --out-dir: metrics.csv, comparisons.csv, baseline.csv (if --data),
interpretation.txt, and plots/.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from car_judge.config import DIMENSIONS, SCALE_MIN, SCALE_MAX  # noqa: E402
from car_judge import data as datamod  # noqa: E402

try:  # both `python -m evaluation.evaluate_predictions` and direct execution
    from . import stats as jstats
except ImportError:  # pragma: no cover
    import stats as jstats  # type: ignore

# ---- plot styling (validated palette; see evaluation/README.md) ----------- #
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
DIM_COLORS = dict(zip(DIMENSIONS, ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]))
SCATTER = "#2a78d6"

GROUP_COLS = ["source", "judge", "holdout_mode", "context_size"]
ITEM_COLS = ["rater", "target_image", "dimension"]


# --------------------------------------------------------------------------- #
# Loading & reference construction
# --------------------------------------------------------------------------- #
def load_predictions(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source"] = os.path.splitext(os.path.basename(p))[0]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True, sort=False)

    # run_experiment.py output lacks the sweep columns; normalize.
    if "context_size" not in df.columns:
        df["context_size"] = df["n_context"]
    if "draw_seed" not in df.columns:
        df["draw_seed"] = 0
    if "holdout_mode" not in df.columns:
        df["holdout_mode"] = "fixed"

    df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    df["true"] = pd.to_numeric(df["true"], errors="coerce")
    df["context_size"] = pd.to_numeric(df["context_size"], errors="coerce").astype(int)
    n_before = len(df)
    df = df.dropna(subset=["predicted"]).reset_index(drop=True)
    if n_before - len(df):
        print(f"[load] dropped {n_before - len(df)} rows with missing predictions")
    print(f"[load] {len(df)} prediction rows from {len(paths)} file(s)")
    return df


def ratings_long(data_csv: str) -> pd.DataFrame:
    """All human ratings as (target_image, dimension, rater, rating) rows."""
    recs = []
    for rt in datamod.load_raters(data_csv):
        for img, dims in rt.ratings.items():
            for dim, val in dims.items():
                recs.append((img, dim, rt.response_id, float(val)))
    return pd.DataFrame(recs, columns=["target_image", "dimension", "rater", "rating"])


def add_reference(
    df: pd.DataFrame,
    mode: str,
    human: Optional[pd.DataFrame],
    exclude_self: bool,
    min_raters: int,
) -> pd.DataFrame:
    if mode == "rater":
        out = df.dropna(subset=["true"]).copy()
        out["reference"] = out["true"]
        return out

    if human is None:
        raise SystemExit("--reference car_mean requires --data (the raw survey CSV)")

    agg = (human.groupby(["target_image", "dimension"])["rating"]
           .agg(ref_sum="sum", ref_count="count").reset_index())
    out = df.merge(agg, on=["target_image", "dimension"], how="left")

    if exclude_self:
        own = human.rename(columns={"rating": "self_rating"})
        out = out.merge(own, on=["target_image", "dimension", "rater"], how="left")
        has_self = out["self_rating"].notna()
        out["ref_sum"] = out["ref_sum"] - out["self_rating"].fillna(0.0)
        out["ref_count"] = out["ref_count"] - has_self.astype(int)

    out = out[out["ref_count"].fillna(0) >= min_raters].copy()
    out["reference"] = out["ref_sum"] / out["ref_count"]
    print(f"[ref ] car_mean (exclude_self={exclude_self}, min_raters={min_raters}): "
          f"{len(out)} rows kept of {len(df)}")
    if not len(out):
        raise SystemExit(
            "no prediction targets have enough co-raters for car_mean — "
            "lower --min-raters or check that --data matches the predictions")
    return out


# --------------------------------------------------------------------------- #
# Metrics table
# --------------------------------------------------------------------------- #
def metrics_table(df: pd.DataFrame, margin: float, alpha: float,
                  bootstrap: bool, n_boot: int) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(GROUP_COLS, sort=True):
        slices = [("ALL", g)] + [(d, g[g["dimension"] == d]) for d in DIMENSIONS]
        for dim, gd in slices:
            if gd.empty:
                continue
            m = jstats.summarize_pair(
                gd["reference"].to_numpy(), gd["predicted"].to_numpy(),
                equivalence_margin=margin, alpha=alpha,
                min_rating=SCALE_MIN, max_rating=SCALE_MAX,
                bootstrap=bootstrap, n_boot=n_boot)
            # context-draw variability: MAE spread across draw seeds
            seed_mae = (gd.assign(err=(gd["reference"] - gd["predicted"]).abs())
                        .groupby("draw_seed")["err"].mean())
            row = dict(zip(GROUP_COLS, keys))
            row.update(dimension=dim, n_seeds=len(seed_mae),
                       mae_seed_sd=float(seed_mae.std(ddof=1)) if len(seed_mae) > 1 else np.nan)
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Paired comparisons (Wilcoxon on per-item absolute errors)
# --------------------------------------------------------------------------- #
def _item_errors(g: pd.DataFrame) -> pd.DataFrame:
    """Mean |error| per test item, averaged over context draws."""
    g = g.assign(err=(g["reference"] - g["predicted"]).abs())
    return g.groupby(ITEM_COLS, as_index=False)["err"].mean()


def _holm(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj

def comparisons_table(df: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon on matched test items.

    (a) across context sizes within a source, (b) across sources at the same
    context size. Items are matched on (rater, target_image, dimension); in
    'remainder' mode test sets differ per size, so only the overlap is used.
    """
    rows = []

    for (src, hold), g in df.groupby(["source", "holdout_mode"]):
        sizes = sorted(g["context_size"].unique())
        per_size = {n: _item_errors(g[g["context_size"] == n]) for n in sizes}
        for n1, n2 in combinations(sizes, 2):
            merged = per_size[n1].merge(per_size[n2], on=ITEM_COLS,
                                        suffixes=("_a", "_b"))
            if len(merged) < 5:
                continue
            stat, p = jstats.wilcoxon_paired(merged["err_a"], merged["err_b"])
            rows.append({
                "comparison": f"context {n1} vs {n2}", "source": src,
                "holdout_mode": hold, "n_items": len(merged),
                "mae_a": merged["err_a"].mean(), "mae_b": merged["err_b"].mean(),
                "wilcoxon_stat": stat, "p": p,
            })

    sources = sorted(df["source"].unique())
    if len(sources) > 1:
        for (hold, n), g in df.groupby(["holdout_mode", "context_size"]):
            per_src = {s: _item_errors(g[g["source"] == s]) for s in sources}
            for s1, s2 in combinations(sources, 2):
                merged = per_src[s1].merge(per_src[s2], on=ITEM_COLS,
                                           suffixes=("_a", "_b"))
                if len(merged) < 5:
                    continue
                stat, p = jstats.wilcoxon_paired(merged["err_a"], merged["err_b"])
                rows.append({
                    "comparison": f"{s1} vs {s2} @ context {n}", "source": "both",
                    "holdout_mode": hold, "n_items": len(merged),
                    "mae_a": merged["err_a"].mean(), "mae_b": merged["err_b"].mean(),
                    "wilcoxon_stat": stat, "p": p,
                })

    out = pd.DataFrame(rows)
    if len(out):
        valid = out["p"].notna()
        out["p_holm"] = np.nan
        out.loc[valid, "p_holm"] = _holm(out.loc[valid, "p"].to_numpy())
        out["significant"] = out["p_holm"] < 0.05
    return out


# --------------------------------------------------------------------------- #
# Human-human baseline (anchor cars)
# --------------------------------------------------------------------------- #
def human_baseline(human: pd.DataFrame, min_overlap: int = 2) -> pd.DataFrame:
    """Mean pairwise inter-rater stats on cars rated by 2+ raters.

    This is the evaluation ceiling: an AI judge cannot be expected to agree
    with a person more than people agree with each other. With only a few
    anchor cars per rater pair these estimates are noisy — treat them as a
    reference point, not ground truth.
    """
    rows = []
    slices = [("ALL", human)] + [(d, human[human["dimension"] == d]) for d in DIMENSIONS]
    for dim, g in slices:
        # rater x car matrix; keep cars seen by >= 2 raters
        pivot = g.pivot_table(index="rater", columns=["target_image", "dimension"],
                              values="rating", aggfunc="first")
        pair_stats = []
        raters = list(pivot.index)
        for r1, r2 in combinations(raters, 2):
            both = pivot.loc[[r1, r2]].dropna(axis=1, how="any")
            if both.shape[1] < min_overlap:
                continue
            a, b = both.loc[r1].to_numpy(), both.loc[r2].to_numpy()
            rho, _ = jstats.spearman(a, b)
            pair_stats.append({
                "n_shared": both.shape[1],
                "mae": jstats.mae(a, b),
                "kappa": jstats.quadratic_weighted_kappa(a, b, SCALE_MIN, SCALE_MAX),
                "icc": jstats.icc2_1(np.column_stack([a, b])),
                "spearman": rho,
            })
        if not pair_stats:
            continue
        ps = pd.DataFrame(pair_stats)
        rows.append({
            "dimension": dim, "n_pairs": len(ps),
            "mean_shared_items": ps["n_shared"].mean(),
            "mae": ps["mae"].mean(),
            "weighted_kappa": ps["kappa"].mean(skipna=True),
            "icc2_1": ps["icc"].mean(skipna=True),
            "spearman_rho": ps["spearman"].mean(skipna=True),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Interpretation (Colab-style decision criteria)
# --------------------------------------------------------------------------- #
def interpret(metrics: pd.DataFrame, baseline: Optional[pd.DataFrame],
              margin: float, fraction: float = 0.8) -> str:
    lines = ["Decision criteria per condition (pooled over dimensions)",
             "=" * 60]
    base = None
    if baseline is not None and len(baseline):
        b = baseline[baseline["dimension"] == "ALL"]
        if len(b):
            base = b.iloc[0]
            lines.append(
                "Human-human baseline (anchor cars, mean over rater pairs): "
                f"kappa={base['weighted_kappa']:.3f} ICC={base['icc2_1']:.3f} "
                f"rho={base['spearman_rho']:.3f} MAE={base['mae']:.3f}")
    if base is None:
        lines.append("No human-human baseline available (--data not given or "
                     "no multiply-rated cars); reporting absolute values only.")
    lines.append("")

    pooled = metrics[metrics["dimension"] == "ALL"].sort_values(GROUP_COLS)
    for _, r in pooled.iterrows():
        name = (f"{r['source']} | {r['judge']} | {r['holdout_mode']} | "
                f"context={int(r['context_size'])}")
        lines.append(name)
        lines.append(f"  n={int(r['n'])}  MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  "
                     f"within1={r['within1']:.3f}  kappa={r['weighted_kappa']:.3f}  "
                     f"ICC={r['icc2_1']:.3f}  rho={r['spearman_rho']:.3f}")
        eq = "PASS" if r["tost_equivalent"] == 1.0 else "fail"
        lines.append(f"  TOST equivalence (+/-{margin}): {eq} "
                     f"(p_lower={r['tost_p_lower']:.4f}, p_upper={r['tost_p_upper']:.4f})")
        if base is not None:
            met, missed = [], []
            checks = [
                ("kappa >= {:.0%} of human".format(fraction),
                 r["weighted_kappa"] >= fraction * base["weighted_kappa"]),
                ("ICC >= {:.0%} of human".format(fraction),
                 r["icc2_1"] >= fraction * base["icc2_1"]),
                ("rho >= {:.0%} of human".format(fraction),
                 r["spearman_rho"] >= fraction * base["spearman_rho"]),
                ("MAE <= human MAE", r["mae"] <= base["mae"]),
                ("TOST equivalent", r["tost_equivalent"] == 1.0),
            ]
            for label, ok in checks:
                (met if bool(ok) else missed).append(label)
            lines.append(f"  criteria met:   {met or '-'}")
            lines.append(f"  criteria NOT met: {missed or '-'}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _condition_label(src: str, hold: str, multi_src: bool) -> str:
    return f"{src} ({hold})" if multi_src else hold


def plot_metric_vs_context(metrics: pd.DataFrame, metric: str, ylabel: str,
                           out_path: str, baseline_val: Optional[float] = None,
                           baseline_label: str = "human–human") -> None:
    conds = list(metrics.groupby(["source", "holdout_mode"]).groups)
    multi_src = metrics["source"].nunique() > 1
    fig, axes = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 4.0),
                             squeeze=False, facecolor=SURFACE)
    for ax, (src, hold) in zip(axes[0], conds):
        m = metrics[(metrics["source"] == src) & (metrics["holdout_mode"] == hold)]
        pooled = m[m["dimension"] == "ALL"].sort_values("context_size")
        ax.plot(pooled["context_size"], pooled[metric], color=INK, linewidth=2,
                marker="o", markersize=6, label="all dimensions", zorder=3)
        lo, hi = metric + "_ci_lo", metric + "_ci_hi"
        if lo in pooled.columns and pooled[lo].notna().any():
            ax.fill_between(pooled["context_size"], pooled[lo], pooled[hi],
                            color=INK, alpha=0.10, linewidth=0, zorder=1)
        for dim in DIMENSIONS:
            md = m[m["dimension"] == dim].sort_values("context_size")
            if md.empty:
                continue
            ax.plot(md["context_size"], md[metric], color=DIM_COLORS[dim],
                    linewidth=1.4, marker="o", markersize=4, alpha=0.9,
                    zorder=2, label=dim)
        if baseline_val is not None and np.isfinite(baseline_val):
            ax.axhline(baseline_val, color=MUTED, linewidth=1.2, linestyle="--")
            ax.annotate(baseline_label, (0.98, baseline_val), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points",
                        fontsize=8, color=MUTED, ha="right")
        ax.set_xlabel("context size (cars shown as examples)")
        ax.set_ylabel(ylabel)
        ax.set_title(_condition_label(src, hold, multi_src), fontsize=10)
        ax.set_xticks(sorted(m["context_size"].unique()))
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2,
                  loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_bland_altman(df: pd.DataFrame, out_dir: str) -> List[str]:
    paths = []
    rng = np.random.RandomState(0)
    for keys, g in df.groupby(GROUP_COLS):
        src, judge, hold, n = keys
        ref = g["reference"].to_numpy(dtype=float)
        mod = g["predicted"].to_numpy(dtype=float)
        ba = jstats.bland_altman(ref, mod)
        mean_ratings = 0.5 * (ref + mod)
        diff = ref - mod
        jx = rng.uniform(-0.12, 0.12, len(diff))  # integer grid -> jitter
        jy = rng.uniform(-0.12, 0.12, len(diff))

        fig, ax = plt.subplots(figsize=(5.6, 4.2), facecolor=SURFACE)
        ax.scatter(mean_ratings + jx, diff + jy, s=18, color=SCATTER,
                   alpha=0.35, linewidths=0, zorder=2)
        ax.axhline(ba["mean_diff"], color=INK, linewidth=1.4, linestyle="--", zorder=3)
        for y, lbl in ((ba["upper_loa"], "+1.96 SD"), (ba["lower_loa"], "−1.96 SD")):
            ax.axhline(y, color=MUTED, linewidth=1.0, linestyle=":", zorder=3)
            ax.annotate(lbl, (0.99, y), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points",
                        fontsize=8, color=MUTED, ha="right")
        ax.annotate(f"mean diff = {ba['mean_diff']:.2f}", (0.99, ba["mean_diff"]),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", fontsize=8, color=INK_2, ha="right")
        ax.set_xlabel("(reference + prediction) / 2")
        ax.set_ylabel("reference − prediction")
        ax.set_title(f"Bland–Altman — {src} | {judge} | {hold} | context={n}",
                     fontsize=10)
        _style_axes(ax)
        fig.tight_layout()
        path = os.path.join(out_dir, f"bland_altman_{src}_{hold}_n{n}.png")
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Statistical evaluation of AI-judge predictions.")
    p.add_argument("--predictions", nargs="+", required=True,
                   help="predictions CSV(s); several files = several models")
    p.add_argument("--reference", choices=["rater", "car_mean"], default="rater")
    p.add_argument("--data", default=None,
                   help="raw Qualtrics export; required for car_mean and for "
                        "the human-human baseline")
    p.add_argument("--exclude-self", action="store_true",
                   help="car_mean: leave the context rater out of the average")
    p.add_argument("--min-raters", type=int, default=2,
                   help="car_mean: min raters per car for a usable reference")
    p.add_argument("--equivalence-margin", type=float, default=1.0,
                   help="TOST bounds in rating points (1-6 scale)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--baseline-fraction", type=float, default=0.8,
                   help="'ballpark' fraction of the human-human baseline an "
                        "agreement metric must reach (Colab used 0.8)")
    p.add_argument("--bootstrap", action="store_true",
                   help="add percentile-bootstrap CIs for MAE and kappa")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--out-dir", default="evaluation/outputs/run")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    os.makedirs(a.out_dir, exist_ok=True)

    df = load_predictions(a.predictions)
    human = ratings_long(a.data) if a.data else None
    df = add_reference(df, a.reference, human, a.exclude_self, a.min_raters)

    print("[eval] computing metrics table ...")
    metrics = metrics_table(df, a.equivalence_margin, a.alpha,
                            a.bootstrap, a.n_boot)
    metrics_path = os.path.join(a.out_dir, "metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    print("[eval] paired comparisons ...")
    comps = comparisons_table(df)
    comps_path = os.path.join(a.out_dir, "comparisons.csv")
    comps.to_csv(comps_path, index=False)

    baseline = None
    if human is not None:
        multi = human.groupby(["target_image", "dimension"])["rater"].transform("nunique")
        shared = human[multi >= 2]
        if len(shared):
            print(f"[eval] human-human baseline from "
                  f"{shared['target_image'].nunique()} multiply-rated cars ...")
            baseline = human_baseline(shared)
            baseline.to_csv(os.path.join(a.out_dir, "baseline.csv"), index=False)
        else:
            print("[eval] no multiply-rated cars found; skipping baseline")

    text = interpret(metrics, baseline, a.equivalence_margin, a.baseline_fraction)
    with open(os.path.join(a.out_dir, "interpretation.txt"), "w") as fh:
        fh.write(text + "\n")

    if not a.no_plots:
        plot_dir = os.path.join(a.out_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        base_all = None
        if baseline is not None and len(baseline):
            b = baseline[baseline["dimension"] == "ALL"]
            base_all = b.iloc[0] if len(b) else None
        specs = [
            ("mae", "mean absolute error (rating points)",
             None if base_all is None else float(base_all["mae"])),
            ("weighted_kappa", "quadratic weighted kappa",
             None if base_all is None else float(base_all["weighted_kappa"])),
            ("icc2_1", "ICC(2,1)",
             None if base_all is None else float(base_all["icc2_1"])),
            ("spearman_rho", "Spearman rho",
             None if base_all is None else float(base_all["spearman_rho"])),
        ]
        for metric, ylabel, base_val in specs:
            plot_metric_vs_context(
                metrics, metric, ylabel,
                os.path.join(plot_dir, f"{metric}_vs_context.png"),
                baseline_val=base_val)
        plot_bland_altman(df, plot_dir)
        print(f"[save] plots -> {plot_dir}")

    print(f"[save] {metrics_path}")
    print(f"[save] {comps_path}")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
