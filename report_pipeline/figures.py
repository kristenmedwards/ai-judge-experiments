"""Report figures — matched to the report_and_figures house style.

Vertical bars; the three baselines use a light->dark neutral grey ramp
(pop_mean lightest -> car_mean darkest) and the two judges are blue / orange,
so a glance separates "naive references" from "the judges". Colour follows the
entity across every figure. Off-white surface, recessive horizontal grid, direct
value labels, bold weight on the judge bars. A constant predictor's ranking
metrics (pop_mean's ICC / Spearman within a dimension) are undefined and drawn
as "n/a".
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .data import DIMS

JUDGE_ORDER = ["pop_mean", "b_nocontext", "car_mean", "b_persona", "g_card_twostage"]
BOLD = {"b_persona", "g_card_twostage"}
COLORS = {
    "pop_mean":        "#D3CDC1",   # light warm grey  (naive floor)
    "b_nocontext":     "#A9A49B",   # mid grey
    "car_mean":        "#6F6A63",   # dark grey
    "b_persona":       "#2E7FD6",   # blue  (recommended judge)
    "g_card_twostage": "#E7622B",   # orange
}
XLAB = {
    "pop_mean": "Population\nmean", "b_nocontext": "No-context\n(0-shot)",
    "car_mean": "Per-car\nmean", "b_persona": "b_persona", "g_card_twostage": "g_card_\ntwostage",
}
LEGLAB = {
    "pop_mean": "Population mean", "b_nocontext": "No-context", "car_mean": "Per-car mean",
    "b_persona": "b_persona", "g_card_twostage": "g_card_twostage",
}
SURFACE, INK, MUTED, GRID = "#FBFAF7", "#1D1D1B", "#8A857C", "#E6E3DD"


def _order(js) -> List[str]:
    js = list(js)
    return [j for j in JUDGE_ORDER if j in js] + [j for j in js if j not in JUDGE_ORDER]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=10)
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)


def _bold_ticklabels(ax, judges):
    for lab, j in zip(ax.get_xticklabels(), judges):
        if j in BOLD:
            lab.set_fontweight("bold")


# --------------------------------------------------------------------------- #
def _single_panel(ax, judges, vals, title, higher_better, fmt="{:.3f}", ymax=None,
                  errs=None):
    x = np.arange(len(judges))
    for i, (xi, j, v) in enumerate(zip(x, judges, vals)):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            ax.text(xi, (ymax or 1) * 0.02, "n/a", ha="center", va="bottom",
                    fontsize=8, color=MUTED, rotation=90)
            continue
        e = None if errs is None else errs[i]
        if e is not None and (not np.isfinite(e) or e == 0):
            e = None
        ax.bar(xi, v, width=0.72, color=COLORS.get(j, "#555"), zorder=3,
               yerr=e, capsize=3, error_kw=dict(lw=1.1, ecolor="#3a3a38", zorder=4))
        ax.text(xi, v + (e or 0), fmt.format(v), ha="center", va="bottom",
                fontsize=9.5, color=INK,
                fontweight=("bold" if j in BOLD else "normal"))
    ax.set_xticks(x); ax.set_xticklabels([XLAB[j] for j in judges], fontsize=9)
    _bold_ticklabels(ax, judges)
    arrow = "higher is better ↑" if higher_better else "lower is better ↓"
    ax.set_title(f"{title}  ({arrow})", fontsize=12.5, loc="left", color=INK, pad=10)
    _style(ax)
    top = ymax if ymax else np.nanmax([v for v in vals if v is not None]) * 1.18
    ax.set_ylim(0, top)


def fig_value_ladder(overall: pd.DataFrame, out: str) -> None:
    d = overall.set_index("judge")
    judges = _order(d.index)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2)); fig.patch.set_facecolor(SURFACE)
    _single_panel(axes[0], judges, [d.loc[j, "mae"] for j in judges],
                  "Overall MAE", higher_better=False)
    _single_panel(axes[1], judges, [d.loc[j, "icc"] for j in judges],
                  "Overall ICC(2,1) agreement", higher_better=True, ymax=0.78)
    _single_panel(axes[2], judges, [d.loc[j, "rho"] for j in judges],
                  "Overall Spearman ρ (rank)", higher_better=True, ymax=0.78)
    fig.suptitle("The exemplars are the whole story: error, agreement and rank from "
                 "naive baseline to best judge", fontsize=15, fontweight="bold",
                 x=0.012, ha="left", color=INK)
    fig.text(0.012, 0.005,
             "Full pool, 672 raters × 20 held-out cars × 5 dimensions, seed-averaged "
             "(seeds 0,1). car_mean = leave-one-out consensus. Population mean is worse "
             "than the 0-shot VLM on MAE and near-zero on rank (ICC 0.14, ρ 0.26).",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94]); fig.savefig(out, dpi=150); plt.close(fig)


def _grouped(perdim: pd.DataFrame, metric: str, title: str, higher_better: bool,
             out: str, drop_constant: bool = False, caption: str = "",
             err: Optional[pd.DataFrame] = None) -> None:
    judges = _order(perdim["judge"].unique())
    if drop_constant:
        judges = [j for j in judges if j != "pop_mean"]
    fig, ax = plt.subplots(figsize=(11.5, 5.0)); fig.patch.set_facecolor(SURFACE)
    w = 0.82 / len(judges); x = np.arange(len(DIMS))
    for i, j in enumerate(judges):
        sub = perdim[perdim["judge"] == j].set_index("scope").reindex(DIMS)
        xs = x + i * w - 0.41 + w / 2
        yerr = None
        if err is not None:
            esub = err[err["judge"] == j].set_index("scope").reindex(DIMS)
            yerr = esub[metric].fillna(0).to_numpy() if metric in esub else None
        ax.bar(xs, sub[metric], width=w * 0.92, color=COLORS.get(j, "#555"), zorder=3,
               yerr=yerr, capsize=2, error_kw=dict(lw=0.9, ecolor="#3a3a38", zorder=4))
        for xi, v in zip(xs, sub[metric]):
            if isinstance(v, float) and np.isnan(v):
                ax.text(xi, 0.006, "n/a", ha="center", va="bottom", fontsize=6.5,
                        color=MUTED, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in DIMS], fontsize=11)
    arrow = "higher is better ↑" if higher_better else "lower is better ↓"
    ax.set_ylabel(f"{metric_label(metric)}  ({arrow})", fontsize=11, color=INK)
    ax.set_title(title, fontsize=13.5, loc="left", fontweight="bold", color=INK, pad=8)
    _style(ax)
    handles = [Patch(facecolor=COLORS[j], label=LEGLAB[j]) for j in judges]
    ax.legend(handles=handles, fontsize=9.5, frameon=False, ncol=len(judges),
              loc="upper center", bbox_to_anchor=(0.5, -0.09))
    if caption:
        fig.text(0.012, 0.005, caption, fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 1]); fig.savefig(out, dpi=150); plt.close(fig)


def metric_label(m: str) -> str:
    return {"mae": "MAE", "rmse": "RMSE", "icc": "ICC(2,1)", "rho": "Spearman ρ",
            "within1": "within-1", "within_rater_rho": "Within-rater ρ"}.get(m, m)


def fig_mae_by_dimension(perdim, out):
    _grouped(perdim, "mae", "MAE by dimension — every judge",
             higher_better=False, out=out,
             caption="Rugged is best-predicted for the exemplar judges; preference "
                     "(holistic) is hardest for every method.")


def fig_icc_by_dimension(perdim, out, err=None):
    _grouped(perdim, "icc", "ICC(2,1) agreement by dimension", higher_better=True,
             out=out, drop_constant=True, err=err,
             caption="Absolute-agreement ICC(2,1), judge-vs-human. pop_mean is constant "
                     "within a dimension, so its ICC is undefined and omitted.")


def fig_spearman_by_dimension(perdim, out, err=None):
    _grouped(perdim, "rho", "Spearman rank ρ by dimension", higher_better=True,
             out=out, drop_constant=True, err=err,
             caption="Rank correlation judge-vs-human. pop_mean is constant within a "
                     "dimension (undefined ρ) and omitted; context lifts ρ most on "
                     "rugged and preference.")


def fig_finalist_perdim(perdim, a, b, out):
    fig, ax = plt.subplots(figsize=(10.5, 5.0)); fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(DIMS)); w = 0.38
    for i, j in enumerate([a, b]):
        sub = perdim[perdim["judge"] == j].set_index("scope").reindex(DIMS)
        xs = x + (i - 0.5) * w
        ax.bar(xs, sub["mae"], width=w * 0.9, color=COLORS.get(j, "#555"),
               label=LEGLAB.get(j, j), zorder=3)
        for xi, v in zip(xs, sub["mae"]):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in DIMS], fontsize=11)
    ax.set_ylabel("MAE  (lower is better ↓)", fontsize=11, color=INK)
    ax.set_title(f"Finalist per-dimension MAE: {a} vs {b}", fontsize=13.5, loc="left",
                 fontweight="bold", color=INK, pad=8)
    _style(ax); ax.legend(fontsize=10, frameon=False)
    fig.text(0.012, 0.005, "The two-stage gain concentrates in preference; the two are "
             "within noise elsewhere.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.04, 1, 1]); fig.savefig(out, dpi=150); plt.close(fig)


def fig_dim_full(metrics: pd.DataFrame, dim: str, out: str,
                 title: Optional[str] = None,
                 err: Optional[pd.DataFrame] = None) -> None:
    d = metrics[metrics["scope"] == dim].set_index("judge")
    e = (err[err["scope"] == dim].set_index("judge")
         if err is not None else None)

    def _yerr(j, cols):
        if e is None or j not in e.index:
            return None
        return [v if np.isfinite(v) else 0
                for v in (float(e.loc[j, m]) if m in e.columns else 0 for m in cols)]

    judges = _order(d.index)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.0, 5.4),
                                   gridspec_kw={"width_ratios": [2, 3]})
    fig.patch.set_facecolor(SURFACE)

    # left: error (MAE, RMSE)
    errcols = ["mae", "rmse"]; x = np.arange(len(errcols)); w = 0.82 / len(judges)
    for i, j in enumerate(judges):
        axL.bar(x + i * w - 0.41 + w / 2, [d.loc[j, m] for m in errcols],
                width=w * 0.9, color=COLORS.get(j, "#555"), zorder=3,
                yerr=_yerr(j, errcols), capsize=2,
                error_kw=dict(lw=0.9, ecolor="#3a3a38", zorder=4))
    axL.set_xticks(x); axL.set_xticklabels(["MAE", "RMSE"], fontsize=11)
    axL.set_title("Error  (lower is better ↓)", fontsize=12.5, loc="left", color=INK, pad=8)
    _style(axL)

    # right: agreement / ranking (ICC, Spearman rho, within-rater rho)
    agr = ["icc", "rho", "within_rater_rho"]; xr = np.arange(len(agr))
    for i, j in enumerate(judges):
        xs = xr + i * w - 0.41 + w / 2
        vals = [d.loc[j, m] for m in agr]
        axR.bar(xs, [0 if (isinstance(v, float) and np.isnan(v)) else v for v in vals],
                width=w * 0.9, color=COLORS.get(j, "#555"), zorder=3,
                yerr=_yerr(j, agr), capsize=2,
                error_kw=dict(lw=0.9, ecolor="#3a3a38", zorder=4))
        for xi, v in zip(xs, vals):
            if isinstance(v, float) and np.isnan(v):
                axR.text(xi, 0.01, "n/a", ha="center", va="bottom", fontsize=7,
                         color=MUTED, rotation=90)
    axR.set_xticks(xr); axR.set_xticklabels(["ICC(2,1)", "Spearman ρ", "Within-rater ρ"],
                                            fontsize=11)
    axR.set_title("Agreement / ranking  (higher is better ↑)", fontsize=12.5,
                  loc="left", color=INK, pad=8)
    _style(axR)

    handles = [Patch(facecolor=COLORS[j], label=LEGLAB[j]) for j in judges]
    axR.legend(handles=handles, fontsize=9.5, frameon=False, ncol=len(judges),
               loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.suptitle(title or f"{dim.capitalize()} — full metric suite across judges",
                 fontsize=14.5, fontweight="bold", x=0.012, ha="left", color=INK)
    fig.text(0.012, 0.004, "Population-mean baseline is constant within a dimension, so "
             "its ranking metrics (ICC, ρ) are undefined (\"n/a\").",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94]); fig.savefig(out, dpi=150); plt.close(fig)


def fig_forest_dmae(tests: pd.DataFrame, out: str) -> None:
    d = tests.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.0, 0.6 * len(d) + 1.5)); fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(d))
    for yi, r in zip(y, d.itertuples()):
        c = "#2E7FD6" if r.ci_hi < 0 else ("#E7622B" if r.ci_lo > 0 else MUTED)
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=c, lw=2.4, zorder=2)
        ax.plot(r.dmae, yi, "o", color=c, ms=7.5, zorder=3)
        ax.text(r.ci_hi + 0.006, yi, f"{r.dmae:+.3f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK, lw=1, ls="--", zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([f"{a}  vs  {b}" for a, b in zip(d["A"], d["B"])],
                                         fontsize=9.5)
    ax.set_xlabel("Δ MAE  (A − B; negative ⇒ A better)", fontsize=11, color=INK)
    ax.set_title("Paired accuracy differences (rater-cluster bootstrap 95% CI)",
                 fontsize=13, loc="left", fontweight="bold", color=INK, pad=8)
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK); ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def make_all(metrics: pd.DataFrame, tests: pd.DataFrame, outdir: str) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    overall = metrics[metrics["scope"] == "overall"].copy()
    perdim = metrics[metrics["scope"].isin(DIMS)].copy()
    written: List[str] = []

    def save(fn, fn_call):
        p = os.path.join(outdir, fn); fn_call(p); written.append(p)

    save("fig1_value_ladder.png", lambda p: fig_value_ladder(overall, p))
    save("fig2_mae_by_dimension.png", lambda p: fig_mae_by_dimension(perdim, p))
    save("fig3_icc_by_dimension.png", lambda p: fig_icc_by_dimension(perdim, p))
    save("fig4_spearman_by_dimension.png", lambda p: fig_spearman_by_dimension(perdim, p))
    if len(tests):
        save("fig8_forest_dmae.png", lambda p: fig_forest_dmae(tests, p))
    if {"b_persona", "g_card_twostage"}.issubset(set(perdim["judge"])):
        save("fig9_finalist_perdim.png",
             lambda p: fig_finalist_perdim(perdim, "b_persona", "g_card_twostage", p))
    for dim in DIMS:
        save(f"fig_dim_{dim}.png", lambda p, dim=dim: fig_dim_full(metrics, dim, p))
    return written
