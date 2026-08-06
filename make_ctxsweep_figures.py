"""Figures for the context-size sweep (run 20260731_134005_ctxsweep).

Three PNGs into outputs/figures/:
  ctx_mae_vs_context.png        MAE vs N (per-seed + seed mean), marginal gain
                                annotated per step, image cost on the x labels
  ctx_agreement_vs_context.png  ICC(2,1) and Spearman rho vs N
  ctx_bias_vs_context.png       signed bias (pred - human) per dimension vs N

Usage:
  python make_ctxsweep_figures.py [--run outputs/runs/20260731_134005_ctxsweep]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Reference dataviz palette (validated adjacent-pair order, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

DIM_ORDER = ["sporty", "luxurious", "modern", "rugged", "preference"]
# fixed dimension colors, shared with the report figures
DIM_COLORS = {"sporty": "#2a78d6", "luxurious": "#008300", "modern": "#e87ba4",
              "rugged": "#eda100", "preference": "#4a3aa7"}


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)


def load_run(run_dir):
    pe = os.path.join(run_dir, "per_experiment")
    met_rows, pred_frames = [], []
    for mp in glob.glob(os.path.join(pe, "*.metrics.json")):
        with open(mp, encoding="utf-8") as fh:
            m = json.load(fh)
        met_rows.append({"n_ctx": m["n_context"], "seed": m["split_seed"],
                         "mae": m["mae"], "icc": m["icc2_1"],
                         "rho": m["spearman_rho"]})
    for cp in glob.glob(os.path.join(pe, "*.csv")):
        df = pd.read_csv(cp, usecols=["dimension", "predicted", "true", "n_context"])
        df["seed"] = int(re.search(r"_s(\d+)\.csv$", cp).group(1))
        pred_frames.append(df)
    met = pd.DataFrame(met_rows).sort_values(["n_ctx", "seed"])
    preds = pd.concat(pred_frames, ignore_index=True)
    preds["bias"] = preds["predicted"] - preds["true"]
    return met, preds


def fig_mae(met, out):
    sizes = sorted(met["n_ctx"].unique())
    mean = met.groupby("n_ctx")["mae"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)

    for seed, grp in met.groupby("seed"):
        g = grp.sort_values("n_ctx")
        ax.plot(g["n_ctx"], g["mae"], color=BASELINE, linewidth=1.2, zorder=2)
    ax.plot(sizes, mean[sizes], color=SERIES[0], linewidth=2, zorder=3,
            marker="o", markersize=6, markerfacecolor=SERIES[0],
            markeredgecolor=SURFACE, markeredgewidth=1.5)

    # marginal gain per step, annotated below the arrival point (the curve
    # descends, so above-the-point labels collide with the incoming segment)
    for a, b in zip(sizes, sizes[1:]):
        d = mean[b] - mean[a]
        ax.annotate(f"{d:+.3f}", (b, mean[b]), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8.5, color=INK2)
    ax.annotate("seed mean", (sizes[2], mean[sizes[2]]),
                textcoords="offset points", xytext=(14, 10), fontsize=9,
                color=SERIES[0], fontweight="bold")
    ax.annotate("individual seeds", (sizes[0], met[met["n_ctx"] == sizes[0]]["mae"].min()),
                textcoords="offset points", xytext=(10, -16), fontsize=8.5,
                color=MUTED)

    ax.set_xticks(sizes)
    ax.set_xlim(-1.2, max(sizes) + 1.8)
    ax.margins(y=0.12)
    ax.set_xticklabels([f"{n}\n({n + 1} img/call)" for n in sizes])
    ax.set_xlabel("context size N  (images per judge call)", color=INK2, fontsize=10)
    ax.set_ylabel("MAE (1–7 scale)", color=INK2, fontsize=10)
    ax.set_title("Held-out MAE vs context size — marginal gain per step",
                 color=INK, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_agreement(met, out):
    sizes = sorted(met["n_ctx"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)

    # the two curves end nearly on top of each other — stagger the end labels
    for col, color, label, dy in [("icc", SERIES[0], "ICC(2,1)", 8),
                                  ("rho", SERIES[1], "Spearman ρ", -12)]:
        mean = met.groupby("n_ctx")[col].mean()
        for seed, grp in met.groupby("seed"):
            g = grp.sort_values("n_ctx")
            ax.plot(g["n_ctx"], g[col], color=color, alpha=0.25, linewidth=1)
        ax.plot(sizes, mean[sizes], color=color, linewidth=2, marker="o",
                markersize=6, markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
        ax.annotate(label, (sizes[-1], mean[sizes[-1]]),
                    textcoords="offset points", xytext=(8, dy),
                    fontsize=9.5, color=color, fontweight="bold")

    ax.set_xticks(sizes)
    ax.set_xlim(-0.8, max(sizes) + 6.5)
    ax.set_xlabel("context size N", color=INK2, fontsize=10)
    ax.set_ylabel("agreement with human ratings", color=INK2, fontsize=10)
    ax.set_title("Judge–human agreement vs context size", color=INK,
                 fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_bias(preds, out):
    sizes = sorted(preds["n_context"].unique())
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax)
    ax.axhline(0, color=BASELINE, linewidth=1.2, zorder=2)

    # modern and luxurious end within ~0.005 of each other — stagger their labels
    label_dy = {"sporty": -3, "luxurious": -14, "modern": 6,
                "rugged": -3, "preference": -3}
    for dim in DIM_ORDER:
        color = DIM_COLORS[dim]
        mean = preds[preds["dimension"] == dim].groupby("n_context")["bias"].mean()
        ls = "--" if dim == "preference" else "-"
        ax.plot(sizes, mean[sizes], color=color, linewidth=2, linestyle=ls,
                marker="o", markersize=6, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=dim)
        ax.annotate(dim, (sizes[-1], mean[sizes[-1]]),
                    textcoords="offset points", xytext=(10, label_dy[dim]),
                    fontsize=9.5, color=color, fontweight="bold")

    ax.set_xticks(sizes)
    ax.set_xlim(-0.8, max(sizes) + 6.0)
    ax.set_xlabel("context size N", color=INK2, fontsize=10)
    ax.set_ylabel("bias: mean(predicted − human)", color=INK2, fontsize=10)
    ax.set_title("Judge bias vs context size, by dimension", color=INK,
                 fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="outputs/runs/20260731_134005_ctxsweep")
    ap.add_argument("--out-dir", default="outputs/figures")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    met, preds = load_run(a.run)
    fig_mae(met, os.path.join(a.out_dir, "ctx_mae_vs_context.png"))
    fig_agreement(met, os.path.join(a.out_dir, "ctx_agreement_vs_context.png"))
    fig_bias(preds, os.path.join(a.out_dir, "ctx_bias_vs_context.png"))
    print(f"wrote 3 figures to {a.out_dir}")


if __name__ == "__main__":
    main()
