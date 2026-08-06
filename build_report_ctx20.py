"""Report: the N=20 full-context judge (ctx20) vs baselines and the fp judges.

Same machinery and house style as build_report.py, with two deliberate changes:
  * g_card_twostage is dropped; ctx20 takes its slot (and its orange).
  * ctx20 lives on a DIFFERENT evaluation design than the fp runs
    (666 raters x 14 held-out cars, anchors in the context pool, persona prompt,
    N=20 random context; vs the fp runs' 672 raters x 20 cars). So:
      - pop_mean / car_mean in the metrics table are synthesized on ctx20's OWN
        cells (exact, fair, free);
      - the paired ctx20-vs-b_nocontext test uses the strict per-seed
        (rater, car, dimension) cell intersection of the two designs — b_nocontext
        was run on all raters, so every ctx20 rater is covered (anchor cars,
        absent from the fp pool, drop out);
      - fp-judge rows (b_persona, b_nocontext) keep their canonical own-design
        numbers and are compared side-by-side, with fp-cell baselines used for
        the fp-internal paired tests, as in build_report.py.
    Every paired test is merged PER SEED (never across), then pooled for the
    rater-cluster bootstrap.

Writes into --out (default report_ctx20/):
  metrics_by_dimension_seedavg.csv    ctx20 + fp judges + ctx-cell baselines
  statistical_tests_seedavg.csv       paired dMAE rows, each tagged with its
                                      cell domain ("cells") and seeds
  provenance.json                     inputs + recomputation check
  figures/*.png                       house-style figure set

Usage:
  python build_report_ctx20.py --check
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from report_pipeline import data as D
from report_pipeline import baselines as B
from report_pipeline import metrics as M
from report_pipeline import figures as F
from build_report import find_human, crosscheck_results_tsv

FP_JUDGES = ["b_persona", "b_nocontext"]          # g_card_twostage: dropped
CTX_GLOB = os.path.join("outputs", "runs", "*_ctx20_full",
                        "per_experiment", "ctx20_s*.csv")

# ---- house style: ctx20 inherits the orange slot g_card_twostage vacated ----
F.JUDGE_ORDER = ["pop_mean", "b_nocontext", "car_mean", "b_persona", "ctx20"]
F.BOLD = {"b_persona", "ctx20"}
F.COLORS = {**F.COLORS, "ctx20": "#E7622B"}
F.XLAB = {**F.XLAB, "b_persona": "N=12\npersona", "ctx20": "N=20\npersona"}
F.LEGLAB = {**F.LEGLAB, "b_persona": "N=12 context, persona prompt",
            "ctx20": "N=20 context, persona prompt"}
# short display names for the forest plot's row labels
FOREST_LAB = {"ctx20": "N=20 persona", "b_persona": "N=12 persona",
              "b_nocontext": "No-context", "car_mean": "Per-car mean",
              "pop_mean": "Population mean"}


def discover_ctx_runs() -> dict:
    """seed -> prediction csv path for the ctx20 full runs (latest dir wins)."""
    found = {}
    for p in sorted(glob.glob(CTX_GLOB)):
        m = re.search(r"ctx20_s(\d+)\.csv$", os.path.basename(p))
        if m:
            found[int(m.group(1))] = p       # sorted() => later timestamp wins
    if not found:
        raise FileNotFoundError(f"no ctx20 runs match {CTX_GLOB}")
    return found


def paired_dmae_seeded(framesA: dict, framesB: dict, n_boot: int = 2000,
                       rng_seed: int = 0) -> dict:
    """Paired dMAE over the per-seed cell intersections of two judges.

    framesA/framesB: seed -> prediction frame. Cells are merged WITHIN each
    common seed (never across seeds), then pooled; the CI resamples raters
    (each rater is one cluster across all their seeds/cells).
    Negative dmae => A more accurate than B. Mirrors metrics.paired_dmae.
    """
    key = ["rater", "target_image", "dimension"]
    seeds = sorted(set(framesA) & set(framesB))
    parts = []
    for s in seeds:
        a = framesA[s][key + ["true", "predicted"]].rename(columns={"predicted": "pA"})
        b = framesB[s][key + ["predicted"]].rename(columns={"predicted": "pB"})
        parts.append(a.merge(b, on=key, how="inner"))
    m = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not len(m):
        return {"dmae": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p": float("nan"), "n_cells": 0, "seeds": seeds}
    m["diff"] = (m["pA"] - m["true"]).abs() - (m["pB"] - m["true"]).abs()
    dmae = float(m["diff"].mean())

    per_rater = m.groupby("rater")["diff"].agg(["sum", "count"])
    sums = per_rater["sum"].to_numpy(float)
    cnts = per_rater["count"].to_numpy(float)
    R = len(sums)
    rng = np.random.RandomState(rng_seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.randint(0, R, size=R)
        boots[i] = sums[idx].sum() / cnts[idx].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2.0 * float(np.mean(boots >= 0) if dmae < 0 else np.mean(boots <= 0))
    return {"dmae": dmae, "ci_lo": float(lo), "ci_hi": float(hi),
            "p": min(1.0, p), "n_cells": int(len(m)), "seeds": seeds}


def fig_value_ladder_ctx(overall: pd.DataFrame, out: str,
                         err: pd.DataFrame = None) -> None:
    """build_report's fig1, with title/caption corrected for this report."""
    import matplotlib.pyplot as plt
    d = overall.set_index("judge")
    judges = F._order(d.index)
    e = err.set_index("judge") if err is not None else None

    def errs(metric):
        if e is None:
            return None
        return [float(e.loc[j, metric]) if j in e.index else 0 for j in judges]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    fig.patch.set_facecolor(F.SURFACE)
    F._single_panel(axes[0], judges, [d.loc[j, "mae"] for j in judges],
                    "Overall MAE", higher_better=False, errs=errs("mae"))
    F._single_panel(axes[1], judges, [d.loc[j, "icc"] for j in judges],
                    "Overall ICC(2,1) agreement", higher_better=True, ymax=0.78,
                    errs=errs("icc"))
    F._single_panel(axes[2], judges, [d.loc[j, "rho"] for j in judges],
                    "Overall Spearman ρ (rank)", higher_better=True, ymax=0.78,
                    errs=errs("rho"))
    fig.suptitle("Twenty of the rater's own cars close most of the gap: error, "
                 "agreement and rank from naive baseline to full-context judge",
                 fontsize=15, fontweight="bold", x=0.012, ha="left", color=F.INK)
    caption = ("N=20: 666 raters × 14 held-out cars × 5 dimensions, seeds 0–2, "
               "persona prompt, random own-car exemplars; pop_mean / car_mean "
               "(leave-one-out consensus) scored on the same cells.\n"
               "No-context and N=12 persona judges: full-pool 672 raters × 20 cars "
               "(own design), side-by-side; the paired N=20 vs no-context test uses "
               "the strict per-seed cell intersection. Error bars: ±1 SD across seeds.")
    fig.text(0.012, 0.005, caption, fontsize=9, color=F.MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_forest_dmae_pos(tests: pd.DataFrame, out: str) -> None:
    """House-style forest plot, positive-oriented: improvement = B - A, so a
    better A plots to the RIGHT of zero. Blue = significant improvement."""
    import matplotlib.pyplot as plt
    d = tests.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.8, 0.6 * len(d) + 1.5))
    fig.patch.set_facecolor(F.SURFACE)
    y = np.arange(len(d))
    for yi, r in zip(y, d.itertuples()):
        imp, lo, hi = -r.dmae, -r.ci_hi, -r.ci_lo
        c = "#2E7FD6" if lo > 0 else ("#E7622B" if hi < 0 else F.MUTED)
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.4, zorder=2)
        ax.plot(imp, yi, "o", color=c, ms=7.5, zorder=3)
        ax.text(hi + 0.006, yi, f"{imp:+.3f}", va="center", fontsize=8.5, color=F.INK)
    ax.axvline(0, color=F.INK, lw=1, ls="--", zorder=1)
    ax.set_xlim(right=float((-d["ci_lo"]).max()) + 0.08)   # room for the labels
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a}  vs  {b}" for a, b in zip(d["A"], d["B"])], fontsize=9.5)
    ax.set_xlabel("MAE improvement  (B − A; positive ⇒ A better)",
                  fontsize=11, color=F.INK)
    ax.set_title("Paired accuracy improvements (rater-cluster bootstrap 95% CI)",
                 fontsize=13, loc="left", fontweight="bold", color=F.INK, pad=8)
    ax.set_facecolor(F.SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(F.MUTED)
    ax.tick_params(colors=F.INK)
    ax.xaxis.grid(True, color=F.GRID, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def make_figures(metrics: pd.DataFrame, tests: pd.DataFrame, figdir: str,
                 exclude: tuple = (), err: pd.DataFrame = None) -> None:
    """Full house-style figure set; `exclude` drops judges from every figure;
    `err` (same shape as metrics, seed SDs) draws ±1-SD-across-seeds error bars."""
    os.makedirs(figdir, exist_ok=True)
    m = metrics[~metrics["judge"].isin(exclude)].copy()
    t = tests[~(tests["A"].isin(exclude) | tests["B"].isin(exclude))].copy()
    e = err[~err["judge"].isin(exclude)].copy() if err is not None else None
    overall = m[m["scope"] == "overall"].copy()
    perdim = m[m["scope"].isin(D.DIMS)].copy()
    e_overall = e[e["scope"] == "overall"] if e is not None else None
    e_perdim = e[e["scope"].isin(D.DIMS)] if e is not None else None

    fig_value_ladder_ctx(overall, os.path.join(figdir, "fig1_value_ladder.png"),
                         err=e_overall)
    F._grouped(perdim, "mae", "MAE by dimension: every judge", higher_better=False,
               out=os.path.join(figdir, "fig2_mae_by_dimension.png"), err=e_perdim,
               caption="Rugged is best-predicted for the exemplar judges; preference "
                       "(holistic) is hardest for every method. Error bars: ±1 SD "
                       "across seeds.")
    F.fig_icc_by_dimension(perdim, os.path.join(figdir, "fig3_icc_by_dimension.png"),
                           err=e_perdim)
    F.fig_spearman_by_dimension(perdim,
                                os.path.join(figdir, "fig4_spearman_by_dimension.png"),
                                err=e_perdim)
    tf = t.copy()
    tf["A"] = tf["A"].map(lambda j: FOREST_LAB.get(j, j))
    tf["B"] = tf["B"].map(lambda j: FOREST_LAB.get(j, j))
    fig_forest_dmae_pos(tf, os.path.join(figdir, "fig8_forest_dmae.png"))
    for dim in D.DIMS:
        F.fig_dim_full(m, dim, os.path.join(figdir, f"fig_dim_{dim}.png"),
                       title=f"{dim.capitalize()}: full metric suite across judges",
                       err=e)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="outputs/runs")
    ap.add_argument("--human", default=None)
    ap.add_argument("--results", default="results.tsv")
    ap.add_argument("--out", default="report_ctx20")
    ap.add_argument("--car-mean-mode", choices=["loo", "include_self"], default="loo")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--check", action="store_true",
                    help="cross-check recomputed MAEs against results.tsv / metrics.json")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    human_csv = find_human(a.human)

    # ---- load predictions: fp judges (minus g_card) + ctx20 ----
    fp_runs = {k: v for k, v in D.discover_runs(a.runs_root).items()
               if k[0] in FP_JUDGES}
    ctx_paths = discover_ctx_runs()
    print(f"[data] fp runs: {sorted(fp_runs)}")
    print(f"[data] ctx20 runs: seeds {sorted(ctx_paths)}")
    print(f"[data] human ratings: {human_csv}")

    fp = {}                                     # judge -> {seed: frame}
    for (j, s), path in fp_runs.items():
        fp.setdefault(j, {})[s] = D.load_predictions(path)
    ctx = {s: D.load_predictions(p) for s, p in ctx_paths.items()}

    H = D.load_human_ratings(human_csv)
    pop = B.population_means(H)
    pools = B.car_rating_pools(H)
    include_self = (a.car_mean_mode == "include_self")

    def synth_baselines(cells):
        pm = B.predict_pop_mean(cells, pop)
        cm, fb = B.predict_car_mean_loo(cells, pools, pop, include_self=include_self)
        return (B.baseline_frame(cells, "pop_mean", pm),
                B.baseline_frame(cells, "car_mean", cm), fb)

    # baselines on ctx20's own cells (the metrics-table + ctx-test reference) …
    ctx_pop, ctx_car = {}, {}
    fallback_total = 0
    for s, df in ctx.items():
        pmf, cmf, fb = synth_baselines(D.cells_of(df))
        ctx_pop[s], ctx_car[s] = pmf, cmf
        fallback_total += fb
    # … and on the fp shared cells (for the fp-internal paired tests only)
    fp_pop, fp_car = {}, {}
    fp_seeds = sorted({s for j in fp.values() for s in j})
    for s in fp_seeds:
        ref = fp.get("b_persona", {}).get(s)
        if ref is None:
            ref = fp["b_nocontext"][s]
        pmf, cmf, fb = synth_baselines(D.cells_of(ref))
        fp_pop[s], fp_car[s] = pmf, cmf
        fallback_total += fb

    # ---- per-seed metric rows, seed-averaged ----
    per_seed = []
    recomputed = {}
    for s, df in ctx.items():
        per_seed.append(pd.DataFrame(M.metrics_for_judge(df, "ctx20")))
        per_seed.append(pd.DataFrame(M.metrics_for_judge(ctx_pop[s], "pop_mean")))
        per_seed.append(pd.DataFrame(M.metrics_for_judge(ctx_car[s], "car_mean")))
        recomputed[("ctx20", s)] = M.pooled_suite(df["true"], df["predicted"])["mae"]
    for j, frames in fp.items():
        for s, df in frames.items():
            per_seed.append(pd.DataFrame(M.metrics_for_judge(df, j)))
            recomputed[(j, s)] = M.pooled_suite(df["true"], df["predicted"])["mae"]

    metrics = M.seed_average(per_seed)
    cols = ["judge", "scope", "mae", "rmse", "within1", "icc", "rho", "wkappa",
            "within_rater_rho", "n"]
    metrics = metrics[[c for c in cols if c in metrics.columns]]
    metrics_path = os.path.join(a.out, "metrics_by_dimension_seedavg.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"[out ] {metrics_path}")

    # seed-to-seed SD of every metric (the figures' error bars)
    allrows = pd.concat(per_seed, ignore_index=True)
    numcols = [c for c in allrows.columns if c not in ("judge", "scope", "n")]
    spread = allrows.groupby(["judge", "scope"], sort=False)[numcols].std().reset_index()
    spread_path = os.path.join(a.out, "metrics_by_dimension_seed_sd.csv")
    spread.to_csv(spread_path, index=False)
    print(f"[out ] {spread_path}")

    # ---- paired tests: (A, B, framesA, framesB, cell-domain label) ----
    comparisons = [
        ("ctx20", "b_nocontext", ctx, fp["b_nocontext"], "ctx∩fp matched cells"),
        ("ctx20", "car_mean",    ctx, ctx_car,           "ctx20 cells"),
        ("ctx20", "pop_mean",    ctx, ctx_pop,           "ctx20 cells"),
        ("b_persona", "b_nocontext", fp.get("b_persona", {}), fp["b_nocontext"], "fp cells"),
        ("b_persona", "car_mean",    fp.get("b_persona", {}), fp_car, "fp cells"),
        ("b_persona", "pop_mean",    fp.get("b_persona", {}), fp_pop, "fp cells"),
        ("car_mean", "b_nocontext",  fp_car, fp["b_nocontext"], "fp cells"),
        ("car_mean", "pop_mean",     fp_car, fp_pop, "fp cells"),
        ("b_nocontext", "pop_mean",  fp["b_nocontext"], fp_pop, "fp cells"),
    ]
    trows = []
    for A_, B_, fA, fB, dom in comparisons:
        r = paired_dmae_seeded(fA, fB, n_boot=a.bootstrap)
        r["seeds"] = ",".join(map(str, r["seeds"]))
        trows.append({"A": A_, "B": B_, "cells": dom, **r})
    tests = pd.DataFrame(trows)
    tests_path = os.path.join(a.out, "statistical_tests_seedavg.csv")
    tests.to_csv(tests_path, index=False)
    print(f"[out ] {tests_path}")

    # ---- provenance + reproduction check ----
    check = {"checked": False, "reason": "run with --check"}
    if a.check:
        check = {"fp_vs_results_tsv": crosscheck_results_tsv(
                     {k: v for k, v in recomputed.items() if k[0] in FP_JUDGES},
                     a.results),
                 "ctx20_vs_run_metrics_json": []}
        for s, path in ctx_paths.items():
            mj = os.path.splitext(path)[0] + ".metrics.json"
            with open(mj, encoding="utf-8") as fh:
                ref = json.load(fh)["mae"]
            check["ctx20_vs_run_metrics_json"].append(
                {"seed": s, "recomputed": round(recomputed[("ctx20", s)], 6),
                 "metrics_json": round(ref, 6),
                 "abs_diff": round(abs(recomputed[("ctx20", s)] - ref), 8)})
    prov = {
        "inputs": {"fp_runs": {f"{j}_s{s}": os.path.abspath(p)
                               for (j, s), p in fp_runs.items()},
                   "ctx20_runs": {f"s{s}": os.path.abspath(p)
                                  for s, p in ctx_paths.items()},
                   "human_csv": os.path.abspath(human_csv),
                   "metric_code": "evaluation/stats.py"},
        "design_note": ("ctx20 = 666 raters x 14 held-out cars, anchors in pool, "
                        "persona prompt, N=20 random context, seeds 0-2. fp = 672 "
                        "raters x 20 cars. Metrics-table baselines are on ctx20 "
                        "cells; fp-internal tests use fp-cell baselines; ctx20 vs "
                        "b_nocontext is paired on strict per-seed cell "
                        "intersections (anchor cells excluded by construction)."),
        "options": {"car_mean_mode": a.car_mean_mode,
                    "car_mean_loo_fallback_cells_to_pop_mean": fallback_total,
                    "bootstrap": a.bootstrap},
        "population_means": {k: round(v, 4) for k, v in pop.items()},
        "reproduction_check": check,
    }
    with open(os.path.join(a.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"[out ] {os.path.join(a.out, 'provenance.json')}")

    # ---- figures (house style; fig1 caption corrected for this report) ----
    if not a.no_figures:
        figdir = os.path.join(a.out, "figures")
        make_figures(metrics, tests, figdir, err=spread)
        print(f"[out ] figures -> {figdir}")
        figdir2 = os.path.join(a.out, "figures_excl_n12")
        make_figures(metrics, tests, figdir2, exclude=("b_persona",), err=spread)
        print(f"[out ] figures (without the N=12 judge) -> {figdir2}")

    # ---- console summary ----
    print("\n=== overall (seed-averaged) ===")
    ov = metrics[metrics["scope"] == "overall"].sort_values("mae")
    print(ov[["judge", "mae", "within1", "icc", "rho"]].to_string(index=False))
    print("\n=== paired dMAE (A - B; negative => A better) ===")
    print(tests[["A", "B", "cells", "seeds", "dmae", "ci_lo", "ci_hi", "p",
                 "n_cells"]].to_string(index=False))
    if a.check:
        print("\n[check]", json.dumps(check, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
