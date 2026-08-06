"""Rebuild the AI-judge report — metrics, statistical tests, and figures —
reproducibly from the saved per-prediction ground truth.

Ground truth  : outputs/runs/*_fp_<judge>_s<seed>/per_experiment/<judge>.csv
                (one row per rater x held-out car x dimension: predicted + true)
Human ratings : the merged long CSV (for the pop_mean / car_mean baselines)
Metric code   : evaluation/stats.py (the committed, paper-grade functions)

It writes, into --out (default report_and_figures_reproduced/, so it never
clobbers the old hand-made report_and_figures/):
  metrics_by_dimension_seedavg.csv   judges + baselines, per dimension + overall
  statistical_tests_seedavg.csv      paired dMAE with rater-cluster bootstrap CIs
  provenance.json                    exact inputs, options, and reproduction check
  figures/*.png

Usage:
  python build_report.py \
    --runs-root outputs/runs \
    --human ../car_ratings_long_July8_July22_merged_with_color_and_Q23.csv \
    --check          # cross-check recomputed judge MAEs against results.tsv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from report_pipeline import data as D
from report_pipeline import baselines as B
from report_pipeline import metrics as M
from report_pipeline import figures as F

# comparisons reported in statistical_tests_seedavg.csv (A vs B; dMAE<0 => A better)
COMPARISONS = [
    ("b_persona", "b_nocontext"),
    ("b_persona", "car_mean"),
    ("b_persona", "pop_mean"),
    ("g_card_twostage", "b_persona"),
    ("car_mean", "b_nocontext"),
    ("car_mean", "pop_mean"),
    ("b_nocontext", "pop_mean"),
]


def find_human(explicit):
    if explicit:
        return explicit
    for c in glob.glob("../car_ratings_long_*merged*with_color_and_Q23.csv") + \
             glob.glob("car_ratings_long_*merged*with_color_and_Q23.csv"):
        return c
    raise SystemExit("could not auto-find the merged human CSV; pass --human")


def crosscheck_results_tsv(recomputed, results_tsv):
    """Compare recomputed per-seed overall MAE to the full-pool rows in results.tsv."""
    if not os.path.exists(results_tsv):
        return {"checked": False, "reason": f"{results_tsv} not found"}
    tsv = pd.read_csv(results_tsv, sep="\t")
    tsv = tsv[tsv["note"].astype(str).str.contains("fp_|confirm800", regex=True, na=False)]
    checks = []
    for (judge, seed), mae in recomputed.items():
        rows = tsv[(tsv["name"] == judge) & (tsv["note"].str.contains(f"seed={seed}", na=False))]
        if not len(rows):
            continue
        ref = float(rows.iloc[0]["score_mae"])
        checks.append({"judge": judge, "seed": seed, "recomputed": round(mae, 6),
                       "results_tsv": round(ref, 6), "abs_diff": round(abs(mae - ref), 8)})
    ok = all(c["abs_diff"] < 1e-4 for c in checks) if checks else False
    return {"checked": bool(checks), "all_match": ok, "rows": checks}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="outputs/runs")
    ap.add_argument("--human", default=None)
    ap.add_argument("--results", default="results.tsv")
    ap.add_argument("--out", default="report_and_figures_reproduced")
    ap.add_argument("--car-mean-mode", choices=["loo", "include_self"], default="loo",
                    help="loo (leave-one-out; honest, default) or include_self "
                         "(reproduces the old, self-leaking number)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--check", action="store_true",
                    help="cross-check recomputed judge MAEs against results.tsv")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    human_csv = find_human(a.human)
    runs = D.discover_runs(a.runs_root)
    seeds = sorted({s for _, s in runs})
    judges = sorted({j for j, _ in runs})
    print(f"[data] {len(runs)} runs: judges={judges} seeds={seeds}")
    print(f"[data] human ratings: {human_csv}")

    H = D.load_human_ratings(human_csv)
    pop = B.population_means(H)
    pools = B.car_rating_pools(H)

    # ---- per-seed metric rows for judges + baselines (identical cells) ----
    per_seed_frames = []
    recomputed_overall = {}
    combined = {}                      # judge -> concat of all seed prediction frames
    fallback_total = 0
    for seed in seeds:
        # reference cells for this seed (all judges share the same split)
        ref_key = ("b_persona", seed) if ("b_persona", seed) in runs else \
                  sorted([k for k in runs if k[1] == seed])[0]
        cells = D.cells_of(D.load_predictions(runs[ref_key]))

        seed_pred_frames = {}
        # real judges
        for (j, s), path in runs.items():
            if s != seed:
                continue
            df = D.load_predictions(path)
            seed_pred_frames[j] = df
            recomputed_overall[(j, seed)] = M.pooled_suite(df["true"], df["predicted"])["mae"]
        # baselines synthesized on this seed's cells
        pm = B.predict_pop_mean(cells, pop)
        cm, fb = B.predict_car_mean_loo(cells, pools, pop,
                                        include_self=(a.car_mean_mode == "include_self"))
        fallback_total += fb
        seed_pred_frames["pop_mean"] = B.baseline_frame(cells, "pop_mean", pm)
        seed_pred_frames["car_mean"] = B.baseline_frame(cells, "car_mean", cm)

        for j, df in seed_pred_frames.items():
            per_seed_frames.append(pd.DataFrame(M.metrics_for_judge(df, j)))
            combined.setdefault(j, []).append(df)

    metrics = M.seed_average(per_seed_frames)
    combined = {j: pd.concat(v, ignore_index=True) for j, v in combined.items()}

    # order columns like the original report
    cols = ["judge", "scope", "mae", "rmse", "within1", "icc", "rho", "wkappa",
            "within_rater_rho", "n"]
    metrics = metrics[[c for c in cols if c in metrics.columns]]
    metrics_path = os.path.join(a.out, "metrics_by_dimension_seedavg.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"[out ] {metrics_path}")

    # ---- paired statistical tests ----
    trows = []
    for A_, B_ in COMPARISONS:
        if A_ in combined and B_ in combined:
            r = M.paired_dmae(combined[A_], combined[B_], n_boot=a.bootstrap)
            trows.append({"A": A_, "B": B_, **r})
    tests = pd.DataFrame(trows)
    tests_path = os.path.join(a.out, "statistical_tests_seedavg.csv")
    tests.to_csv(tests_path, index=False)
    print(f"[out ] {tests_path}")

    # ---- provenance + reproduction check ----
    check = crosscheck_results_tsv(recomputed_overall, a.results) if a.check else \
        {"checked": False, "reason": "run with --check"}
    prov = {
        "inputs": {"runs_root": os.path.abspath(a.runs_root),
                   "runs": {f"{j}_s{s}": os.path.abspath(p) for (j, s), p in runs.items()},
                   "human_csv": os.path.abspath(human_csv),
                   "metric_code": "evaluation/stats.py"},
        "options": {"car_mean_mode": a.car_mean_mode,
                    "car_mean_loo_fallback_cells_to_pop_mean": fallback_total,
                    "seeds": seeds, "bootstrap": a.bootstrap},
        "population_means": {k: round(v, 4) for k, v in pop.items()},
        "reproduction_check_vs_results_tsv": check,
    }
    with open(os.path.join(a.out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"[out ] {os.path.join(a.out, 'provenance.json')}")
    if a.check:
        tag = "ALL MATCH" if check.get("all_match") else "MISMATCH/none"
        print(f"[check] recomputed judge MAE vs results.tsv: {tag}")
        for c in check.get("rows", []):
            print(f"        {c['judge']:16s} s{c['seed']}  recomputed={c['recomputed']:.6f} "
                  f"results.tsv={c['results_tsv']:.6f}  diff={c['abs_diff']:.1e}")

    # ---- figures ----
    if not a.no_figures:
        figs = F.make_all(metrics, tests, os.path.join(a.out, "figures"))
        print(f"[out ] {len(figs)} figures -> {os.path.join(a.out, 'figures')}")

    # ---- console summary ----
    print("\n=== overall (seed-averaged) ===")
    ov = metrics[metrics["scope"] == "overall"].sort_values("mae")
    print(ov[["judge", "mae", "within1", "icc", "rho"]].to_string(index=False))
    if len(tests):
        print("\n=== paired dMAE (A - B; negative => A better) ===")
        print(tests[["A", "B", "dmae", "ci_lo", "ci_hi", "p"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
