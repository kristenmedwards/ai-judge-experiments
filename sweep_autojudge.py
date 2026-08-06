"""AUTOMATED sweep — iterate over judge designs until overall MAE < target.

Unlike ``run_autojudge.py`` (one round, agent-driven keep/revert), this driver
runs the whole ladder from ``judge_program.md`` by itself: it greedily searches
one lever at a time, carries the best-so-far forward, logs every experiment to
``results.tsv``, and STOPS as soon as the best overall MAE drops below
``--target-mae`` (default 1.0). If the ladder doesn't reach the target, an
optional randomized phase keeps proposing new combinations until the target is
met or the ``--max-experiments`` safety cap is hit (so it can't loop or spend
forever).

Cost: each experiment ≈ ``raters × test_size`` model calls, each sending
``1 + n_context`` images. Validate the mechanics FREE with ``--mock`` first, then
run live. Example:

    # free dry-of-mechanics
    python sweep_autojudge.py --data ../car_ratings_long_..._Q23.csv \
      --image-root ../selected_2000_isometric_upload_chunks_renamed \
      --raters 20 --test-size 20 --target-mae 1.0 --mock

    # live (uses OPENAI_* from .env)
    python sweep_autojudge.py --data <long.csv> --image-root <chunks> \
      --raters 20 --test-size 20 --target-mae 1.0 --max-experiments 30
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
from dataclasses import replace
from typing import List, Optional, Tuple

from car_judge import env as envmod
from car_judge.config import RunConfig, JudgeConfig
from car_judge.inner_loop import run_inner, InnerResult
from run_autojudge import (append_results_row, write_predictions,
                           git_commit_short, make_run_id, run_dir_for,
                           compute_extra_metrics, _fmt_extra)


class Sweeper:
    def __init__(self, run_cfg: RunConfig, target_mae: float,
                 max_experiments: int, keep_going: bool, log: bool = True,
                 run_dir: str = "outputs", run_id: str = ""):
        self.rc = run_cfg
        self.target = target_mae
        self.max_experiments = max_experiments
        self.keep_going = keep_going
        self.log = log
        self.n = 0
        self.best: Optional[Tuple[JudgeConfig, InnerResult]] = None
        # per-run output namespacing (nothing from a previous run is overwritten)
        self.run_dir = run_dir
        self.run_id = run_id
        self.per_run_tsv = os.path.join(run_dir, "results.tsv")

    # --- evaluate one config, log it, update best, return (result, is_new_best) ---
    def evaluate(self, jc: JudgeConfig) -> Tuple[Optional[InnerResult], bool]:
        if self.n >= self.max_experiments:
            return None, False
        self.n += 1
        jc.validate()
        print(f"\n===== experiment {self.n}/{self.max_experiments}: "
              f"{jc.name} ({jc.summary()}) =====")
        res = run_inner(self.rc, jc)
        is_best = self.best is None or res.score < self.best[1].score - 1e-9
        if is_best:
            self.best = (jc, res)
        # richer metric suite beyond MAE (ICC / Spearman / kappa / RMSE)
        xm = compute_extra_metrics(res.rows)
        if self.log:
            tag = f"[run={self.run_id}] " if self.run_id else ""
            extra = f" · {_fmt_extra(xm)}" if xm else ""
            row = {
                "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                "commit": git_commit_short(),
                "name": jc.name,
                "config": json.dumps(jc.to_dict(), separators=(",", ":")),
                "score_mae": f"{res.score:.6f}",
                "within1": f"{res.within1:.4f}",
                "n_raters": res.n_raters, "n_preds": res.n_predictions,
                "unparsed": res.n_unparsed,
                "status": "keep" if is_best else "discard",
                "note": f"{tag}sweep · {jc.summary()}{extra}",
            }
            append_results_row(row)                       # append-only master log
            append_results_row(row, path=self.per_run_tsv)  # + this run's own log
        # Keep EVERY experiment's per-prediction rows, not just the final best.
        # The configs differ by ~0.015 MAE, which is within measurement noise, so
        # deciding keep/discard needs a paired test across configs — impossible if
        # the losers' per-item errors are thrown away. Namespaced by run so a
        # re-run never overwrites a previous run's exp files.
        exp_csv = os.path.join(self.run_dir, "per_experiment",
                               f"exp{self.n:02d}_{jc.name}.csv")
        write_predictions(res.rows, exp_csv)
        if xm:  # full per-experiment metric suite alongside the predictions
            with open(os.path.splitext(exp_csv)[0] + ".metrics.json", "w",
                      encoding="utf-8") as fh:
                json.dump({"name": jc.name, "split_seed": self.rc.split_seed,
                           "macro_mae": res.score, **xm}, fh, indent=2)
        print(f"      MAE={res.score:.4f}  {_fmt_extra(xm)}  "
              f"(best so far {self.best[1].score:.4f} by '{self.best[0].name}')")
        return res, is_best

    def hit_target(self) -> bool:
        return self.best is not None and self.best[1].score < self.target

    def done(self) -> bool:
        return self.hit_target() or self.n >= self.max_experiments

    # --- the greedy ladder (judge_program.md phases) ---
    def run_ladder(self, context_sizes: List[int]) -> None:
        base = JudgeConfig(name="e0_baseline", n_context=0, context_strategy="random")
        self.evaluate(base)
        if self.done():
            return

        # Phase 1 — context size sweep (random)
        best_base = self.best[0]
        for N in context_sizes:
            if self.done():
                return
            self.evaluate(replace(best_base, name=f"e1_ctx{N}", n_context=N,
                                  context_strategy="random", context_order="as_selected"))
        base = self.best[0]  # carries best N* (may still be N=0)

        # Phase 2 — smarter context at N* (only if context helps)
        if base.n_context > 0:
            for strat in ("diverse", "rag_clip", "rag_profile"):
                if self.done():
                    return
                order = "similar_last" if strat.startswith("rag_") else "as_selected"
                self.evaluate(replace(base, name=f"e2_{strat}",
                                      context_strategy=strat, context_order=order))
            base = self.best[0]

            # Phase 3 — ordering control for rag_* strategies
            if base.context_strategy.startswith("rag_") and not self.done():
                self.evaluate(replace(base, name="e3_shuffle", context_order="shuffle"))
                base = self.best[0]

        # Phase 4 — person side-info, added greedily/cumulatively
        for lever, kw in (("q23", dict(include_q23=True)),
                          ("demo", dict(include_demographics=True)),
                          ("owned", dict(include_owned=True))):
            if self.done():
                return
            cand = replace(base, name=f"e4_{lever}", **kw)
            _, better = self.evaluate(cand)
            if better:
                base = self.best[0]  # keep the side-info that helped

        # Phase 5 — prompt wording
        for variant in ("persona", "concise"):
            if self.done():
                return
            _, better = self.evaluate(replace(base, name=f"e5_{variant}",
                                              prompt_variant=variant))
            if better:
                base = self.best[0]
        if not self.done():
            instr = ("Match this specific person's numbers, including their leniency "
                     "and quirks — do not substitute your own opinion. When unsure, "
                     "imitate the pattern in their example ratings.")
            _, better = self.evaluate(replace(base, name="e5_instr",
                                              extra_instructions=instr))
            if better:
                base = self.best[0]

    # --- randomized combinations after the ladder, until target or cap ---
    def run_random(self, context_sizes: List[int]) -> None:
        strategies = ["random", "diverse", "rag_clip", "rag_profile"]
        variants = ["default", "persona", "concise"]
        orders = ["as_selected", "similar_last", "shuffle"]
        i = 0
        while not self.done():
            i += 1
            rng = random.Random(f"rand:{i}")  # deterministic per step (no Math.random)
            N = rng.choice(context_sizes)
            strat = rng.choice(strategies) if N > 0 else "random"
            cfg = JudgeConfig(
                name=f"r{i}",
                n_context=N, context_strategy=strat,
                context_order=(rng.choice(orders) if strat.startswith("rag_") else "as_selected"),
                include_q23=bool(rng.getrandbits(1)),
                include_demographics=bool(rng.getrandbits(1)),
                include_owned=bool(rng.getrandbits(1)),
                prompt_variant=rng.choice(variants),
            )
            self.evaluate(cfg)


def parse_args(argv):
    p = argparse.ArgumentParser(description="Automated judge sweep until MAE < target.")
    p.add_argument("--data", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--raters", type=int, default=20)
    p.add_argument("--test-size", type=int, default=20)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--context-sizes", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--target-mae", type=float, default=1.0)
    p.add_argument("--max-experiments", type=int, default=30,
                   help="hard safety cap so the sweep can't run/spend forever")
    p.add_argument("--keep-going", action="store_true",
                   help="after the ladder, try randomized combos until target/cap")
    p.add_argument("--run-tag", default=None,
                   help="optional label appended to the run id "
                        "(outputs/runs/<timestamp>_<tag>/)")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-log", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    creds = envmod.resolve_credentials(a.model, a.base_url, a.api_key)
    run_cfg = RunConfig(
        data_csv=a.data, image_root=a.image_root,
        base_url=creds.base_url, api_key=creds.api_key, model=creds.model,
        max_concurrency=a.concurrency, n_raters=a.raters, test_size=a.test_size,
        split_seed=a.split_seed, mock=a.mock, dry_run=a.dry_run,
    )
    if not (a.mock or a.dry_run) and not creds.is_live_ready:
        print("!! No OPENAI_API_KEY. Put it in .env / export it, or use --mock.",
              file=sys.stderr)
        return 2

    est = a.raters * a.test_size
    # Isolate this run's outputs under outputs/runs/<run_id>/ so nothing from a
    # previous sweep is overwritten.
    run_id = make_run_id(a.run_tag)
    run_dir = run_dir_for(run_id)
    print(f"[sweep] target MAE < {a.target_mae} · cap {a.max_experiments} experiments · "
          f"~{est} model calls/experiment ({'MOCK' if a.mock else 'LIVE'})")
    print(f"[sweep] run id: {run_id}   ->   {run_dir}/")

    # provenance: record exactly how this run was launched (never a secret)
    meta = {
        "run_id": run_id, "commit": git_commit_short(),
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "data": a.data, "image_root": a.image_root,
        "raters": a.raters, "test_size": a.test_size, "split_seed": a.split_seed,
        "context_sizes": a.context_sizes, "target_mae": a.target_mae,
        "max_experiments": a.max_experiments, "keep_going": a.keep_going,
        "mock": a.mock, "dry_run": a.dry_run, "model": creds.model,
    }
    with open(os.path.join(run_dir, "sweep_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    sw = Sweeper(run_cfg, a.target_mae, a.max_experiments, a.keep_going,
                 log=not a.no_log, run_dir=run_dir, run_id=run_id)
    sw.run_ladder(a.context_sizes)
    if a.keep_going and not sw.hit_target():
        print("\n[sweep] ladder exhausted without hitting target; randomized search…")
        sw.run_random(a.context_sizes)

    jc, res = sw.best
    print("\n" + "=" * 64)
    if sw.hit_target():
        print(f"[sweep] TARGET MET: MAE {res.score:.4f} < {a.target_mae} "
              f"after {sw.n} experiments.")
    else:
        print(f"[sweep] stopped at {sw.n} experiments (cap={a.max_experiments}). "
              f"Best MAE {res.score:.4f} did NOT reach {a.target_mae}.")
        if not a.keep_going:
            print("        (re-run with --keep-going to try randomized combinations.)")
    print(f"[sweep] best judge: {jc.name}  ({jc.summary()})")
    print(f"[sweep] per-dimension MAE: " +
          " ".join(f"{d}={v:.3f}" for d, v in res.per_dimension_mae.items()))
    # Winner artifacts go INTO this run's folder — the top-level
    # best_judge_config.json (a prior run's record) is left untouched.
    write_predictions(res.rows, os.path.join(run_dir, f"best_{jc.name}.csv"))
    best_cfg_path = os.path.join(run_dir, "best_judge_config.json")
    with open(best_cfg_path, "w", encoding="utf-8") as fh:
        json.dump(jc.to_dict(), fh, indent=2)
    print(f"[sweep] wrote {best_cfg_path} + {run_dir}/best_{jc.name}.csv")
    print(f"[sweep] this run's log: {os.path.join(run_dir, 'results.tsv')} · "
          f"master log: results.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
