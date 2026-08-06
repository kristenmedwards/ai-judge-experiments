"""Context-SIZE ablation: N = 0,4,8,12,16,20 exemplars, everything else fixed.

Unlike ``sweep_autojudge.py`` (greedy, one-lever-at-a-time), this driver answers a
single clean question: *how does held-out accuracy scale with the number of the
rater's own cars shown as context?* It runs one round per context size on an
IDENTICAL rater set and IDENTICAL held-out test set, so the six points form a
properly paired curve (only ``n_context`` changes between them).

Design choices that make N=20 actually reachable (see the harness notes):

* ``include_anchors_in_pool=True`` — the 4 shared "anchor" cars are folded back
  into each rater's pool, so more cars carry a full rating and can serve as
  context. (Default harness behaviour keeps them out as a clean reference; here
  we deliberately opt in, because context depth is the goal of this sweep.)
* ``min_cars = test_size + max(context_sizes)`` — a rater is only included if
  they have enough fully-rated cars to supply BOTH the held-out test set AND the
  largest context size. That is the "raters that meet the context criteria"
  filter: every included rater can produce all six context sizes, so the curve is
  on one constant population instead of a shrinking one.
* ``test_size`` must satisfy ``test_size + max(context) <= 34`` (the deepest
  raters rated 34 cars incl. anchors). With max context 20 that means
  ``test_size <= 14``. 14 keeps the most held-out targets; smaller trades eval
  data for more context-sampling headroom at N=20.

Because eligibility and the split depend only on ``test_size`` /
``split_seed`` / ``include_anchors_in_pool`` (never on ``n_context``), the kept
raters and their held-out cars are byte-identical across every N — the sweep is
paired by construction.

Cost: each config sends ``n_kept_raters * test_size * (1 + n_context)`` image
inputs. Validate with ``--mock`` first (free), then go live.

Example
-------
    # free dry check of the mechanics (needs the image folder to exist):
    python sweep_context_size.py --data ../car_ratings_long_..._Q23.csv \
        --image-root ../selected_2000_isometric_upload_chunks_renamed \
        --test-size 14 --mock

    # live, all eligible raters, two seeds:
    python sweep_context_size.py --data ../car_ratings_long_..._Q23.csv \
        --image-root ../selected_2000_isometric_upload_chunks_renamed \
        --test-size 14 --split-seeds 0 1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

from car_judge import env as envmod
from car_judge.config import RunConfig, JudgeConfig
from car_judge.inner_loop import run_inner

# Reuse the EXACT logging / metric helpers run_autojudge uses, so rows in
# results.tsv and the .metrics.json files are directly comparable to prior runs.
from run_autojudge import (
    append_results_row, compute_extra_metrics, _fmt_extra,
    write_predictions, make_run_id, run_dir_for, git_commit_short,
)

DEFAULT_SIZES = [0, 4, 8, 12, 16, 20]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="merged long csv (…_with_color_and_Q23.csv)")
    p.add_argument("--image-root", required=True, help="folder with chunk_XX/car_N.png")

    # the sweep itself
    p.add_argument("--context-sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                   help="context sizes to sweep (default: 0 4 8 12 16 20)")
    p.add_argument("--test-size", type=int, default=14,
                   help="held-out cars per rater. Must satisfy "
                        "test_size + max(context) <= 34; default 14")
    p.add_argument("--split-seeds", type=int, nargs="+", default=[0],
                   help="one round per seed per size (e.g. 0 1). Same seed => "
                        "same raters & held-out set across all sizes")
    p.add_argument("--prompt-variant", default="persona",
                   choices=["default", "concise", "persona", "two_stage"],
                   help="held CONSTANT across the sweep so only context size varies "
                        "(default: persona, the recommended judge)")
    p.add_argument("--context-strategy", default="random",
                   help="held constant across the sweep (default: random)")

    # anchors: on by default here (that is the whole point of this sweep)
    p.add_argument("--no-anchors", action="store_true",
                   help="do NOT include anchor cars in the pool (reverts to the "
                        "clean-reference default; then test_size+max(ctx) must be "
                        "<= ~30 and you lose the July-8 raters anyway)")

    # population
    p.add_argument("--max-raters", type=int, default=None,
                   help="cap eligible raters (for cheap validation); default = ALL "
                        "raters that meet the context criteria")
    p.add_argument("--min-cars", type=int, default=None,
                   help="override the eligibility floor (default: test_size + max "
                        "context). Set to 34 when running a SUBSET of sizes so the "
                        "rater pool stays identical to a full 0-20 sweep")

    # endpoint (else env / .env)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-retries", type=int, default=4)

    # run mode
    p.add_argument("--mock", action="store_true", help="deterministic offline judge")
    p.add_argument("--dry-run", action="store_true", help="build requests, no server")
    p.add_argument("--run-tag", default="ctxsweep",
                   help="artifacts land under outputs/runs/<ts>_<tag>/")
    p.add_argument("--no-log", action="store_true", help="don't append to results.tsv")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)

    sizes = sorted(set(a.context_sizes))
    max_ctx = max(sizes)
    include_anchors = not a.no_anchors

    # Guard: with anchors the deepest raters have 34 cars; without, ~30.
    ceiling = 34 if include_anchors else 30
    if a.test_size + max_ctx > ceiling:
        print(f"!! test_size ({a.test_size}) + max context ({max_ctx}) = "
              f"{a.test_size + max_ctx} > {ceiling}. No rater has that many "
              f"fully-rated cars, so the sweep would keep 0 raters. "
              f"Lower --test-size to <= {ceiling - max_ctx} or drop the top size.",
              file=sys.stderr)
        return 2

    # THE eligibility floor: enough cars for the held-out set AND the biggest N
    # (or an explicit override, to pin the pool of a partial sweep to a full one's).
    min_cars = a.min_cars if a.min_cars is not None else (a.test_size + max_ctx)

    creds = envmod.resolve_credentials(a.model, a.base_url, a.api_key)
    if not (a.mock or a.dry_run) and not creds.is_live_ready:
        print("!! No OPENAI_API_KEY. Put it in .env, or use --mock / --dry-run.",
              file=sys.stderr)
        return 2

    run_id = make_run_id(a.run_tag)
    run_dir = run_dir_for(run_id)
    print(f"[sweep] context sizes {sizes}  test_size={a.test_size}  "
          f"include_anchors={include_anchors}  min_cars(floor)={min_cars}  "
          f"seeds={a.split_seeds}  prompt={a.prompt_variant}")
    print(f"[sweep] run dir: {run_dir}")

    summary = []          # (seed, n_context, n_raters, mae, within1, xm) per round
    for seed in a.split_seeds:
        for n_ctx in sizes:
            judge_cfg = JudgeConfig(
                name=f"ctx{n_ctx}",
                n_context=n_ctx,
                context_strategy=a.context_strategy,
                context_order="as_selected",
                prompt_variant=a.prompt_variant,
                temperature=0.0,
            )
            run_cfg = RunConfig(
                data_csv=a.data, image_root=a.image_root,
                base_url=creds.base_url, api_key=creds.api_key, model=creds.model,
                temperature=0.0, max_tokens=a.max_tokens,
                max_concurrency=a.concurrency, max_retries=a.max_retries,
                # --- population held fixed across every size (only n_context varies) ---
                n_raters=a.max_raters,                     # None => all eligible
                test_size=a.test_size,
                split_seed=seed,
                min_cars=min_cars,                         # the context-criteria filter
                include_anchors_in_pool=include_anchors,   # anchors as usable context
                dry_run=a.dry_run, mock=a.mock,
            )

            tag = f"ctx{n_ctx}_s{seed}"
            print(f"\n[round] {tag}  (n_context={n_ctx}, seed={seed})")
            result = run_inner(run_cfg, judge_cfg)

            pred_path = os.path.join(run_dir, "per_experiment", f"{tag}.csv")
            write_predictions(result.rows, pred_path)

            xm = compute_extra_metrics(result.rows) if not a.dry_run else {}
            if xm:
                metrics_path = os.path.splitext(pred_path)[0] + ".metrics.json"
                with open(metrics_path, "w", encoding="utf-8") as fh:
                    json.dump({"name": judge_cfg.name, "n_context": n_ctx,
                               "split_seed": seed, "macro_mae": result.score,
                               "n_raters": result.n_raters, **xm}, fh, indent=2)

            if not a.no_log and not a.dry_run:
                note = f"[run={run_id}] ctxsweep size={n_ctx} seed={seed} " \
                       f"anchors={include_anchors} test_size={a.test_size}"
                if xm:
                    note += " · " + _fmt_extra(xm)
                row = {
                    "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                    "commit": git_commit_short(),
                    "name": judge_cfg.name,
                    "config": json.dumps(judge_cfg.to_dict(), separators=(",", ":")),
                    "score_mae": f"{result.score:.6f}",
                    "within1": f"{result.within1:.4f}",
                    "n_raters": result.n_raters,
                    "n_preds": result.n_predictions,
                    "unparsed": result.n_unparsed,
                    "status": "ctxsweep",
                    "note": note,
                }
                append_results_row(row)                                   # master log
                append_results_row(row, path=os.path.join(run_dir, "results.tsv"))

            summary.append((seed, n_ctx, result.n_raters, result.score,
                            result.within1, xm))

    # --- tidy end-of-run table (the context-scaling curve) ---
    if not a.dry_run:
        print("\n=== context-size sweep summary ===")
        print(f"{'seed':>4} {'N':>3} {'raters':>7} {'MAE':>8} {'within1':>8} "
              f"{'ICC':>6} {'rho':>6}")
        for seed, n_ctx, nr, mae, w1, xm in summary:
            icc = xm.get("icc2_1", float("nan")) if xm else float("nan")
            rho = xm.get("spearman_rho", float("nan")) if xm else float("nan")
            print(f"{seed:>4} {n_ctx:>3} {nr:>7} {mae:>8.4f} {w1:>8.4f} "
                  f"{icc:>6.3f} {rho:>6.3f}")
        # seed-averaged MAE per size, if >1 seed
        if len(a.split_seeds) > 1:
            print("\nseed-averaged MAE by context size:")
            for n_ctx in sizes:
                maes = [s[3] for s in summary if s[1] == n_ctx]
                print(f"  N={n_ctx:>2}: MAE={sum(maes)/len(maes):.4f}  (seeds {a.split_seeds})")
    print(f"\n[done] artifacts under {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
