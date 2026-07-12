"""Sweep context sizes (e.g. N = 0, 5, 10, 15) with hold-out testing.

For every rater the sweep queries the judges at several context sizes and
writes ONE combined predictions CSV (same columns as ``run_experiment`` plus
``holdout_mode`` / ``context_size`` / ``draw_seed``) that the evaluation
pipeline (``evaluation/evaluate_predictions.py``) consumes directly.

Hold-out schemes (``--holdout-mode``):

fixed      One held-out test set of ``--test-size`` cars per rater, fixed
           across ALL context sizes (the run_experiment default). Context is
           re-drawn from the remaining pool ``--repeats`` times per size.
           Best for paired comparisons across context sizes: every condition
           is scored on identical items.

remainder  "Hold-multiple-out": for each repeat, shuffle the rater's cars,
           use the first N as context and ALL remaining cars as the test
           set. Maximizes test items per condition, but test sets differ
           across context sizes, so cross-size comparisons are unpaired.

loo        "Hold-one-out": every car is the test item once; the context is
           drawn from the other cars. Most data-efficient, most requests
           (n_cars x len(context_sizes) x repeats per rater).

Examples
--------
Dry run — count requests before spending GPU time:

    python -m car_judge.run_context_sweep \
      --data ".../Prolific.csv" --image-root ".../chunks" \
      --raters 3 --context-sizes 0 5 10 15 --holdout-mode fixed \
      --test-size 8 --repeats 3 --dry-run

Live sweep:

    python -m car_judge.run_context_sweep \
      --data ".../Prolific.csv" --image-root ".../chunks" \
      --base-url http://gpu:8000/v1 --model Qwen/Qwen3.5-9B \
      --raters 5 --context-sizes 0 5 10 15 --holdout-mode fixed \
      --test-size 8 --repeats 3 --out outputs/sweep_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

from .config import RunConfig, DIMENSIONS
from .client import QwenVLMClient
from . import data as datamod
from . import metrics as metricsmod
from .judges import make_judge, Prediction

FIELDS = [
    "rater", "judge", "holdout_mode", "context_size", "draw_seed",
    "target_image", "car_name", "dimension",
    "predicted", "true", "abs_error",
    "n_context", "n_images", "enable_thinking",
]


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep in-context sizes with hold-out testing.")
    p.add_argument("--data", required=True, help="Prolific Qualtrics CSV export")
    p.add_argument("--image-root", required=True, help="folder with chunk_XX/car_N.png")
    p.add_argument("--car-name-mapping", default=None)

    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable-thinking", action="store_true")

    p.add_argument("--raters", type=int, default=2)
    p.add_argument("--context-sizes", type=int, nargs="+", default=[0, 5, 10, 15],
                   help="context sizes to sweep; 0 = the no-context judge")
    p.add_argument("--holdout-mode", choices=["fixed", "remainder", "loo"],
                   default="fixed")
    p.add_argument("--test-size", type=int, default=8,
                   help="held-out cars per rater (fixed mode only)")
    p.add_argument("--repeats", type=int, default=1,
                   help="independent context draws per (rater, size)")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--max-targets", type=int, default=None,
                   help="cap test targets per (rater, size, repeat); mainly to "
                        "keep loo/remainder sweeps affordable")

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default="outputs/sweep_predictions.csv")
    return p.parse_args(argv)


def make_run_config(a: argparse.Namespace) -> RunConfig:
    kw = dict(
        data_csv=a.data,
        image_root=a.image_root,
        car_name_mapping=a.car_name_mapping,
        temperature=a.temperature,
        max_tokens=a.max_tokens,
        seed=a.seed,
        enable_thinking=a.enable_thinking,
        n_raters=a.raters,
        test_size=a.test_size,
        split_seed=a.split_seed,
        dry_run=a.dry_run,
        out_path=a.out,
    )
    if a.base_url:
        kw["base_url"] = a.base_url
    if a.api_key:
        kw["api_key"] = a.api_key
    if a.model:
        kw["model"] = a.model
    return RunConfig(**kw)


# --------------------------------------------------------------------------- #
# Hold-out condition generators
# --------------------------------------------------------------------------- #
# Each condition is (context_size, draw_seed, context_images, test_images).
Condition = Tuple[int, int, List[str], List[str]]


def _eligible_cars(rater: datamod.Rater) -> List[str]:
    cars = [c for c in rater.fully_rated_cars() if c not in rater.anchor_images]
    return sorted(cars)


def iter_conditions(
    rater: datamod.Rater,
    mode: str,
    context_sizes: List[int],
    repeats: int,
    test_size: int,
    split_seed: int,
    max_targets: Optional[int],
) -> Iterator[Condition]:
    cars = _eligible_cars(rater)
    rid = rater.response_id

    def cap(targets: List[str]) -> List[str]:
        return targets[:max_targets] if max_targets else targets

    if mode == "fixed":
        split = datamod.make_split(rater, test_size=test_size, split_seed=split_seed)
        for n in context_sizes:
            # 0-shot doesn't depend on the draw, so one repeat suffices.
            for r in range(1 if n == 0 else repeats):
                ctx = datamod.sample_context(split, n, draw_seed=r)
                yield n, r, ctx, cap(split.test_images)

    elif mode == "remainder":
        for n in context_sizes:
            for r in range(1 if n == 0 else repeats):
                rng = random.Random(f"{rid}:rem:{split_seed}:{n}:{r}")
                shuffled = cars[:]
                rng.shuffle(shuffled)
                ctx, targets = shuffled[:n], sorted(shuffled[n:])
                if not targets:
                    raise ValueError(
                        f"rater {rid}: context_size={n} leaves no test cars "
                        f"({len(cars)} available)")
                yield n, r, ctx, cap(targets)

    elif mode == "loo":
        for n in context_sizes:
            if n > len(cars) - 1:
                raise ValueError(
                    f"rater {rid}: context_size={n} too large for loo with "
                    f"{len(cars)} cars")
            for r in range(1 if n == 0 else repeats):
                for target in cap(cars):
                    rest = [c for c in cars if c != target]
                    rng = random.Random(f"{rid}:loo:{split_seed}:{n}:{r}:{target}")
                    ctx = rng.sample(rest, n) if n else []
                    yield n, r, ctx, [target]
    else:
        raise ValueError(f"unknown holdout mode {mode!r}")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _rows(pred: Prediction, rater_id: str, truth: Dict[str, int], car_name: str,
          holdout_mode: str, context_size: int, draw_seed: int,
          enable_thinking: bool) -> List[dict]:
    rows = []
    for dim in DIMENSIONS:
        pv, tv = pred.predicted.get(dim), truth.get(dim)
        err = abs(pv - tv) if (pv is not None and tv is not None) else ""
        rows.append({
            "rater": rater_id,
            "judge": pred.judge,
            "holdout_mode": holdout_mode,
            "context_size": context_size,
            "draw_seed": draw_seed,
            "target_image": pred.target_image,
            "car_name": car_name,
            "dimension": dim,
            "predicted": "" if pv is None else pv,
            "true": "" if tv is None else tv,
            "abs_error": err,
            "n_context": pred.n_context,
            "n_images": pred.n_images,
            "enable_thinking": enable_thinking,
        })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = make_run_config(a)

    print(f"[load] images from {cfg.image_root}")
    image_index = datamod.build_image_index(cfg.image_root)
    car_names = datamod.load_car_names(cfg.car_name_mapping)

    min_cars = (a.test_size + 1 if a.holdout_mode == "fixed"
                else max(a.context_sizes) + 1)
    raters = datamod.load_raters(cfg.data_csv, min_cars=min_cars)
    print(f"[load] {len(raters)} usable raters (need >= {min_cars} rated cars)")
    raters = raters[:a.raters]

    mode = "DRY RUN (no server)" if a.dry_run else f"LIVE -> {cfg.base_url} [{cfg.model}]"
    print(f"[run ] {mode}; holdout={a.holdout_mode} sizes={a.context_sizes} "
          f"repeats={a.repeats} thinking={cfg.enable_thinking}")

    client = QwenVLMClient(cfg)
    judges = {
        "no_context": make_judge("no_context", client, image_index),
        "in_context": make_judge("in_context", client, image_index),
    }

    all_rows: List[dict] = []
    n_requests = 0
    max_images = 0
    # NoContextJudge caches internally: 0-shot depends only on the target image.
    score_bucket: Dict[tuple, list] = defaultdict(list)

    for rater in raters:
        n_eligible = len(_eligible_cars(rater))
        print(f"\n[rater] {rater.response_id}: {n_eligible} eligible cars")
        for n, r, ctx_imgs, targets in iter_conditions(
                rater, a.holdout_mode, a.context_sizes, a.repeats,
                a.test_size, a.split_seed, a.max_targets):
            context = [(img, rater.ratings[img]) for img in ctx_imgs]
            for target in targets:
                truth = rater.ratings[target]
                if n == 0:
                    pred = judges["no_context"].predict(target)
                else:
                    pred = judges["in_context"].predict(target, context)
                if not pred.from_cache:
                    n_requests += 1
                max_images = max(max_images, pred.n_images)

                if not a.dry_run:
                    for dim in DIMENSIONS:
                        pv, tv = pred.predicted.get(dim), truth.get(dim)
                        if pv is not None and tv is not None:
                            score_bucket[(n, dim)].append((tv, pv))

                all_rows.extend(_rows(
                    pred, rater.response_id, truth, car_names.get(target, ""),
                    a.holdout_mode, n, r, cfg.enable_thinking))
            print(f"  [cond] N={n:3d} draw={r} context={len(context):3d} "
                  f"targets={len(targets):3d}")

    if a.dry_run:
        print(f"\n[dry ] total model requests = {n_requests} "
              f"(0-shot cached per unique target)")
        print(f"[dry ] max images in one request = {max_images}. "
              f"Serve with --limit-mm-per-prompt image={max_images} (or higher).")
        print("[dry ] no predictions written (dry run).")
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[save] {len(all_rows)} rows -> {a.out}")

    print("\n[score] MAE by context size (pooled over dimensions):")
    for n in sorted(a.context_sizes):
        flat = []
        for dim in DIMENSIONS:
            flat += score_bucket.get((n, dim), [])
        s = metricsmod.summarize(flat)
        print(f"  N={n:3d}  n={s['n']:4d}  MAE={s['mae']:.3f}  "
              f"within1={s['within1']:.3f}")
    print("\n[next] evaluate with:\n"
          f"  python evaluation/evaluate_predictions.py --predictions {a.out} "
          f"--reference rater --out-dir evaluation/outputs/sweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
