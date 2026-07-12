"""CLI driver: query the Qwen VLM judges over held-out cars and save predictions.

Examples
--------
Dry run (no server; inspect the requests):

    python -m car_judge.run_experiment \
      --data "../Car Aesthetics Ratings - Prolific_July 8, 2026_10.14.csv" \
      --image-root "../selected_2000_isometric_upload_chunks_renamed" \
      --raters 2 --test-size 8 --context-size 10 --dry-run

Live run against a served model:

    python -m car_judge.run_experiment \
      --data ".../Prolific.csv" --image-root ".../chunks" \
      --base-url http://gpu:8000/v1 --model Qwen/Qwen3.5-9B \
      --raters 5 --test-size 8 --context-size 10 --out outputs/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from .config import RunConfig, DIMENSIONS
from .client import QwenVLMClient
from . import data as datamod
from . import metrics as metricsmod
from .judges import make_judge, Prediction
from .prompts import count_images


def parse_args(argv: List[str]) -> RunConfig:
    p = argparse.ArgumentParser(description="Run Qwen VLM car-aesthetics judges.")
    p.add_argument("--data", required=True, help="Prolific Qualtrics CSV export")
    p.add_argument("--image-root", required=True, help="folder with chunk_XX/car_N.png")
    p.add_argument("--car-name-mapping", default=None, help="optional car_name_mapping.csv")

    p.add_argument("--base-url", default=None, help="vLLM OpenAI-compatible base URL")
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable-thinking", action="store_true", help="Qwen reasoning mode")

    p.add_argument("--judges", nargs="+", default=["no_context", "in_context"],
                   choices=["no_context", "in_context"])
    p.add_argument("--raters", type=int, default=2, help="number of raters to run")
    p.add_argument("--test-size", type=int, default=8, help="held-out cars per rater")
    p.add_argument("--context-size", type=int, default=10, help="exemplars for in_context")
    p.add_argument("--split-seed", type=int, default=0)

    p.add_argument("--dry-run", action="store_true",
                   help="build requests but do not contact a server")
    p.add_argument("--out", default=None,
                   help="predictions CSV path; if omitted, an organized run "
                        "folder is auto-generated under outputs/")
    a = p.parse_args(argv)

    kw = dict(
        data_csv=a.data,
        image_root=a.image_root,
        car_name_mapping=a.car_name_mapping,
        temperature=a.temperature,
        max_tokens=a.max_tokens,
        seed=a.seed,
        enable_thinking=a.enable_thinking,
        judges=a.judges,
        n_raters=a.raters,
        test_size=a.test_size,
        context_size=a.context_size,
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


def build_out_path(cfg: RunConfig) -> str:
    """Auto-generate an organized, non-colliding predictions path from the config.

    Layout: outputs/<judges>__<timestamp>__<model>__<params>/predictions.csv
    e.g.    outputs/no_context__2026-07-11T14-30-22__Qwen3.5-9B__r5_t8_seed0/
    Parameters that change results are encoded in the folder name so runs are
    sortable by date and greppable by parameter.
    """
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model = cfg.model.split("/")[-1]
    judges = "+".join(cfg.judges)
    parts = [f"r{cfg.n_raters}", f"t{cfg.test_size}"]
    if "in_context" in cfg.judges:
        parts.append(f"c{cfg.context_size}")
    if cfg.enable_thinking:
        parts.append("think")
    parts.append(f"seed{cfg.split_seed}")
    slug = f"{judges}__{ts}__{model}__{'_'.join(parts)}"
    return os.path.join("outputs", slug, "predictions.csv")


def write_run_info(path: str, cfg: RunConfig, rater_ids: List[str],
                   duration_s: float, n_rows: int,
                   scores: Dict[str, Dict[str, dict]]) -> None:
    """Save a run_info.json next to the predictions with the key run metadata."""
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(duration_s, 1),
        "judges": cfg.judges,
        "model": cfg.model,
        "n_raters": cfg.n_raters,
        "raters": rater_ids,
        "test_size": cfg.test_size,
        "context_size": cfg.context_size if "in_context" in cfg.judges else None,
        "split_seed": cfg.split_seed,
        "enable_thinking": cfg.enable_thinking,
        "n_prediction_rows": n_rows,
        "scores": scores,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)


def write_predictions(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "rater", "judge", "target_image", "car_name", "dimension",
        "predicted", "true", "abs_error",
        "n_context", "n_images", "enable_thinking",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _prediction_rows(pred: Prediction, rater_id: str, truth: Dict[str, int],
                     car_name: str, enable_thinking: bool) -> List[dict]:
    rows = []
    for dim in DIMENSIONS:
        pv = pred.predicted.get(dim)
        tv = truth.get(dim)
        err = abs(pv - tv) if (pv is not None and tv is not None) else ""
        rows.append({
            "rater": rater_id,
            "judge": pred.judge,
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


def main(argv: List[str] | None = None) -> int:
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    run_start = time.time()

    print(f"[load] images from {cfg.image_root}")
    image_index = datamod.build_image_index(cfg.image_root)
    print(f"[load] {len(image_index)} car images indexed")

    car_names = datamod.load_car_names(cfg.car_name_mapping)

    print(f"[load] raters from {cfg.data_csv}")
    raters = datamod.load_raters(cfg.data_csv, min_cars=cfg.test_size + 1)
    print(f"[load] {len(raters)} usable raters "
          f"(need >= {cfg.test_size + 1} rated cars)")
    raters = raters[:cfg.n_raters]

    mode = "DRY RUN (no server)" if cfg.dry_run else f"LIVE -> {cfg.base_url} [{cfg.model}]"
    print(f"[run ] {mode}; judges={cfg.judges} test_size={cfg.test_size} "
          f"context_size={cfg.context_size} thinking={cfg.enable_thinking}")

    client = QwenVLMClient(cfg)
    judges = {name: make_judge(name, client, image_index) for name in cfg.judges}

    all_rows: List[dict] = []
    max_images_seen = 0
    # collected (true, pred) pairs per (judge, dimension) for a quick summary
    score_bucket: Dict[tuple, list] = defaultdict(list)

    for rater in raters:
        split = datamod.make_split(
            rater, test_size=cfg.test_size, split_seed=cfg.split_seed,
            include_anchors_in_pool=cfg.include_anchors_in_pool,
        )
        context_imgs = datamod.sample_context(split, cfg.context_size, draw_seed=cfg.split_seed)
        context = [(img, rater.ratings[img]) for img in context_imgs]
        print(f"\n[rater] {rater.response_id}: "
              f"{len(split.context_pool)} pool / {len(split.test_images)} held-out; "
              f"context={len(context)}")

        for target in split.test_images:
            truth = rater.ratings[target]
            car_name = car_names.get(target, "")
            for name, judge in judges.items():
                if name == "in_context":
                    pred = judge.predict(target, context)
                else:
                    pred = judge.predict(target)
                max_images_seen = max(max_images_seen, pred.n_images)

                if pred.from_cache:
                    print(f"  [hit] {name:10s} target={target} "
                          f"-> reused (already rated; 0-shot is rater-independent)")
                elif cfg.dry_run:
                    print(f"  [dry] {name:10s} target={target} "
                          f"images={pred.n_images} "
                          f"(system+user turns={len(pred.messages)})")
                else:
                    print(f"  [got] {name:10s} target={target} -> {pred.predicted}")

                if not cfg.dry_run:
                    for dim in DIMENSIONS:
                        pv, tv = pred.predicted.get(dim), truth.get(dim)
                        if pv is not None and tv is not None:
                            score_bucket[(name, dim)].append((tv, pv))

                all_rows.extend(
                    _prediction_rows(pred, rater.response_id, truth, car_name,
                                     cfg.enable_thinking))

    nc = judges.get("no_context")
    if nc is not None:
        pairs = nc.n_unique_cars + nc.n_cache_hits
        print(f"\n[cache] no_context: {nc.n_unique_cars} cars rated for "
              f"{pairs} rater-car pairs ({nc.n_cache_hits} repeats reused).")

    if cfg.dry_run:
        need = max_images_seen
        print(f"\n[dry ] max images in one request = {need}. "
              f"Serve with --limit-mm-per-prompt image={need} (or higher).")
        print("[dry ] no predictions written (dry run).")
        return 0

    if cfg.out_path is None:
        cfg.out_path = build_out_path(cfg)
    write_predictions(cfg.out_path, all_rows)
    duration = time.time() - run_start
    print(f"\n[save] {len(all_rows)} rows -> {cfg.out_path}")

    print("\n[score] quick summary (true vs predicted):")
    scores: Dict[str, Dict[str, dict]] = {}
    for name in cfg.judges:
        scores[name] = {}
        flat = []
        for dim in DIMENSIONS:
            recs = score_bucket.get((name, dim), [])
            flat += recs
            s = metricsmod.summarize(recs)
            scores[name][dim] = s
            print(f"  {name:10s} {dim:10s} n={s['n']:3d} "
                  f"MAE={s['mae']:.3f} within1={s['within1']:.3f}")
        s = metricsmod.summarize(flat)
        scores[name]["ALL"] = s
        print(f"  {name:10s} {'ALL':10s} n={s['n']:3d} "
              f"MAE={s['mae']:.3f} within1={s['within1']:.3f}")

    info_path = os.path.join(os.path.dirname(cfg.out_path), "run_info.json")
    write_run_info(info_path, cfg, [r.response_id for r in raters],
                   duration, len(all_rows), scores)
    print(f"\n[done] run took {duration:.1f}s; metadata -> {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
