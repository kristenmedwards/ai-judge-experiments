"""OUTER-LOOP DRIVER — one experiment round of the personalized-judge search.

This is the command you run once per experiment (the analogue of autoresearch's
``uv run train.py``). It:

  1. loads the current JudgeConfig (from ``judge_config.py`` or a --config file),
  2. runs the INNER loop (score it against real raters -> overall MAE),
  3. writes the per-prediction CSV, and
  4. appends one row to ``results.tsv``.

The OUTER loop itself is agent-driven (see ``judge_program.md``): you edit
``judge_config.py``, rerun this, and keep the edit (git commit) only if MAE
dropped, else revert (git reset). ``results.tsv`` is your experiment log.

Offline first (no key, no spend):
    python run_autojudge.py --data testdata/car_ratings_long.csv \
        --image-root testdata/images --mock --raters 5 --test-size 6

Live (after putting OPENAI_API_KEY + OPENAI_MODEL in .env):
    python run_autojudge.py --data ../car_ratings_long_..._Q23.csv \
        --image-root ../selected_2000_isometric_upload_chunks_renamed \
        --raters 25 --test-size 20
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import fields as _dc_fields
from typing import Optional

from car_judge import env as envmod
from car_judge.config import RunConfig, JudgeConfig
from car_judge.inner_loop import run_inner

RESULTS_TSV = "results.tsv"
RESULTS_HEADER = ["timestamp", "commit", "name", "config", "score_mae",
                  "within1", "n_raters", "n_preds", "unparsed", "status", "note"]


def load_judge_config(path: Optional[str]) -> JudgeConfig:
    """Load a JudgeConfig from judge_config.py (default) or a .json / .py file."""
    if path is None or path.endswith(".py"):
        # import the module and read CONFIG
        import importlib.util
        modpath = path or "judge_config.py"
        spec = importlib.util.spec_from_file_location("judge_config_dynamic", modpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        cfg = getattr(mod, "CONFIG")
        if not isinstance(cfg, JudgeConfig):
            raise TypeError(f"{modpath} CONFIG must be a JudgeConfig")
        return cfg
    if path.endswith(".json"):
        with open(path) as fh:
            d = json.load(fh)
        valid = {f.name for f in _dc_fields(JudgeConfig)}
        return JudgeConfig(**{k: v for k, v in d.items() if k in valid})
    raise ValueError("config must be a .py (with CONFIG) or .json file")


def git_commit_short() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def append_results_row(row: dict, path: str = RESULTS_TSV) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_HEADER, delimiter="\t")
        if new:
            w.writeheader()
        w.writerow(row)


def write_predictions(rows, path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_args(argv):
    p = argparse.ArgumentParser(description="Run one personalized-judge experiment round.")
    p.add_argument("--data", required=True, help="merged long csv (…_with_color_and_Q23.csv)")
    p.add_argument("--image-root", required=True, help="folder with chunk_XX/car_N.png")
    p.add_argument("--config", default=None, help="judge_config.py (default) or a .json/.py")

    # endpoint (else env / .env)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=4)

    # experiment plumbing
    p.add_argument("--raters", type=int, default=5)
    p.add_argument("--test-size", type=int, default=20)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--min-cars", type=int, default=None)

    # run mode
    p.add_argument("--mock", action="store_true", help="deterministic offline judge")
    p.add_argument("--dry-run", action="store_true", help="build requests, no server")
    p.add_argument("--out", default=None, help="predictions csv (default outputs/<name>.csv)")
    p.add_argument("--note", default="", help="free-text note for results.tsv")
    p.add_argument("--no-log", action="store_true", help="don't append to results.tsv")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    creds = envmod.resolve_credentials(a.model, a.base_url, a.api_key)
    judge_cfg = load_judge_config(a.config)

    run_cfg = RunConfig(
        data_csv=a.data, image_root=a.image_root,
        base_url=creds.base_url, api_key=creds.api_key, model=creds.model,
        temperature=judge_cfg.temperature or a.temperature,
        max_tokens=a.max_tokens, max_concurrency=a.concurrency,
        n_raters=a.raters, test_size=a.test_size, split_seed=a.split_seed,
        min_cars=a.min_cars, dry_run=a.dry_run, mock=a.mock,
    )
    if not (a.mock or a.dry_run) and not creds.is_live_ready:
        print("!! No OPENAI_API_KEY found. Put it in .env or export it, or use "
              "--mock / --dry-run for offline runs.", file=sys.stderr)
        return 2

    result = run_inner(run_cfg, judge_cfg)

    out = a.out or os.path.join("outputs", f"pred_{judge_cfg.name}.csv")
    write_predictions(result.rows, out)
    print(f"[save] {len(result.rows)} prediction rows -> {out}")

    if not a.no_log and not a.dry_run:
        append_results_row({
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit_short(),
            "name": judge_cfg.name,
            "config": json.dumps(judge_cfg.to_dict(), separators=(",", ":")),
            "score_mae": f"{result.score:.6f}",
            "within1": f"{result.within1:.4f}",
            "n_raters": result.n_raters,
            "n_preds": result.n_predictions,
            "unparsed": result.n_unparsed,
            "status": "pending",   # you set keep/discard after comparing to best
            "note": a.note or judge_cfg.summary(),
        })
        print(f"[log ] appended to {RESULTS_TSV} (set status keep/discard yourself).")
        print("[next] compare score_mae to the best so far; if lower, `git commit` "
              "the judge_config.py change (keep); else `git checkout judge_config.py` (revert).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
