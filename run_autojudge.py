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

# --- per-run output namespacing -------------------------------------------- #
# Every sweep/run writes its artifacts under outputs/runs/<run_id>/ so nothing
# from a previous run is ever overwritten. The top-level results.tsv stays a
# single append-only master log (each row's `note` carries the run_id).
RUNS_ROOT = os.path.join("outputs", "runs")


def make_run_id(tag: Optional[str] = None) -> str:
    """A unique id for one run: local timestamp, plus an optional --run-tag."""
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (tag or "").strip().replace(os.sep, "-").replace(" ", "-")
    return f"{stamp}_{tag}" if tag else stamp


def run_dir_for(run_id: str) -> str:
    """Create and return outputs/runs/<run_id>/ (with its per_experiment/ subdir)."""
    d = os.path.join(RUNS_ROOT, run_id)
    os.makedirs(os.path.join(d, "per_experiment"), exist_ok=True)
    return d


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


# --- richer metric suite (beyond MAE) -------------------------------------- #
# MAE stays the primary optimization target (inner_loop.py, untouched). This is
# ADDITIONAL reporting: ICC(2,1), Spearman rho, quadratic-weighted kappa and RMSE
# — computed with the SAME functions the paper's evaluation module uses
# (evaluation/stats.py), so numbers are directly comparable. Correlation metrics
# are far less sensitive than MAE to which held-out cars a seed happened to draw,
# so they compare more cleanly across seeds.
def compute_extra_metrics(rows) -> dict:
    """Per-dimension + pooled ICC/Spearman/kappa/RMSE from per-cell prediction rows.

    Returns {} if numpy/scipy or evaluation.stats are unavailable (never fatal)."""
    try:
        import numpy as np
        from evaluation import stats as st
    except Exception:
        return {}
    graded = [r for r in rows if r.get("predicted") is not None
              and r.get("true") is not None]
    if not graded:
        return {}

    def suite(sub):
        t = np.array([r["true"] for r in sub], dtype=float)
        p = np.array([r["predicted"] for r in sub], dtype=float)
        rho, _ = st.spearman(t, p)
        return {
            "n": int(len(sub)), "mae": st.mae(t, p), "rmse": st.rmse(t, p),
            "within1": st.within_k(t, p, 1), "icc2_1": st.icc2_1(np.column_stack([t, p])),
            "spearman_rho": rho,
            "weighted_kappa": st.quadratic_weighted_kappa(t, p),
        }

    dims = sorted({r["dimension"] for r in graded})
    per_dim = {d: suite([r for r in graded if r["dimension"] == d]) for d in dims}

    # within-rater mean Spearman: does the judge rank each PERSON'S own cars like
    # they do? (the personalization-quality metric MAE can't see)
    import numpy as np
    per_rater_rho = []
    raters = {}
    for r in graded:
        raters.setdefault(r["rater"], []).append(r)
    for rid, rs in raters.items():
        t = np.array([r["true"] for r in rs], dtype=float)
        p = np.array([r["predicted"] for r in rs], dtype=float)
        rho, _ = st.spearman(t, p)
        if rho == rho:  # not nan
            per_rater_rho.append(rho)
    wr = float(np.mean(per_rater_rho)) if per_rater_rho else float("nan")

    out = suite(graded)                       # pooled over all cells
    out["within_rater_spearman"] = wr
    out["per_dimension"] = per_dim
    return out


def _fmt_extra(xm: dict) -> str:
    if not xm:
        return ""
    return (f"ICC={xm.get('icc2_1', float('nan')):.3f} "
            f"rho={xm.get('spearman_rho', float('nan')):.3f} "
            f"wkappa={xm.get('weighted_kappa', float('nan')):.3f} "
            f"rmse={xm.get('rmse', float('nan')):.3f} "
            f"inRaterRho={xm.get('within_rater_spearman', float('nan')):.3f}")


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
    p.add_argument("--max-retries", type=int, default=4,
                   help="per-call API retries; raise for long unattended runs "
                        "so one transient burst can't kill a 3000-call round")

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
    p.add_argument("--run-tag", default=None,
                   help="if set, write this round under outputs/runs/<ts>_<tag>/ "
                        "(nothing in outputs/ is overwritten) and stamp the run id "
                        "into the results.tsv note")
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
        max_retries=a.max_retries,
        n_raters=a.raters, test_size=a.test_size, split_seed=a.split_seed,
        min_cars=a.min_cars, dry_run=a.dry_run, mock=a.mock,
    )
    if not (a.mock or a.dry_run) and not creds.is_live_ready:
        print("!! No OPENAI_API_KEY found. Put it in .env or export it, or use "
              "--mock / --dry-run for offline runs.", file=sys.stderr)
        return 2

    result = run_inner(run_cfg, judge_cfg)

    # Optional per-run namespacing: --run-tag routes outputs under
    # outputs/runs/<run_id>/ so a re-run never clobbers a previous round.
    run_id = make_run_id(a.run_tag) if a.run_tag is not None else None
    run_dir = run_dir_for(run_id) if run_id else None
    default_pred = (os.path.join(run_dir, "per_experiment", f"{judge_cfg.name}.csv")
                    if run_dir else os.path.join("outputs", f"pred_{judge_cfg.name}.csv"))
    out = a.out or default_pred
    write_predictions(result.rows, out)
    print(f"[save] {len(result.rows)} prediction rows -> {out}")

    # richer metric suite (MAE is still the primary target; these are extra views)
    xm = compute_extra_metrics(result.rows) if not a.dry_run else {}
    if xm:
        print(f"[eval] MAE={result.score:.4f} {_fmt_extra(xm)}")
        metrics_path = os.path.splitext(out)[0] + ".metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump({"name": judge_cfg.name, "split_seed": a.split_seed,
                       "macro_mae": result.score, **xm}, fh, indent=2)
        print(f"[eval] full metric suite -> {metrics_path}")

    if not a.no_log and not a.dry_run:
        note = a.note or judge_cfg.summary()
        if xm:
            note = f"{note} · {_fmt_extra(xm)}"
        if run_id:
            note = f"[run={run_id}] {note}"
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
            "status": "pending",   # you set keep/discard after comparing to best
            "note": note,
        }
        append_results_row(row)                       # append-only master log
        if run_dir:                                   # + a per-run copy
            append_results_row(row, path=os.path.join(run_dir, "results.tsv"))
        print(f"[log ] appended to {RESULTS_TSV} (set status keep/discard yourself).")
        print("[next] compare score_mae to the best so far; if lower, `git commit` "
              "the judge_config.py change (keep); else `git checkout judge_config.py` (revert).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
