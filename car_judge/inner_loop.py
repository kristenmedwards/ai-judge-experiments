"""The INNER loop: score one JudgeConfig against real raters.

For each rater a fixed held-out test set of ``test_size`` cars is reserved (seed
stable, so every JudgeConfig is scored on identical items). The judge predicts the
five ratings for each held-out car using only that rater's OTHER cars as context
plus whatever side-info the config enables. We then aggregate error.

Primary score = **overall MAE**, macro-averaged so every rater counts equally:
per rater, mean |pred-true| over their held-out (car x dimension) cells; then the
mean of that across raters. Lower is better — this is the value the outer loop
tries to reduce (the analogue of autoresearch's val_bpb).

Also reported: per-dimension MAE, within-1 accuracy, exact accuracy, a micro
(pooled) MAE, and the number of unparseable predictions.
"""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import data as datamod
from .config import RunConfig, JudgeConfig, DIMENSIONS
from .client import VLMClient
from .long_data import load_long_raters, LongRater
from .personalized_judge import PersonalizedJudge, Prediction


# --------------------------------------------------------------------------- #
# attention screen (same as the analysis pipeline)
# --------------------------------------------------------------------------- #
def passes_attention(rater: LongRater) -> bool:
    p = rater.profile
    ai = str(p.get("attn_check_ai", "")).strip()
    color = str(p.get("attn_check_color", "")).strip().upper()
    ai_ok = ai in ("4", "4.0")
    color_ok = (color == "A") or (color == "")   # tolerate missing color check
    return ai_ok and color_ok


@dataclass
class InnerResult:
    score: float                        # overall MAE (macro over raters) — LOWER better
    per_dimension_mae: Dict[str, float]
    within1: float
    exact: float
    micro_mae: float
    n_raters: int
    n_predictions: int
    n_unparsed: int
    rows: List[dict] = field(default_factory=list)  # tidy per (rater,car,dim) rows

    def summary_line(self) -> str:
        pd_ = " ".join(f"{d[:4]}={self.per_dimension_mae[d]:.3f}" for d in DIMENSIONS)
        return (f"MAE={self.score:.4f} | {pd_} | within1={self.within1:.3f} "
                f"| raters={self.n_raters} preds={self.n_predictions} "
                f"unparsed={self.n_unparsed}")


def _prediction_rows(pred: Prediction, rid: str, truth: Dict[str, int],
                     jc: JudgeConfig) -> List[dict]:
    rows = []
    for dim in DIMENSIONS:
        pv, tv = pred.predicted.get(dim), truth.get(dim)
        err = abs(pv - tv) if (pv is not None and tv is not None) else None
        rows.append({
            "rater": rid, "judge": jc.name, "target_image": pred.target_image,
            "dimension": dim, "predicted": pv, "true": tv, "abs_error": err,
            "n_context": pred.n_context, "n_images": pred.n_images,
            "context_strategy": jc.context_strategy,
            "include_demographics": jc.include_demographics,
            "include_q23": jc.include_q23,
        })
    return rows


def run_inner(run_cfg: RunConfig, judge_cfg: JudgeConfig,
              verbose: bool = True) -> InnerResult:
    judge_cfg.validate()
    image_index = datamod.build_image_index(run_cfg.image_root)

    min_cars = run_cfg.min_cars or (run_cfg.test_size + 1)
    raters = load_long_raters(run_cfg.data_csv, min_cars=1)
    raters = [r for r in raters if passes_attention(r)]
    # Robustness: keep only cars whose image is actually present, then require the
    # rater still has enough fully-rated cars for the split. Count NON-ANCHOR cars:
    # make_split excludes anchors from the test/context pool by default, so a rater
    # with 21 fully-rated cars but 4 anchors has only 17 usable and would crash
    # make_split at test_size=20 (bit at the full-pool eval; invisible at n=150).
    kept = []
    for r in raters:
        r.ratings = {img: v for img, v in r.ratings.items() if img in image_index}
        r.anchor_images = [a for a in r.anchor_images if a in image_index]
        usable = r.fully_rated_cars()
        if not run_cfg.include_anchors_in_pool:
            anchors = set(r.anchor_images)
            usable = [c for c in usable if c not in anchors]
        if len(usable) >= min_cars:
            kept.append(r)
    raters = kept
    if run_cfg.n_raters:
        raters = raters[:run_cfg.n_raters]
    if verbose:
        mode = ("MOCK" if run_cfg.mock else "DRY-RUN" if run_cfg.dry_run
                else f"LIVE[{run_cfg.model}]")
        print(f"[inner] {mode} judge='{judge_cfg.name}' ({judge_cfg.summary()}) "
              f"raters={len(raters)} test_size={run_cfg.test_size}")

    client = VLMClient(run_cfg)
    judge = PersonalizedJudge(judge_cfg, client, image_index)

    # build the full task list first (rater, target, pool)
    tasks: List[Tuple[LongRater, str, List[str]]] = []
    for rater in raters:
        split = datamod.make_split(
            rater, test_size=run_cfg.test_size, split_seed=run_cfg.split_seed,
            include_anchors_in_pool=run_cfg.include_anchors_in_pool,
        )
        for target in split.test_images:
            tasks.append((rater, target, split.context_pool))

    def _do(task):
        rater, target, pool = task
        pred = judge.predict(rater, target, pool, seed=run_cfg.split_seed)
        return rater, target, pred

    results = []
    if run_cfg.dry_run:
        # no server calls; still build every request to size images / inspect
        for t in tasks:
            results.append(_do(t))
    else:
        workers = max(1, run_cfg.max_concurrency)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_do, t) for t in tasks]
            for f in as_completed(futs):
                results.append(f.result())

    # ---- aggregate ----
    rows: List[dict] = []
    per_rater_abs: Dict[str, List[float]] = {}
    dim_abs: Dict[str, List[float]] = {d: [] for d in DIMENSIONS}
    all_abs: List[float] = []
    within1_hits = exact_hits = graded = 0
    n_unparsed = 0
    max_images = 0

    for rater, target, pred in results:
        max_images = max(max_images, pred.n_images)
        truth = rater.ratings[target]
        rows.extend(_prediction_rows(pred, rater.response_id, truth, judge_cfg))
        if pred.dry_run:
            continue
        if not pred.predicted:
            n_unparsed += 1
        for dim in DIMENSIONS:
            pv, tv = pred.predicted.get(dim), truth.get(dim)
            if pv is None or tv is None:
                continue
            e = abs(pv - tv)
            per_rater_abs.setdefault(rater.response_id, []).append(e)
            dim_abs[dim].append(e)
            all_abs.append(e)
            graded += 1
            within1_hits += (e <= 1)
            exact_hits += (e == 0)

    if run_cfg.dry_run:
        if verbose:
            print(f"[inner] DRY-RUN built {len(tasks)} requests; "
                  f"max images/request = {max_images}")
        return InnerResult(float("nan"), {d: float("nan") for d in DIMENSIONS},
                           float("nan"), float("nan"), float("nan"),
                           len(raters), len(tasks), 0, rows)

    per_rater_mae = [statistics.mean(v) for v in per_rater_abs.values() if v]
    score = statistics.mean(per_rater_mae) if per_rater_mae else float("nan")
    per_dim = {d: (statistics.mean(dim_abs[d]) if dim_abs[d] else float("nan"))
               for d in DIMENSIONS}
    result = InnerResult(
        score=score,
        per_dimension_mae=per_dim,
        within1=(within1_hits / graded) if graded else float("nan"),
        exact=(exact_hits / graded) if graded else float("nan"),
        micro_mae=(statistics.mean(all_abs) if all_abs else float("nan")),
        n_raters=len(raters),
        n_predictions=len(tasks),
        n_unparsed=n_unparsed,
        rows=rows,
    )
    if verbose:
        print("[inner] " + result.summary_line())
    return result
