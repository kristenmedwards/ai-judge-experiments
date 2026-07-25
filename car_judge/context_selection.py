"""Strategies for choosing which of a rater's cars become in-context exemplars.

All strategies share one signature::

    select(pool, target, n, *, rater, image_index, seed) -> List[str]

``pool`` is the rater's context pool (their rated cars minus the held-out set),
``target`` is the held-out car being predicted. Strategies never look at the
target's true ratings — only its image / metadata — so there is no leakage.

Strategies
----------
random       seeded uniform sample (baseline).
first        first ``n`` in sorted order (fully deterministic).
diverse      greedy max-diversity by car metadata (varied body styles/colors).
rag_profile  nearest cars to the target by metadata similarity (same body
             style / color / size). Cheap, no image decode.
rag_image    nearest cars to the target by a lightweight image feature
             (downsampled RGB). Falls back to rag_profile if Pillow is absent.

``rag_*`` returns the most-similar cars; ``context_order='similar_last'`` (in the
judge) then places the closest exemplar nearest the target.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Dict, List, Optional

_IMG_FEATURE_CACHE: Dict[str, Optional[tuple]] = {}


# --------------------------------------------------------------------------- #
# lightweight image feature (downsampled RGB), Pillow-optional
# --------------------------------------------------------------------------- #
def _image_feature(path: str, grid: int = 12) -> Optional[tuple]:
    if path in _IMG_FEATURE_CACHE:
        return _IMG_FEATURE_CACHE[path]
    feat = None
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB").resize((grid, grid))
            raw = im.tobytes()               # RGBRGB... bytes, avoids deprecated getdata
        flat = [b / 255.0 for b in raw]
        norm = sum(v * v for v in flat) ** 0.5 or 1.0
        feat = tuple(v / norm for v in flat)
    except Exception:
        feat = None
    _IMG_FEATURE_CACHE[path] = feat
    return feat


def _cosine(a: tuple, b: tuple) -> float:
    return sum(x * y for x, y in zip(a, b))


def _meta_similarity(a: dict, b: dict) -> float:
    """Crude metadata similarity in [0, ~4]: shared body_style/color + close size."""
    if not a or not b:
        return 0.0
    s = 0.0
    if a.get("body_style") and a.get("body_style") == b.get("body_style"):
        s += 2.0
    if a.get("car_color") and a.get("car_color") == b.get("car_color"):
        s += 1.0
    try:
        if a.get("seats") is not None and b.get("seats") is not None:
            s += max(0.0, 1.0 - abs(float(a["seats"]) - float(b["seats"])) / 4.0)
    except (TypeError, ValueError):
        pass
    return s


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #
def _random(pool, target, n, *, rater, image_index, seed):
    rng = random.Random(f"{rater.response_id}:ctx:{seed}")
    return rng.sample(pool, min(n, len(pool)))


def _first(pool, target, n, *, rater, image_index, seed):
    return sorted(pool)[:n]


def _diverse(pool, target, n, *, rater, image_index, seed):
    """Greedy: repeatedly add the car least similar (by metadata) to those chosen."""
    meta = getattr(rater, "car_meta", {})
    rng = random.Random(f"{rater.response_id}:div:{seed}")
    remaining = sorted(pool)
    if not remaining:
        return []
    chosen = [rng.choice(remaining)]
    remaining.remove(chosen[0])
    while remaining and len(chosen) < n:
        best, best_score = None, 1e9
        for c in remaining:
            worst = max(_meta_similarity(meta.get(c, {}), meta.get(k, {})) for k in chosen)
            if worst < best_score:
                best, best_score = c, worst
        chosen.append(best)
        remaining.remove(best)
    return chosen


def _rag_profile(pool, target, n, *, rater, image_index, seed):
    meta = getattr(rater, "car_meta", {})
    tmeta = meta.get(target, {})
    scored = sorted(pool, key=lambda c: -_meta_similarity(tmeta, meta.get(c, {})))
    return scored[:n]


def _rag_image(pool, target, n, *, rater, image_index, seed):
    tpath = image_index.get(target)
    tfeat = _image_feature(tpath) if tpath else None
    if tfeat is None:
        return _rag_profile(pool, target, n, rater=rater, image_index=image_index, seed=seed)
    scored = []
    for c in pool:
        p = image_index.get(c)
        f = _image_feature(p) if p else None
        scored.append((c, _cosine(tfeat, f) if f else -1.0))
    scored.sort(key=lambda t: -t[1])
    return [c for c, _ in scored[:n]]


def _rag_clip(pool, target, n, *, rater, image_index, seed):
    """Nearest cars by precomputed CLIP embedding (data/car_clip_embeddings.npz).

    Falls back to rag_image (pixel feature) for any car — or a whole run — that the
    CLIP index doesn't cover, so a partial index still works."""
    from .clip_index import load_default
    idx = load_default()
    if idx is None or not idx.has(target):
        return _rag_image(pool, target, n, rater=rater, image_index=image_index, seed=seed)
    sel = idx.nearest(target, pool, n)
    if len(sel) < n:  # top up from pixel-RAG if the CLIP index is partial
        extra = [c for c in _rag_image(pool, target, n, rater=rater,
                                       image_index=image_index, seed=seed) if c not in sel]
        sel = sel + extra[: n - len(sel)]
    return sel


_STRATEGIES = {
    "random": _random,
    "first": _first,
    "diverse": _diverse,
    "rag_profile": _rag_profile,
    "rag_image": _rag_image,
    "rag_clip": _rag_clip,
}


def select_context(
    strategy: str,
    pool: List[str],
    target: str,
    n: int,
    *,
    rater,
    image_index: Dict[str, str],
    seed: int = 0,
) -> List[str]:
    if n <= 0 or not pool:
        return []
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown context strategy {strategy!r}")
    return _STRATEGIES[strategy](
        pool, target, n, rater=rater, image_index=image_index, seed=seed
    )
