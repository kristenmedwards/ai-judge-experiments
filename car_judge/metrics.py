"""Lightweight scoring for predicted vs. true ratings (pure python)."""

from __future__ import annotations

from typing import Dict, List, Tuple


def _pairs(records: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(int(t), int(p)) for t, p in records if t is not None and p is not None]


def mae(records: List[Tuple[int, int]]) -> float:
    pairs = _pairs(records)
    if not pairs:
        return float("nan")
    return sum(abs(t - p) for t, p in pairs) / len(pairs)


def exact_accuracy(records: List[Tuple[int, int]]) -> float:
    pairs = _pairs(records)
    if not pairs:
        return float("nan")
    return sum(1 for t, p in pairs if t == p) / len(pairs)


def within_one(records: List[Tuple[int, int]]) -> float:
    pairs = _pairs(records)
    if not pairs:
        return float("nan")
    return sum(1 for t, p in pairs if abs(t - p) <= 1) / len(pairs)


def summarize(records: List[Tuple[int, int]]) -> Dict[str, float]:
    return {
        "n": len(_pairs(records)),
        "mae": mae(records),
        "exact": exact_accuracy(records),
        "within1": within_one(records),
    }
