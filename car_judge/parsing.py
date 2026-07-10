"""Robust parser turning a model's text response into five integer ratings."""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from .config import DIMENSIONS, SCALE_MIN, SCALE_MAX


class RatingParseError(ValueError):
    pass


def _clamp(v: int) -> int:
    return max(SCALE_MIN, min(SCALE_MAX, v))


def _extract_json_blob(text: str) -> Optional[dict]:
    """Find the first {...} block that parses as a JSON object."""
    # Strip common markdown fences first.
    cleaned = re.sub(r"```(?:json)?", "", text)
    # Scan for balanced brace spans, try to parse each.
    starts = [m.start() for m in re.finditer(r"\{", cleaned)]
    for s in starts:
        depth = 0
        for e in range(s, len(cleaned)):
            if cleaned[e] == "{":
                depth += 1
            elif cleaned[e] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[s:e + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
    return None


def _regex_fallback(text: str) -> Dict[str, int]:
    """Last resort: pull '<dim> : <int>' pairs out of free text."""
    out: Dict[str, int] = {}
    for dim in DIMENSIONS:
        m = re.search(rf'["\']?{dim}["\']?\s*[:=]\s*(\d+)', text, re.IGNORECASE)
        if m:
            out[dim] = _clamp(int(m.group(1)))
    return out


def parse_ratings(text: str, strict: bool = False) -> Dict[str, int]:
    """Parse model output into ``{dimension: int}`` for all five dimensions.

    Tries JSON first, then a per-dimension regex fallback. Values are clamped to
    the valid scale. Raises :class:`RatingParseError` if strict and any
    dimension is missing.
    """
    if text is None:
        text = ""

    result: Dict[str, int] = {}
    blob = _extract_json_blob(text)
    if blob:
        for dim in DIMENSIONS:
            if dim in blob:
                try:
                    result[dim] = _clamp(int(round(float(blob[dim]))))
                except (TypeError, ValueError):
                    pass

    if len(result) < len(DIMENSIONS):
        for dim, val in _regex_fallback(text).items():
            result.setdefault(dim, val)

    missing = [d for d in DIMENSIONS if d not in result]
    if missing and strict:
        raise RatingParseError(
            f"missing dimensions {missing} in model output: {text[:200]!r}"
        )
    return result
