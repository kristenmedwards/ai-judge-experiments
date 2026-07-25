"""Loader for the merged tidy LONG csv (one row per rater x car).

This is the richer data source used by the personalization harness:
``car_ratings_long_July8_July22_merged_with_color_and_Q23.csv``. Unlike the raw
Qualtrics export loader in ``data.py``, every row already carries the rater's
demographics, Big Five, hobbies, owned brands/body-styles, and the Q23
free-text ("what influenced your ratings"), so a judge can be personalized.

A ``LongRater`` duck-types ``data.Rater`` (``response_id`` / ``ratings`` /
``anchor_images`` / ``fully_rated_cars()``) so ``data.make_split`` and
``data.sample_context`` work unchanged — and it adds ``.profile`` and
``.car_meta`` for personalization and metadata-based context selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .config import DIMENSIONS

# long-csv column -> canonical dimension key
_METRIC_COLS = {
    "sporty": "sporty",
    "luxurious": "luxurious",
    "modern": "modern",
    "rugged": "rugged",
    "overall_preference": "preference",
}

# columns that are per-rating, not per-person (excluded from the profile)
_NON_PROFILE = set(_METRIC_COLS) | {
    "wave", "rater", "Q19", "car", "is_anchor", "car_name",
    "body_style", "doors", "seats", "car_color",
}

# per-car metadata columns worth keeping for context selection / RAG
_CAR_META_COLS = ["car_name", "body_style", "doors", "seats", "car_color"]

Q23_COL = "rating_influences_text"


@dataclass
class LongRater:
    response_id: str
    ratings: Dict[str, Dict[str, int]] = field(default_factory=dict)   # img -> {dim:int}
    anchor_images: List[str] = field(default_factory=list)
    profile: Dict[str, object] = field(default_factory=dict)          # demographics etc.
    car_meta: Dict[str, Dict[str, object]] = field(default_factory=dict)  # img -> meta

    def fully_rated_cars(self) -> List[str]:
        return [img for img, d in self.ratings.items()
                if all(k in d for k in DIMENSIONS)]

    @property
    def q23(self) -> str:
        return str(self.profile.get(Q23_COL) or "").strip()


def _img_name(car_id: object) -> Optional[str]:
    """'car_1009' (or 'car_1009.png') -> 'car_1009.png'."""
    if car_id is None or (isinstance(car_id, float) and pd.isna(car_id)):
        return None
    s = str(car_id).strip()
    if not s:
        return None
    return s if s.endswith(".png") else f"{s}.png"


def _coerce_rating(v) -> Optional[int]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def load_long_raters(csv_path: str, min_cars: int = 1) -> List[LongRater]:
    """Parse the merged long csv into personalizable raters.

    A car counts only if all five dimensions are present in its row. Raters with
    fewer than ``min_cars`` fully-rated cars are dropped.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in _METRIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"long csv missing metric columns: {missing}")

    is_anchor = (df["is_anchor"].astype(str).str.lower().isin(["true", "1"])
                 if "is_anchor" in df.columns else pd.Series(False, index=df.index))

    raters: List[LongRater] = []
    for rid, sub in df.groupby("rater", sort=False):
        ratings: Dict[str, Dict[str, int]] = {}
        anchors: List[str] = []
        car_meta: Dict[str, Dict[str, object]] = {}
        for idx, row in sub.iterrows():
            img = _img_name(row.get("car"))
            if img is None:
                continue
            dims = {}
            for col, dim in _METRIC_COLS.items():
                val = _coerce_rating(row.get(col))
                if val is not None:
                    dims[dim] = val
            if not dims:
                continue
            ratings.setdefault(img, {}).update(dims)
            if bool(is_anchor.loc[idx]) and img not in anchors:
                anchors.append(img)
            car_meta.setdefault(img, {k: row.get(k) for k in _CAR_META_COLS if k in sub.columns})

        # profile = first non-null value per person-level column
        profile: Dict[str, object] = {}
        for col in sub.columns:
            if col in _NON_PROFILE:
                continue
            s = sub[col].dropna()
            if len(s):
                profile[col] = s.iloc[0]

        rater = LongRater(response_id=str(rid), ratings=ratings,
                          anchor_images=anchors, profile=profile, car_meta=car_meta)
        if len(rater.fully_rated_cars()) >= min_cars:
            raters.append(rater)
    return raters


# --------------------------------------------------------------------------- #
# Profile -> prompt text
# --------------------------------------------------------------------------- #
_HOBBY_PREFIX = "hobby_"
_BRAND_PREFIX = "brand_owned_"
_BODY_OWNED_PREFIX = "body_style_owned_"
_PRETTY = {
    "prolific_age": "age", "prolific_sex": "sex",
    "num_children_under18": "children under 18",
    "driving_location": "driving location", "driving_frequency": "driving frequency",
    "bfi_openness": "openness", "bfi_conscientiousness": "conscientiousness",
    "bfi_extraversion": "extraversion", "bfi_agreeableness": "agreeableness",
    "bfi_neuroticism": "neuroticism",
    "prolific_household_income_us_participants_only": "household income",
    "married_or_partner": "married/partnered",
}


def _fmt_val(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def render_profile(profile: Dict[str, object], fields: List[str]) -> str:
    """Render selected demographic fields as a compact 'key: value' block."""
    lines = []
    for f in fields:
        if f in profile and profile[f] not in (None, ""):
            lines.append(f"- {_PRETTY.get(f, f)}: {_fmt_val(profile[f])}")
    return "\n".join(lines)


def render_owned(profile: Dict[str, object]) -> str:
    """One line each for brands owned and body styles owned (one-hot => list)."""
    brands = [k[len(_BRAND_PREFIX):].replace("_", " ")
              for k, v in profile.items()
              if k.startswith(_BRAND_PREFIX) and not k.endswith(("_text", "specify"))
              and str(v) in ("1", "1.0", "True", "true")]
    bodies = [k[len(_BODY_OWNED_PREFIX):].replace("_", " ")
              for k, v in profile.items()
              if k.startswith(_BODY_OWNED_PREFIX)
              and str(v) in ("1", "1.0", "True", "true")]
    hobbies = [k[len(_HOBBY_PREFIX):].replace("_", " ")
               for k, v in profile.items()
               if k.startswith(_HOBBY_PREFIX) and not k.endswith(("_text", "specify"))
               and str(v) in ("1", "1.0", "True", "true")]
    out = []
    if brands:
        out.append("- brands owned/driven: " + ", ".join(sorted(brands)))
    if bodies:
        out.append("- body styles owned: " + ", ".join(sorted(bodies)))
    if hobbies:
        out.append("- hobbies: " + ", ".join(sorted(hobbies)))
    return "\n".join(out)
