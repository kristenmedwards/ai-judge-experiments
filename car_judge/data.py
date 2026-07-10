"""Load the Prolific Qualtrics export and index the car images.

The export is tricky (see the experiment plan, section 3): Qualtrics split each
rater's 30 ratings across four question blocks (Q25/Q26, Q28/Q29, Q20/Q21,
Q30/Q31) with loop-merge prefixes, and there are three header rows before data.

Rather than key on fragile block/loop numbers, we read the **question text**
(header row 2), which for every rating column embeds the image filename and the
dimension, e.g.:

    "... how well does this car fit ...? - car_1.png - Sporty"
    "... how much do you like this car? - car_1.png - How much do you like this car?"

That makes the loader robust to the four-block layout and to the anchor repeats.
"""

from __future__ import annotations

import csv
import os
import re
import random
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional

from .config import DIMENSIONS

# Map the wording in the header text to a canonical dimension key.
_DIM_PATTERNS = {
    "sporty": "sporty",
    "luxurious": "luxurious",
    "modern": "modern",
    "rugged": "rugged",
    "how much do you like this car?": "preference",
}

# Captures "- car_123.png - <dimension text>" at the end of a header cell.
_HEADER_RX = re.compile(
    r"-\s*(car_\d+\.png)\s*-\s*"
    r"(Sporty|Luxurious|Modern|Rugged|How much do you like this car\?)\s*$",
    re.IGNORECASE,
)

# Fixed "anchor" cars are shown to every rater via dedicated anchor question
# blocks (Q28/Q29 and Q30/Q31). In this export those are 4 cars rated by all
# participants; the remaining ~26 cars per rater are assigned uniquely via
# loop-merge. Anchors give the inter-rater agreement / ceiling point (plan §5),
# so by default we keep them out of each rater's context pool and held-out set.
_ANCHOR_BLOCK_RX = re.compile(r"^\d+_Q(?:28|29|30|31)_\d+$")


@dataclass
class Rater:
    """One survey respondent and their car ratings."""

    response_id: str
    # image filename -> {dimension: int rating}
    ratings: Dict[str, Dict[str, int]] = field(default_factory=dict)
    anchor_images: List[str] = field(default_factory=list)

    def fully_rated_cars(self) -> List[str]:
        """Cars for which all five dimensions are present."""
        return [img for img, d in self.ratings.items()
                if all(k in d for k in DIMENSIONS)]


# --------------------------------------------------------------------------- #
# Image index
# --------------------------------------------------------------------------- #
def build_image_index(image_root: str) -> Dict[str, str]:
    """Map every ``car_N.png`` basename to its absolute path.

    Images live in ``<image_root>/chunk_XX/car_N.png`` and filenames are globally
    unique (car_1 .. car_2000), so a flat basename index is unambiguous.
    """
    index: Dict[str, str] = {}
    for path in glob(os.path.join(image_root, "**", "*.png"), recursive=True):
        base = os.path.basename(path)
        if re.fullmatch(r"car_\d+\.png", base):
            index.setdefault(base, os.path.abspath(path))
    if not index:
        raise FileNotFoundError(
            f"No car_*.png images found under {image_root!r}. "
            "Point --image-root at the folder containing chunk_XX/ subfolders."
        )
    return index


# --------------------------------------------------------------------------- #
# Ratings loader
# --------------------------------------------------------------------------- #
def _rating_column_map(
    header_names: List[str], question_text_row: List[str]
) -> Dict[int, tuple]:
    """col index -> (image_filename, dimension, is_anchor) for every rating column."""
    colmap: Dict[int, tuple] = {}
    for i, text in enumerate(question_text_row):
        m = _HEADER_RX.search(text.strip())
        if not m:
            continue
        image = m.group(1)
        dim = _DIM_PATTERNS[m.group(2).strip().lower()]
        is_anchor = (
            bool(_ANCHOR_BLOCK_RX.match(header_names[i]))
            if i < len(header_names) else False
        )
        colmap[i] = (image, dim, is_anchor)
    return colmap


def _coerce_rating(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(round(float(value)))
    except ValueError:
        return None


def load_raters(data_csv: str, min_cars: int = 1) -> List[Rater]:
    """Parse the Prolific export into a list of :class:`Rater`.

    Rows with fewer than ``min_cars`` rated cars (e.g. blank/preview responses)
    are skipped.
    """
    with open(data_csv, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))

    if len(rows) < 4:
        raise ValueError("export has fewer than the expected 3 header rows + data")

    header_names, question_text, _import_ids = rows[0], rows[1], rows[2]
    data_rows = rows[3:]

    colmap = _rating_column_map(header_names, question_text)
    if not colmap:
        raise ValueError("no rating columns matched; is this the Prolific export?")

    try:
        rid_idx = header_names.index("ResponseId")
    except ValueError:
        rid_idx = header_names.index("ResponseID") if "ResponseID" in header_names else 8

    # Images that come from the fixed anchor blocks (shown to every rater).
    anchor_images_global = sorted({img for _, (img, _d, anch) in colmap.items() if anch})

    raters: List[Rater] = []
    for row in data_rows:
        if rid_idx >= len(row):
            continue
        ratings: Dict[str, Dict[str, int]] = {}
        for i, (img, dim, _anch) in colmap.items():
            if i >= len(row):
                continue
            val = _coerce_rating(row[i])
            if val is None:
                continue
            ratings.setdefault(img, {})[dim] = val
        if len(ratings) >= min_cars:
            raters.append(Rater(
                response_id=row[rid_idx],
                ratings=ratings,
                anchor_images=[a for a in anchor_images_global if a in ratings],
            ))
    return raters


def load_car_names(mapping_csv: Optional[str]) -> Dict[str, str]:
    """Optional: map ``car_N.png`` -> human-readable car name for logging."""
    names: Dict[str, str] = {}
    if not mapping_csv or not os.path.exists(mapping_csv):
        return names
    with open(mapping_csv, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            fn = r.get("upload_filename") or ""
            nm = r.get("actual_car_name") or ""
            if fn:
                names[fn] = nm
    return names


# --------------------------------------------------------------------------- #
# Held-out / context split
# --------------------------------------------------------------------------- #
@dataclass
class RaterSplit:
    rater: Rater
    test_images: List[str]      # fixed held-out targets
    context_pool: List[str]     # cars context may be drawn from


def make_split(
    rater: Rater,
    test_size: int,
    split_seed: int = 0,
    include_anchors_in_pool: bool = False,
) -> RaterSplit:
    """Reserve a fixed held-out test set; everything else is the context pool.

    Only fully-rated cars are used so both context labels and test targets have
    all five dimensions. The split is deterministic given ``split_seed`` so the
    test set stays fixed across context sizes (see plan section 4).
    """
    cars = rater.fully_rated_cars()
    if not include_anchors_in_pool:
        cars = [c for c in cars if c not in rater.anchor_images]
    cars = sorted(cars)  # stable order before seeded shuffle

    if test_size >= len(cars):
        raise ValueError(
            f"rater {rater.response_id}: test_size={test_size} but only "
            f"{len(cars)} fully-rated non-anchor cars available"
        )

    rng = random.Random(f"{rater.response_id}:{split_seed}")
    shuffled = cars[:]
    rng.shuffle(shuffled)
    test_images = sorted(shuffled[:test_size])
    context_pool = sorted(shuffled[test_size:])
    return RaterSplit(rater=rater, test_images=test_images, context_pool=context_pool)


def sample_context(
    split: RaterSplit,
    context_size: int,
    draw_seed: int = 0,
) -> List[str]:
    """Draw ``context_size`` exemplar cars from the pool (image-order randomized).

    Returns an empty list for a 0-shot judge. Averaging over multiple draw_seeds
    separates the ICL effect from context-sampling noise (plan section 4).
    """
    if context_size <= 0:
        return []
    pool = split.context_pool
    k = min(context_size, len(pool))
    rng = random.Random(f"{split.rater.response_id}:ctx:{draw_seed}")
    return rng.sample(pool, k)
