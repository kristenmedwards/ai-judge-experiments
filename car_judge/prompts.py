"""System rubric and chat-message construction.

Message format (per experiment plan, section 9):
  - system : rating rubric + 1-6 scale definition + required JSON output
  - per exemplar : an image part (base64 data URI) FOLLOWED BY a text part
    giving that car's ratings  (image-before-text ordering, per Qin et al.)
  - target : the target image + a request for the five ratings as JSON
"""

from __future__ import annotations

import base64
import json
import os
from functools import lru_cache
from typing import Dict, List

from .config import DIMENSIONS, SCALE_MIN, SCALE_MAX

SYSTEM_RUBRIC = f"""You are shown an image of a car, presented as a clean isometric render. Answer the following two questions using an integer scale from {SCALE_MIN} to {SCALE_MAX}.

Based on the image shown, how well does this car fit each of the following descriptions? Rate each description from {SCALE_MIN} (Not at all) to {SCALE_MAX} (Very much):
- sporty
- luxurious
- modern
- rugged

Based on the image shown, how much do you like this car? Rate from {SCALE_MIN} (Dislike a great deal) to {SCALE_MAX} (Like a great deal); report this as "preference".

Respond with ONLY a JSON object, no prose, no markdown fences, exactly these keys:
{{"sporty": <int>, "luxurious": <int>, "modern": <int>, "rugged": <int>, "preference": <int>}}
Every value must be an integer from {SCALE_MIN} to {SCALE_MAX}."""

# Extra sentence appended when we are giving the model exemplars from one person.
PERSONALIZATION_NOTE = (
    "\n\nBelow are example cars that ONE specific person rated, each image "
    "followed by that person's ratings. Study their taste, then predict how "
    "that SAME person would rate the final target car."
)


@lru_cache(maxsize=4096)
def encode_image(path: str) -> str:
    """Return a base64 ``data:`` URI for an image file (cached)."""
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _image_part(data_uri: str) -> dict:
    return {"type": "image_url", "image_url": {"url": data_uri}}


def _text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def _ratings_text(ratings: Dict[str, int]) -> str:
    ordered = {k: ratings[k] for k in DIMENSIONS if k in ratings}
    return "This person's ratings for the car above: " + json.dumps(ordered)


def build_no_context_messages(target_path: str) -> List[dict]:
    """0-shot: rubric + target image + request."""
    return [
        {"role": "system", "content": SYSTEM_RUBRIC},
        {"role": "user", "content": [
            _text_part("Rate this car by answering both questions above. Respond with the JSON object only."),
            _image_part(encode_image(target_path)),
        ]},
    ]


def build_in_context_messages(
    exemplars: List[tuple],   # list of (image_path, ratings_dict)
    target_path: str,
) -> List[dict]:
    """Few-shot: rubric (+ personalization note) then, for each exemplar, an
    image part followed by its ratings text; finally the target image + request.
    All exemplars + target go in a single user turn so vLLM counts them as one
    multi-image prompt (remember ``--limit-mm-per-prompt image=N``)."""
    content: List[dict] = []
    for img_path, ratings in exemplars:
        content.append(_image_part(encode_image(img_path)))
        content.append(_text_part(_ratings_text(ratings)))
    content.append(_text_part(
        "Now predict how this SAME person would rate the following target car. "
        "Respond with the JSON object only."
    ))
    content.append(_image_part(encode_image(target_path)))

    return [
        {"role": "system", "content": SYSTEM_RUBRIC + PERSONALIZATION_NOTE},
        {"role": "user", "content": content},
    ]


def count_images(messages: List[dict]) -> int:
    """How many image parts a message list contains (for --limit-mm sizing)."""
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if p.get("type") == "image_url")
    return n
