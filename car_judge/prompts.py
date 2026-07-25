"""System rubric and chat-message construction.

Message format (per experiment plan, section 9):
  - system : rating rubric + 1-6 scale + required JSON output (+ optional persona
    block describing the specific person, when personalization is enabled)
  - per exemplar : an image part FOLLOWED BY a text part with that car's ratings
    (image-before-text ordering, per Qin et al.)
  - target : the target image + a request for the five ratings as JSON

``build_messages`` is the config-driven entry point used by the harness; the
original ``build_no_context_messages`` / ``build_in_context_messages`` remain for
back-compat with run_experiment.py / run_context_sweep.py.
"""

from __future__ import annotations

import base64
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from .config import DIMENSIONS, SCALE_MIN, SCALE_MAX

SYSTEM_RUBRIC = f"""You are shown an image of a hypothetical car, presented as an isometric render. Answer the following two questions using an integer scale from {SCALE_MIN} to {SCALE_MAX}.

Based on the image shown, how well does this car fit each of the following descriptions? Rate each description from {SCALE_MIN} (Not at all) to {SCALE_MAX} (Very much):
- sporty
- luxurious
- modern
- rugged

Based on the image shown, how much do you like this car? Rate from {SCALE_MIN} (Dislike a great deal) to {SCALE_MAX} (Like a great deal); report this as "preference".

Respond with ONLY a JSON object, no prose, no markdown fences, exactly these keys:
{{"sporty": <int>, "luxurious": <int>, "modern": <int>, "rugged": <int>, "preference": <int>}}
Every value must be an integer from {SCALE_MIN} to {SCALE_MAX}."""

# A tighter wording variant (outer-loop lever: prompt_variant).
SYSTEM_CONCISE = f"""Rate each car's visual design on five 1-{SCALE_MAX} integer scales: sporty, luxurious, modern, rugged, and preference (how much you like it). {SCALE_MIN}=lowest, {SCALE_MAX}=highest.
Output ONLY this JSON: {{"sporty": <int>, "luxurious": <int>, "modern": <int>, "rugged": <int>, "preference": <int>}}."""

# A persona-forward variant that leans harder on matching the specific person.
SYSTEM_PERSONA = SYSTEM_RUBRIC + (
    "\n\nYou are not giving your own opinion. You are predicting, as precisely as "
    "possible, the exact 1-6 numbers ONE particular person would enter — including "
    "their personal quirks, leniency, and what they care about."
)

SYSTEM_VARIANTS = {
    "default": SYSTEM_RUBRIC,
    "concise": SYSTEM_CONCISE,
    "persona": SYSTEM_PERSONA,
}

PERSONALIZATION_NOTE = (
    "\n\nBelow are example cars that ONE specific person rated, each image "
    "followed by that person's ratings. Study their taste, then predict how "
    "that SAME person would rate the final target car."
)


@lru_cache(maxsize=8192)
def encode_image(path: str) -> str:
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


# --------------------------------------------------------------------------- #
# original builders (kept for back-compat)
# --------------------------------------------------------------------------- #
def build_no_context_messages(target_path: str) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_RUBRIC},
        {"role": "user", "content": [
            _text_part("Rate this car by answering both questions above. Respond with the JSON object only."),
            _image_part(encode_image(target_path)),
        ]},
    ]


def build_in_context_messages(exemplars: List[tuple], target_path: str) -> List[dict]:
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


# --------------------------------------------------------------------------- #
# config-driven builder used by the harness
# --------------------------------------------------------------------------- #
def build_persona_block(profile_text: str = "", q23_text: str = "",
                        owned_text: str = "") -> str:
    """Assemble the 'about this person' block from whichever pieces are enabled."""
    parts = []
    if profile_text:
        parts.append("About the person you are predicting:\n" + profile_text)
    if owned_text:
        parts.append(owned_text)
    if q23_text:
        parts.append('In their own words, what they said they pay attention to when '
                     f'rating cars: "{q23_text}"')
    return "\n".join(parts)


def build_messages(
    cfg,                                   # JudgeConfig
    target_path: str,
    exemplars: Optional[List[tuple]] = None,   # [(image_path, ratings_dict)]
    profile_text: str = "",
    q23_text: str = "",
    owned_text: str = "",
) -> List[dict]:
    """Build a chat request from a JudgeConfig plus already-selected exemplars and
    already-rendered person text. Exemplar/target image parts respect
    ``cfg.image_before_text``."""
    exemplars = exemplars or []
    system = SYSTEM_VARIANTS.get(cfg.prompt_variant, SYSTEM_RUBRIC)
    if exemplars:
        system = system + PERSONALIZATION_NOTE

    persona = build_persona_block(profile_text, q23_text, owned_text)
    if persona:
        system = system + "\n\n" + persona
    if getattr(cfg, "extra_instructions", ""):
        system = system + "\n\n" + cfg.extra_instructions

    content: List[dict] = []
    for img_path, ratings in exemplars:
        if cfg.image_before_text:
            content.append(_image_part(encode_image(img_path)))
            content.append(_text_part(_ratings_text(ratings)))
        else:
            content.append(_text_part(_ratings_text(ratings).replace("above", "below")))
            content.append(_image_part(encode_image(img_path)))

    ask = ("Now predict how this SAME person would rate the following target car. "
           if exemplars else "Rate this car. ")
    content.append(_text_part(ask + "Respond with the JSON object only."))
    content.append(_image_part(encode_image(target_path)))

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def count_images(messages: List[dict]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if p.get("type") == "image_url")
    return n

if __name__ == "__main__":
    import copy, json
    def print_prompt(messages, keep=40):
        view = copy.deepcopy(messages)
        for m in view:
            c = m["content"]
            if isinstance(c, list):
                for p in c:
                    if p.get("type") == "image_url":
                        u = p["image_url"]["url"]
                        p["image_url"]["url"] = u[:keep] + f"...<{len(u)} chars, img>"
        print(json.dumps(view, indent=2))

    IMG = "../Ai_judges_plus/data/output/even_style_sampling/selected_2000_isometric_upload_chunks_renamed/chunk_01"
    exemplars = [
        (f"{IMG}/car_{i}.png", {"sporty": 3, "luxurious": 3, "modern": 4, "rugged": 2, "preference": 4})
        for i in range(1, 6)          # 5 context examples
    ]
    target = f"{IMG}/car_100.png"

    msgs = build_in_context_messages(exemplars, target)
    print_prompt(msgs)
