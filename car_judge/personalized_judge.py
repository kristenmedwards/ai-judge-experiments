"""A single configurable judge whose behaviour is fully determined by a JudgeConfig.

``n_context=0`` with everything else off == the no-context baseline. Turning on
context / demographics / Q23 / RAG etc. are the outer-loop levers. One judge class
covers the whole search space so the outer loop only ever edits a JudgeConfig.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import prompts
from . import context_selection as ctxsel
from .client import VLMClient
from .config import JudgeConfig
from .long_data import render_profile, render_owned
from .parsing import parse_ratings


@dataclass
class Prediction:
    target_image: str
    predicted: Dict[str, int]
    n_context: int
    n_images: int
    raw_text: str = ""
    context_images: List[str] = field(default_factory=list)
    dry_run: bool = False
    mock: bool = False
    messages: Optional[List[dict]] = None   # kept on dry-run for inspection


class PersonalizedJudge:
    def __init__(self, cfg: JudgeConfig, client: VLMClient, image_index: Dict[str, str]):
        cfg.validate()
        self.cfg = cfg
        self.client = client
        self.image_index = image_index

    def _path(self, image_filename: str) -> str:
        try:
            return self.image_index[image_filename]
        except KeyError as e:
            raise KeyError(f"image {image_filename!r} not in image index") from e

    def _order(self, imgs: List[str], seed: int) -> List[str]:
        if self.cfg.context_order == "shuffle":
            rng = random.Random(f"order:{seed}")
            out = imgs[:]
            rng.shuffle(out)
            return out
        if self.cfg.context_order == "similar_last":
            # rag_* returns most-similar first; put most-similar nearest the target.
            return list(reversed(imgs))
        return imgs  # as_selected

    def predict(self, rater, target: str, context_pool: List[str],
                seed: int = 0) -> Prediction:
        cfg = self.cfg
        ctx_imgs = ctxsel.select_context(
            cfg.context_strategy, context_pool, target, cfg.n_context,
            rater=rater, image_index=self.image_index, seed=seed,
        )
        ctx_imgs = self._order(ctx_imgs, seed)
        exemplars = [(self._path(img), rater.ratings[img]) for img in ctx_imgs]

        profile_text = q23_text = owned_text = ""
        profile = getattr(rater, "profile", {}) or {}
        if cfg.include_demographics:
            profile_text = render_profile(profile, cfg.profile_fields)
        if cfg.include_q23:
            q23_text = getattr(rater, "q23", "")
        if cfg.include_owned:
            owned_text = render_owned(profile)

        messages = prompts.build_messages(
            cfg, self._path(target), exemplars,
            profile_text=profile_text, q23_text=q23_text, owned_text=owned_text,
        )
        n_images = prompts.count_images(messages)
        resp = self.client.complete(messages, n_images=n_images)
        predicted = {} if resp.dry_run else parse_ratings(resp.text)
        return Prediction(
            target_image=target,
            predicted=predicted,
            n_context=len(ctx_imgs),
            n_images=n_images,
            raw_text=resp.text,
            context_images=ctx_imgs,
            dry_run=resp.dry_run,
            mock=resp.mock,
            messages=messages if resp.dry_run else None,
        )
