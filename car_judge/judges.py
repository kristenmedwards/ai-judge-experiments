"""The two judge types: no-context (0-shot) and in-context (few-shot)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from . import prompts
from .client import QwenVLMClient
from .parsing import parse_ratings


@dataclass
class Prediction:
    judge: str
    target_image: str
    predicted: Dict[str, int]          # {dimension: int}, possibly incomplete
    n_context: int
    n_images: int
    raw_text: str = ""
    context_images: List[str] = field(default_factory=list)
    dry_run: bool = False
    messages: Optional[List[dict]] = None  # kept only on dry run for inspection
    from_cache: bool = False  # reused 0-shot prediction; no request was sent


class BaseJudge:
    name = "base"

    def __init__(self, client: QwenVLMClient, image_index: Dict[str, str]):
        self.client = client
        self.image_index = image_index

    def _path(self, image_filename: str) -> str:
        try:
            return self.image_index[image_filename]
        except KeyError as e:
            raise KeyError(f"image {image_filename!r} not found in image index") from e

    def _run(self, messages: List[dict], target_image: str, n_context: int,
             context_images: List[str]) -> Prediction:
        n_images = prompts.count_images(messages)
        resp = self.client.complete(messages, n_images=n_images)
        predicted = {} if resp.dry_run else parse_ratings(resp.text)
        return Prediction(
            judge=self.name,
            target_image=target_image,
            predicted=predicted,
            n_context=n_context,
            n_images=n_images,
            raw_text=resp.text,
            context_images=context_images,
            dry_run=resp.dry_run,
            messages=messages if resp.dry_run else None,
        )


class NoContextJudge(BaseJudge):
    """0-shot judge. Rates each car exactly once.

    The prompt is the rubric plus the target image — no rater identity, no
    exemplars — so the prediction is a function of the image alone and cannot
    differ between two raters who happen to share a car. Repeat targets are
    therefore served from cache instead of re-querying the model.

    This is deliberately *not* done for :class:`InContextJudge`: its ratings are
    conditioned on that rater's exemplars, so the same car genuinely does get a
    different prediction per rater and must be re-queried.

    The cache assumes deterministic decoding, which holds at the default
    ``temperature=0``. Above that, a repeat car reuses the first sample rather
    than drawing a fresh one.
    """

    name = "no_context"

    def __init__(self, client: QwenVLMClient, image_index: Dict[str, str]):
        super().__init__(client, image_index)
        self._cache: Dict[str, Prediction] = {}
        self.n_cache_hits = 0

    @property
    def n_unique_cars(self) -> int:
        """Distinct cars actually sent to the model."""
        return len(self._cache)

    def predict(self, target_image: str) -> Prediction:
        cached = self._cache.get(target_image)
        if cached is not None:
            self.n_cache_hits += 1
            # Copy, so the flag never leaks back onto the stored prediction.
            return replace(cached, from_cache=True)
        messages = prompts.build_no_context_messages(self._path(target_image))
        pred = self._run(messages, target_image, n_context=0, context_images=[])
        self._cache[target_image] = pred
        return pred


class InContextJudge(BaseJudge):
    name = "in_context"

    def predict(
        self,
        target_image: str,
        context: List[Tuple[str, Dict[str, int]]],  # [(image_filename, ratings)]
    ) -> Prediction:
        exemplars = [(self._path(img), ratings) for img, ratings in context]
        messages = prompts.build_in_context_messages(exemplars, self._path(target_image))
        return self._run(
            messages, target_image,
            n_context=len(context),
            context_images=[img for img, _ in context],
        )


def make_judge(name: str, client: QwenVLMClient, image_index: Dict[str, str]) -> BaseJudge:
    if name == "no_context":
        return NoContextJudge(client, image_index)
    if name == "in_context":
        return InContextJudge(client, image_index)
    raise ValueError(f"unknown judge {name!r}")
