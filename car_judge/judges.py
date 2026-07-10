"""The two judge types: no-context (0-shot) and in-context (few-shot)."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    name = "no_context"

    def predict(self, target_image: str) -> Prediction:
        messages = prompts.build_no_context_messages(self._path(target_image))
        return self._run(messages, target_image, n_context=0, context_images=[])


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
