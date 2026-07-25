"""Chat-completion client for an OpenAI (or OpenAI-compatible) vision model.

Targets the OpenAI API by default (``base_url=None`` => api.openai.com). Point
``base_url`` at a vLLM / Azure / proxy endpoint to reuse the same code path.

Three modes:
  * live      — real API calls (needs OPENAI_API_KEY; model must accept images).
  * dry_run   — builds requests but never contacts a server (inspect/size prompts).
  * mock      — deterministic offline "model" that returns plausible ratings from
                the context it is shown, so the ENTIRE inner/outer loop can be
                exercised and unit-tested with no key and no network.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from .config import RunConfig, DIMENSIONS, SCALE_MIN, SCALE_MAX


@dataclass
class VLMResponse:
    text: str
    n_images: int
    raw: object = None
    dry_run: bool = False
    mock: bool = False


# --------------------------------------------------------------------------- #
# Deterministic offline mock
# --------------------------------------------------------------------------- #
def _mock_ratings_from_messages(messages: List[dict]) -> dict:
    """Produce deterministic ratings so offline runs are reproducible AND weakly
    'learn' from context: if exemplar ratings are present in the prompt we anchor
    on their per-dimension mean; otherwise we hash the target image bytes. This
    gives the outer loop a non-trivial (if fake) signal to optimize against."""
    # Gather any exemplar ratings already written into the prompt text.
    per_dim: dict[str, list] = {d: [] for d in DIMENSIONS}
    target_uri = ""
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            texts = [content]
            uris = []
        elif isinstance(content, list):
            texts = [p.get("text", "") for p in content if p.get("type") == "text"]
            uris = [p["image_url"]["url"] for p in content if p.get("type") == "image_url"]
        else:
            texts, uris = [], []
        for t in texts:
            for d in DIMENSIONS:
                for mt in re.finditer(rf'"{d}"\s*:\s*(\d)', t):
                    per_dim[d].append(int(mt.group(1)))
        if uris:
            target_uri = uris[-1]  # last image is the target

    seed = int(hashlib.sha256(target_uri.encode("utf-8")).hexdigest(), 16)
    out = {}
    for i, d in enumerate(DIMENSIONS):
        if per_dim[d]:
            base = sum(per_dim[d]) / len(per_dim[d])
        else:
            base = SCALE_MIN + (seed >> (i * 5)) % (SCALE_MAX - SCALE_MIN + 1)
        jitter = ((seed >> (i * 7)) % 3) - 1  # -1,0,+1 deterministic wobble
        out[d] = max(SCALE_MIN, min(SCALE_MAX, int(round(base)) + jitter))
    return out


class VLMClient:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._client = None
        if not (cfg.dry_run or cfg.mock):
            self._client = self._make_client()

    def _make_client(self):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for live runs. "
                "`pip install openai`, or use --dry-run / --mock."
            ) from e
        kwargs = dict(api_key=self.cfg.api_key or "EMPTY",
                      timeout=self.cfg.request_timeout, max_retries=0)
        if self.cfg.base_url:
            kwargs["base_url"] = self.cfg.base_url
        return OpenAI(**kwargs)

    def _extra_body(self) -> dict:
        # Only meaningful for vLLM/Qwen; harmless to omit for OpenAI.
        if self.cfg.base_url and self.cfg.enable_thinking:
            return {"chat_template_kwargs": {"enable_thinking": True}}
        return {}

    def complete(self, messages: List[dict], n_images: int = 0) -> VLMResponse:
        if self.cfg.mock:
            import json
            return VLMResponse(text=json.dumps(_mock_ratings_from_messages(messages)),
                               n_images=n_images, mock=True)
        if self.cfg.dry_run or self._client is None:
            return VLMResponse(text="", n_images=n_images, dry_run=True)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                extra = self._extra_body()
                kwargs = dict(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )
                if self.cfg.seed is not None:
                    kwargs["seed"] = self.cfg.seed
                if extra:
                    kwargs["extra_body"] = extra
                resp = self._client.chat.completions.create(**kwargs)
                return VLMResponse(text=resp.choices[0].message.content or "",
                                   n_images=n_images, raw=resp)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(
            f"model request failed after {self.cfg.max_retries} attempts: {last_err}"
        )


# Back-compat alias: existing modules import QwenVLMClient.
QwenVLMClient = VLMClient
