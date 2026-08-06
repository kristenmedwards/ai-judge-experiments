"""Chat-completion client for an OpenAI (or OpenAI-compatible) vision model.

Targets the OpenAI API by default (``base_url=None`` => api.openai.com). Point
``base_url`` at a vLLM / Azure / proxy endpoint to reuse the same code path.

Three modes:
  * live      — real API calls (needs OPENAI_API_KEY; model must accept images).
  * dry_run   — builds requests but never contacts a server (inspect/size prompts).
  * mock      — deterministic offline "model" that returns plausible ratings from
                the context it is shown, so the ENTIRE inner/outer loop can be
                exercised and unit-tested with no key and no network.

Parameter self-healing (live only): newer OpenAI models (o-series / GPT-5+)
reject ``max_tokens`` in favour of ``max_completion_tokens`` and only accept the
default ``temperature``. On a 400 "unsupported_parameter" the client adapts the
request once (renames the token param / drops the rejected field) and retries
without spending a network-retry, then remembers the fix for every later call.
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
        # Request-shape state, adjusted on the fly if the endpoint rejects a param.
        self._token_param = "max_tokens"   # some models require 'max_completion_tokens'
        self._drop_params: set[str] = set()  # fields the endpoint reported unsupported
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

    def _build_kwargs(self, messages: List[dict]) -> dict:
        """Assemble the create() kwargs honouring any learned param adjustments."""
        kwargs: dict = dict(model=self.cfg.model, messages=messages)
        if "temperature" not in self._drop_params:
            kwargs["temperature"] = self.cfg.temperature
        if "max_tokens" not in self._drop_params:
            kwargs[self._token_param] = self.cfg.max_tokens
        if self.cfg.seed is not None and "seed" not in self._drop_params:
            kwargs["seed"] = self.cfg.seed
        extra = self._extra_body()
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    def _adapt_to_error(self, err: Exception) -> Optional[str]:
        """If ``err`` is an 'unsupported parameter' 400, adjust request shape and
        return a short note describing the change (so the caller can retry for
        free). Return None if the error is not a recognised param problem."""
        msg = str(err)
        low = msg.lower()
        if "unsupported" not in low and "not supported" not in low:
            return None
        # Model wants max_completion_tokens rather than max_tokens.
        if "max_completion_tokens" in low and self._token_param != "max_completion_tokens":
            self._token_param = "max_completion_tokens"
            return "use max_completion_tokens"
        # Otherwise OpenAI names the offending field in 'param'; drop it and retry.
        m = re.search(r"'param':\s*'([^']+)'", msg) or re.search(r"parameter:\s*'([^']+)'", msg)
        param = m.group(1) if m else None
        if param in ("max_tokens", "max_completion_tokens") and self._token_param != "max_completion_tokens":
            self._token_param = "max_completion_tokens"
            return "use max_completion_tokens"
        # Normalise both token param names to the single 'max_tokens' drop-key.
        key = "max_tokens" if param in ("max_tokens", "max_completion_tokens") else param
        if key and key not in self._drop_params:
            self._drop_params.add(key)
            return f"drop unsupported '{key}'"
        return None

    def complete(self, messages: List[dict], n_images: int = 0) -> VLMResponse:
        if self.cfg.mock:
            import json
            return VLMResponse(text=json.dumps(_mock_ratings_from_messages(messages)),
                               n_images=n_images, mock=True)
        if self.cfg.dry_run or self._client is None:
            return VLMResponse(text="", n_images=n_images, dry_run=True)

        last_err: Optional[Exception] = None
        attempt = 0
        adapt_budget = 4  # cap param-rewrites so a persistent 400 can't loop forever
        while attempt < self.cfg.max_retries:
            try:
                resp = self._client.chat.completions.create(**self._build_kwargs(messages))
                return VLMResponse(text=resp.choices[0].message.content or "",
                                   n_images=n_images, raw=resp)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if adapt_budget > 0:
                    note = self._adapt_to_error(e)
                    if note:
                        adapt_budget -= 1
                        # corrected a rejected parameter — retry immediately, no penalty
                        continue
                attempt += 1
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(
            f"model request failed after {self.cfg.max_retries} attempts: {last_err}"
        )


# Back-compat alias: existing modules import QwenVLMClient.
QwenVLMClient = VLMClient
