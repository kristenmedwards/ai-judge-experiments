"""Client for a Qwen-VL model served via vLLM's OpenAI-compatible endpoint.

The model is assumed to run elsewhere (a GPU box). This wraps the stock ``openai``
SDK, pointing ``base_url`` at that server. ``dry_run=True`` never contacts a
server; it just records the request that *would* be sent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from .config import RunConfig


@dataclass
class VLMResponse:
    text: str
    n_images: int
    raw: object = None       # underlying SDK response (None on dry run)
    dry_run: bool = False


class QwenVLMClient:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._client = None
        if not cfg.dry_run:
            self._client = self._make_client()

    def _make_client(self):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for live runs. "
                "Install it with `pip install openai`, or use --dry-run."
            ) from e
        return OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.request_timeout,
            max_retries=0,  # we handle retries ourselves for clearer logging
        )

    # ------------------------------------------------------------------ #
    def _extra_body(self) -> dict:
        # Qwen's reasoning toggle is passed through vLLM's chat_template_kwargs.
        return {"chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking}}

    def complete(self, messages: List[dict], n_images: int = 0) -> VLMResponse:
        """Send one chat-completion request and return the text response."""
        if self.cfg.dry_run or self._client is None:
            return VLMResponse(text="", n_images=n_images, dry_run=True)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    seed=self.cfg.seed,
                    extra_body=self._extra_body(),
                )
                return VLMResponse(
                    text=resp.choices[0].message.content or "",
                    n_images=n_images,
                    raw=resp,
                )
            except Exception as e:  # noqa: BLE001 - surface after retries
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(
            f"VLM request failed after {self.cfg.max_retries} attempts: {last_err}"
        )
