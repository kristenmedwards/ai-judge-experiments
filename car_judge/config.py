"""Run configuration and shared constants."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

# The five rating dimensions, in canonical order. These are also the required
# JSON keys the model must return.
DIMENSIONS: List[str] = ["sporty", "luxurious", "modern", "rugged", "preference"]

SCALE_MIN = 1
SCALE_MAX = 6


@dataclass
class RunConfig:
    """Everything a run needs. Endpoint fields default to env vars so a live run
    can be configured without editing code."""

    # --- data ---
    data_csv: str
    image_root: str
    car_name_mapping: str | None = None  # optional, for human-readable logging

    # --- model / endpoint (served elsewhere via vLLM OpenAI-compatible API) ---
    base_url: str = field(default_factory=lambda: os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("VLM_API_KEY", "EMPTY"))
    model: str = field(default_factory=lambda: os.environ.get("VLM_MODEL", "Qwen/Qwen3.5-9B"))
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = 0            # request-level seed for reproducibility
    enable_thinking: bool = False   # Qwen reasoning toggle (E2 reasoning lever)
    request_timeout: float = 120.0
    max_retries: int = 3

    # --- experiment ---
    judges: List[str] = field(default_factory=lambda: ["no_context", "in_context"])
    n_raters: int = 2               # how many raters to run
    test_size: int = 8              # fixed held-out cars per rater
    context_size: int = 10          # exemplars for the in-context judge
    split_seed: int = 0             # seed for held-out / context sampling
    include_anchors_in_pool: bool = False  # keep the 2 repeated anchors out by default

    # --- run control ---
    dry_run: bool = False
    out_path: str = "outputs/predictions.csv"

    def __post_init__(self) -> None:
        for j in self.judges:
            if j not in ("no_context", "in_context"):
                raise ValueError(f"unknown judge '{j}'")
        if self.context_size < 0:
            raise ValueError("context_size must be >= 0")
