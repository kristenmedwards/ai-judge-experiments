"""Run configuration, the five rating dimensions, and the JudgeConfig artifact.

Two config objects live here:

* ``RunConfig``  — endpoint + experiment plumbing (model, seeds, data paths).
* ``JudgeConfig`` — the *thing the outer loop mutates*. It fully specifies one
  AI judge: how much context it sees, whether it sees demographics / the rater's
  Q23 free-text, how context cars are chosen, ordering, and the prompt variant.
  This is the analogue of Karpathy autoresearch's ``train.py`` — the editable
  artifact whose changes we keep or revert based on the inner-loop score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# The five rating dimensions, in canonical order. Also the required JSON keys the
# model must return. (In the merged long CSV, "preference" is stored as the
# column "overall_preference"; the loader maps it to "preference".)
DIMENSIONS: List[str] = ["sporty", "luxurious", "modern", "rugged", "preference"]

SCALE_MIN = 1
SCALE_MAX = 6


# ===================================================================== #
# RunConfig — endpoint + experiment plumbing
# ===================================================================== #
@dataclass
class RunConfig:
    """Everything a run needs that is NOT part of the judge design itself.
    Endpoint fields default to env vars (OPENAI_* preferred, VLM_* fallback) so a
    live run needs no code edits — just a .env or exported variables."""

    # --- data ---
    data_csv: str
    image_root: str
    car_name_mapping: str | None = None

    # --- model / endpoint ---
    base_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("VLM_BASE_URL")
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
        or os.environ.get("VLM_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("OPENAI_MODEL")
        or os.environ.get("VLM_MODEL", "gpt-5.6")
    )
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int | None = 0
    enable_thinking: bool = False   # legacy Qwen reasoning toggle (ignored by OpenAI)
    request_timeout: float = 120.0
    max_retries: int = 4
    max_concurrency: int = 4        # parallel requests for the inner loop

    # --- experiment plumbing ---
    n_raters: int = 5
    test_size: int = 20             # held-out target cars per rater (the "N remaining")
    split_seed: int = 0
    # The 4 anchor cars are rated by every participant, so they carry the
    # inter-rater agreement ceiling. Kept out of each rater's context pool and
    # held-out set by default so they stay a clean reference.
    include_anchors_in_pool: bool = False
    min_cars: Optional[int] = None  # min rated cars to include a rater (auto if None)

    # --- back-compat with the original run_experiment / run_context_sweep CLIs ---
    judges: List[str] = field(default_factory=lambda: ["no_context", "in_context"])
    context_size: int = 10

    # --- run control ---
    dry_run: bool = False
    mock: bool = False              # deterministic offline judge (no server, no key)
    out_path: str = "outputs/predictions.csv"

    def __post_init__(self) -> None:
        for j in self.judges:
            if j not in ("no_context", "in_context"):
                raise ValueError(f"unknown judge '{j}'")
        if self.context_size < 0:
            raise ValueError("context_size must be >= 0")


# ===================================================================== #
# JudgeConfig — the editable artifact the outer loop mutates
# ===================================================================== #
CONTEXT_STRATEGIES = ("random", "diverse", "rag_clip", "rag_image", "rag_profile", "first")
PROFILE_FIELDS_DEFAULT = [
    "prolific_age", "prolific_sex", "num_children_under18",
    "driving_location", "driving_frequency",
    "bfi_openness", "bfi_conscientiousness", "bfi_extraversion",
    "bfi_agreeableness", "bfi_neuroticism",
]


@dataclass
class JudgeConfig:
    """One fully-specified AI judge. Mutating any field defines a new experiment.

    Keep this JSON-serializable (only str/int/float/bool/list) so it round-trips
    cleanly into results.tsv and into the LLM-readable experiment log."""

    name: str = "baseline_no_context"

    # --- context (few-shot exemplars drawn from cars THIS rater rated) ---
    n_context: int = 0                       # 0 => the no-context baseline
    context_strategy: str = "random"         # one of CONTEXT_STRATEGIES
    context_order: str = "as_selected"       # as_selected | similar_last | shuffle
    image_before_text: bool = True           # exemplar image then its ratings (Qin et al.)

    # --- side information about the person ---
    include_demographics: bool = False
    profile_fields: List[str] = field(default_factory=lambda: list(PROFILE_FIELDS_DEFAULT))
    include_q23: bool = False                # the rater's "what I look at" free-text
    include_owned: bool = False              # brands / body styles they own
    include_guide: bool = False              # guide-derived per-person "taste card"
                                             # (rater_matching_guide.md via guide_note.py)

    # --- prompt wording ---
    prompt_variant: str = "default"          # key into prompts.SYSTEM_VARIANTS
    extra_instructions: str = ""             # free-form text appended to the system prompt

    # --- model knobs that belong to the judge design (not the endpoint) ---
    temperature: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """One-line human/LLM-readable description for the results log."""
        bits = [f"N={self.n_context}", self.context_strategy]
        if self.include_demographics:
            bits.append("demo")
        if self.include_q23:
            bits.append("q23")
        if self.include_owned:
            bits.append("owned")
        if self.include_guide:
            bits.append("guide")
        if self.prompt_variant != "default":
            bits.append(f"prompt={self.prompt_variant}")
        if self.context_order != "as_selected":
            bits.append(self.context_order)
        return " · ".join(bits)

    def validate(self) -> None:
        if self.context_strategy not in CONTEXT_STRATEGIES:
            raise ValueError(f"context_strategy must be one of {CONTEXT_STRATEGIES}")
        if self.n_context < 0:
            raise ValueError("n_context must be >= 0")
        if self.context_order not in ("as_selected", "similar_last", "shuffle"):
            raise ValueError("bad context_order")
