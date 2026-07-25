"""Environment / credential resolution for OpenAI (and OpenAI-compatible) endpoints.

The API key is NEVER hard-coded and never needs to be pasted into a prompt. It is
read from the environment, optionally seeded from a local ``.env`` file that stays
on your machine and is git-ignored:

    OPENAI_API_KEY=sk-...              # required for live runs
    OPENAI_MODEL=gpt-5.6              # the exact model id you want to call
    OPENAI_BASE_URL=...              # optional; override for Azure / vLLM / proxy

Back-compat: the older VLM_* variables (VLM_BASE_URL / VLM_API_KEY / VLM_MODEL)
are still honored as fallbacks so the original Qwen/vLLM path keeps working.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def load_dotenv_if_present(path: str = ".env") -> bool:
    """Load KEY=VALUE lines from a local .env into os.environ (does not overwrite
    variables already set). Uses python-dotenv if installed, else a tiny parser.
    Returns True if a file was found and read."""
    # Prefer python-dotenv (handles quoting/exports) but degrade gracefully.
    try:
        from dotenv import load_dotenv  # type: ignore
        return bool(load_dotenv(path, override=False))
    except Exception:
        pass
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    return True


@dataclass
class Credentials:
    api_key: str
    model: str
    base_url: Optional[str]  # None => the SDK's default (api.openai.com)

    @property
    def is_live_ready(self) -> bool:
        return bool(self.api_key) and self.api_key not in ("", "EMPTY")


def resolve_credentials(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    dotenv_path: str = ".env",
) -> Credentials:
    """Resolve (api_key, model, base_url), preferring explicit args, then env,
    then a .env file, then legacy VLM_* fallbacks. base_url stays None for the
    default OpenAI endpoint."""
    load_dotenv_if_present(dotenv_path)

    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLM_API_KEY") or ""
    model = (
        model
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("VLM_MODEL")
        or "gpt-5.6"  # placeholder default; set OPENAI_MODEL to the exact id you have
    )
    base_url = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("VLM_BASE_URL")
        or None
    )
    return Credentials(api_key=api_key, model=model, base_url=base_url)
