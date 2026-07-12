"""Quick connectivity check for the remote vLLM server.

Usage (env vars, same names the harness uses):

    export VLM_BASE_URL="http://<gpu-host>:8000/v1"
    export VLM_API_KEY="EMPTY"               # only if they set --api-key
    export VLM_MODEL="Qwen/Qwen3.5-9B"
    python scripts/ping_vlm.py               # text-only ping + model list
    python scripts/ping_vlm.py --image       # also send a 1x1 image (vision path)

Exit code 0 = reachable and the target model answered.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys

# A 1x1 red PNG, so --image needs no files on disk.
_TINY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42m"
    b"NkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ping the remote vLLM server.")
    ap.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "Qwen/Qwen3.5-9B"))
    ap.add_argument("--image", action="store_true", help="also test the vision path")
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("!! openai SDK not installed: pip install 'openai>=1.30.0'", file=sys.stderr)
        return 2

    print(f"[cfg] base_url={args.base_url}  model={args.model}")
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=30, max_retries=0)

    # 1) List models -> proves URL + auth, and that the model is loaded.
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as e:  # noqa: BLE001
        print(f"!! could not reach {args.base_url}: {e}", file=sys.stderr)
        return 1
    print(f"[ok ] server reachable; models served: {served}")
    if args.model not in served:
        print(f"!! WARNING: your --model '{args.model}' is not in {served}", file=sys.stderr)

    # 2) One tiny chat completion -> proves the chat template + generation work.
    content = "Reply with the single word: pong."
    if args.image:
        uri = "data:image/png;base64," + base64.b64encode(_TINY_PNG).decode("ascii")
        content = [
            {"type": "text", "text": "Reply with the single word: pong."},
            {"type": "image_url", "image_url": {"url": uri}},
        ]
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=8,
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001
        print(f"!! chat completion failed{' (vision path)' if args.image else ''}: {e}", file=sys.stderr)
        return 1
    print(f"[ok ] model replied: {resp.choices[0].message.content!r}")
    print("[done] server is reachable and answering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
