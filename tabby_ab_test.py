"""Compare TabbyAPI completion against the backend's exact Qwen FIM prompt.

Example:
    python tabby_ab_test.py --prefix "def add(a, b):\n    " --suffix "\n"

The script speaks to an OpenAI-compatible completions endpoint and prints the
first-token latency plus the first completion text. It intentionally imports
``build_qwen_fim_prompt`` so TabbyAPI and the custom backend are tested with
the same prompt structure.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from typing import Any, Iterator

from engine import build_qwen_fim_prompt


DEFAULT_TABBY_URL = "http://127.0.0.1:5000/v1/completions"


def iter_sse_payloads(response: Any) -> Iterator[dict[str, Any]]:
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue

        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break

        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def extract_completion_chunk(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""

    choice = choices[0]
    if "text" in choice:
        return str(choice.get("text") or "")

    delta = choice.get("delta") or {}
    return str(delta.get("content") or "")


def run_tabby_probe(args: argparse.Namespace) -> None:
    document = args.prefix + args.suffix
    prompt = build_qwen_fim_prompt(document, len(args.prefix))
    request_payload: dict[str, Any] = {
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": True,
    }

    if args.model:
        request_payload["model"] = args.model

    request = urllib.request.Request(
        args.url,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.perf_counter()
    first_token_at: float | None = None
    chunks: list[str] = []

    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for payload in iter_sse_payloads(response):
            chunk = extract_completion_chunk(payload)
            if not chunk:
                continue

            if first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(chunk)

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    ttft_ms = (
        int((first_token_at - started_at) * 1000)
        if first_token_at is not None
        else None
    )

    print(f"tabby_url={args.url}")
    print(f"ttft_ms={ttft_ms}")
    print(f"elapsed_ms={elapsed_ms}")
    print(f"prompt={prompt!r}")
    print(f"completion={''.join(chunks)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("TABBY_API_URL", DEFAULT_TABBY_URL))
    parser.add_argument("--model", default=os.getenv("TABBY_MODEL", ""))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--timeout", type=float, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run_tabby_probe(parse_args())
