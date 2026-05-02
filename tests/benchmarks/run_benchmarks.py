"""One-command benchmark suite for model, parameter, and scenario testing.

User-facing command:
    python tests/benchmarks/run_benchmarks.py

The suite writes all outputs to:
    tests/benchmarks/outputs/<timestamped_run_id>/

It is intentionally self-contained so this file is the only file users need to
run. JSON config/scenario files next to it are optional and have embedded
fallback defaults.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import json
import os
import re
import statistics
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
OUTPUT_ROOT = BENCHMARK_DIR / "outputs"
DEFAULT_CONFIG_PATH = BENCHMARK_DIR / "benchmark_config.json"
DEFAULT_SCENARIOS_PATH = BENCHMARK_DIR / "scenarios.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CONFIG: dict[str, Any] = {
    "repetitions": 1,
    "draft_modes": ["target_only", "draft_1b", "draft_3b"],
    "fast_path_modes": ["llama_instruct", "qwen_fim", "granite_fim"],
    "speculative_draft_tokens": [1, 2, 3, 4],
    "rewrite": {
        "temperature": [0.2],
        "top_p": [0.8],
        "top_k": [40],
        "repetition_penalty": [1.05],
        "max_new_tokens": [128],
    },
    "autocomplete": {
        "temperature": [0.2],
        "top_p": [0.9],
        "top_k": [40],
        "repetition_penalty": [1.05],
        "max_new_tokens": [12],
    },
    "correction": {
        "temperature": [0.2],
        "top_p": [0.9],
        "top_k": [40],
        "repetition_penalty": [1.05],
        "max_new_tokens": [256],
    },
}

DEFAULT_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "rewrite_short_professional",
        "task": "rewrite",
        "name": "Short professional rewrite",
        "text": "Hey Sam, I looked at the thing and it seems mostly fine but some parts are kind of confusing.",
        "instruction": "Make this sound more professional and concise.",
    },
    {
        "id": "rewrite_long_ai_ethics",
        "task": "rewrite",
        "name": "Long AI ethics paragraph",
        "text": (
            "The rapid development of artificial intelligence has changed how "
            "people use technology, creating better productivity and new ethical "
            "problems. Many organizations use AI tools to simplify routine work "
            "and decisions, but the risks of depending too much on automated "
            "systems remain controversial. Critics argue that too much reliance "
            "on automation can reduce human oversight and lead to unexpected "
            "outcomes in healthcare and finance. Supporters argue that AI can "
            "process large datasets accurately and enable discoveries that were "
            "once impossible. Finding the right balance between using AI and "
            "keeping human accountability is therefore important."
        ),
        "instruction": "Rewrite this as a polished academic paragraph while preserving the meaning.",
    },
    {
        "id": "autocomplete_paragraph_end",
        "task": "autocomplete",
        "name": "Autocomplete at paragraph end",
        "document_text": "The research team evaluated the deployment risks and concluded that the safest next step was",
        "cursor": 88,
        "reference_completions": [
            "to pause the rollout.",
            "to delay the rollout.",
            "to proceed cautiously.",
        ],
    },
    {
        "id": "autocomplete_mid_sentence",
        "task": "autocomplete",
        "name": "Autocomplete mid sentence",
        "document_text": "The committee approved the revised policy because  would reduce operational uncertainty for regional teams.",
        "cursor": 47,
        "reference_completions": [
            "it",
            "the change",
            "the update",
        ],
    },
    {
        "id": "correction_typos_grammar",
        "task": "correction",
        "name": "Typos and grammar correction",
        "document_text": "The manager were happy with the report, but it contain several typo and missing punctuation",
        "cursor": 83,
    },
]

CSV_FIELDS = [
    "timestamp",
    "run_id",
    "row_id",
    "task",
    "scenario_id",
    "scenario_name",
    "repetition",
    "target_model",
    "draft_model_label",
    "draft_model_path",
    "fast_path_mode",
    "fast_path_model_path",
    "fast_model",
    "speculative_enabled",
    "speculative_draft_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_new_tokens",
    "prompt_chars",
    "prompt_tokens",
    "output_chars",
    "output_tokens",
    "ttft_ms",
    "total_ms",
    "tokens_per_second",
    "accepted_draft_tokens",
    "rejected_draft_tokens",
    "draft_acceptance_rate",
    "stop_reason",
    "cancelled",
    "error",
    "empty_output",
    "role_leak",
    "code_leak",
    "missing_word_suspected",
    "malformed_json",
    "quality_flags",
    "quality_score",
    "output_preview",
    "raw_output",
]

SUMMARY_FIELDS = [
    "task",
    "draft_model_label",
    "speculative_draft_tokens",
    "scenario_id",
    "runs",
    "errors",
    "avg_ttft_ms",
    "avg_total_ms",
    "avg_tps",
    "avg_acceptance_rate",
    "quality_flag_rate",
    "avg_quality_score",
]

QUALITY_COMPARISON_FIELDS = [
    "task",
    "scenario_id",
    "baseline_label",
    "candidate_label",
    "baseline_quality",
    "candidate_quality",
    "quality_delta",
]

TELEMETRY_RE = re.compile(
    r"\[(?P<label>[^\]]+)\]\s+TTFT:\s+(?P<ttft>\d+)ms\s+\|\s+TPS:\s+"
    r"(?P<tps>[0-9.]+)\s+\|\s+(?:Draft Acceptance:\s+(?P<acceptance>[^|]+)\|\s+)?"
    r"Total Time:\s+(?P<total>\d+)ms\s+\|\s+Tokens:\s+(?P<tokens>\d+)"
)
STOP_REASON_RE = re.compile(r"\[HEAVY-PATH\]\s+Stop reason:\s+(?P<reason>.+)")
ACCEPTANCE_FRACTION_RE = re.compile(r"(?P<accepted>\d+)\s*/\s*(?P<draft>\d+)")
ACCEPTANCE_PERCENT_RE = re.compile(r"(?P<percent>[0-9.]+)%")


@dataclass(frozen=True)
class BenchmarkCase:
    task: str
    scenario: dict[str, Any]
    repetition: int
    draft_model_label: str
    draft_model_path: str
    fast_path_mode: str
    fast_path_model_path: str
    speculative_enabled: bool
    speculative_draft_tokens: int | None
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_new_tokens: int


class CancelEvent:
    def __init__(self) -> None:
        self._cancelled = False

    def is_set(self) -> bool:
        return self._cancelled

    def set(self) -> None:
        self._cancelled = True


@contextlib.contextmanager
def patched_environ(values: dict[str, str]) -> Iterator[None]:
    old_values: dict[str, str | None] = {}
    for key, value in values.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def bool_csv(value: bool) -> str:
    return "true" if value else "false"


def number_or_empty(value: Any) -> Any:
    return "" if value is None else value


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalize_text(text).lower())


def preview(text: str, limit: int = 220) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def safe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def mean(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in values if safe_float(value) is not None]
    if not clean:
        return None
    return statistics.fmean(clean)


def parse_telemetry(log_text: str) -> dict[str, Any]:
    telemetry: dict[str, Any] = {
        "ttft_ms": None,
        "total_ms": None,
        "tokens_per_second": None,
        "output_tokens": None,
        "accepted_draft_tokens": None,
        "rejected_draft_tokens": None,
        "draft_acceptance_rate": None,
        "stop_reason": "",
    }

    for match in TELEMETRY_RE.finditer(log_text):
        telemetry["ttft_ms"] = int(match.group("ttft"))
        telemetry["total_ms"] = int(match.group("total"))
        telemetry["tokens_per_second"] = float(match.group("tps"))
        telemetry["output_tokens"] = int(match.group("tokens"))
        acceptance = (match.group("acceptance") or "").strip()
        if acceptance:
            fraction = ACCEPTANCE_FRACTION_RE.search(acceptance)
            percent = ACCEPTANCE_PERCENT_RE.search(acceptance)
            if fraction:
                accepted = int(fraction.group("accepted"))
                draft = int(fraction.group("draft"))
                telemetry["accepted_draft_tokens"] = accepted
                telemetry["rejected_draft_tokens"] = max(0, draft - accepted)
                telemetry["draft_acceptance_rate"] = accepted / draft if draft else None
            elif percent:
                telemetry["draft_acceptance_rate"] = float(percent.group("percent")) / 100

    stop_matches = list(STOP_REASON_RE.finditer(log_text))
    if stop_matches:
        telemetry["stop_reason"] = stop_matches[-1].group("reason").strip()

    return telemetry


def quality_flags(task: str, output: str, error: str) -> dict[str, bool]:
    normalized = normalize_text(output)
    lower = normalized.lower()
    role_leak = any(
        marker in lower
        for marker in (
            "assistant:",
            "user:",
            "system:",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|end_of_text|>",
        )
    )
    code_leak = any(
        marker in output
        for marker in ("```", "\nimport ", "\ndef ", "\nclass ", "{", "}")
    )
    missing_word_suspected = bool(
        re.search(r"\b\w+(?:toseen|oflying|unforeseen?outcomes|inconce$)", lower)
        or re.search(r"\b(?:of|to|and|between|ensuring|with|that|a|the)$", lower)
    )
    malformed_json = task == "correction" and bool(error)
    return {
        "empty_output": not bool(normalized),
        "role_leak": role_leak,
        "code_leak": code_leak,
        "missing_word_suspected": missing_word_suspected,
        "malformed_json": malformed_json,
    }


def token_f1(reference: str, candidate: str) -> float:
    reference_tokens = normalize_tokens(reference)
    candidate_tokens = normalize_tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0

    reference_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    for token in candidate_tokens:
        candidate_counts[token] = candidate_counts.get(token, 0) + 1

    overlap = 0
    for token, count in candidate_counts.items():
        overlap += min(count, reference_counts.get(token, 0))
    if overlap <= 0:
        return 0.0

    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def quality_score(
    case: BenchmarkCase,
    output: str,
    error: str,
    flags: dict[str, bool],
) -> float | None:
    if error:
        return 0.0
    if flags["empty_output"] or flags["role_leak"] or flags["code_leak"]:
        return 0.0

    references = case.scenario.get("reference_completions")
    if case.task == "autocomplete" and isinstance(references, list) and references:
        best = max(token_f1(str(reference), output) for reference in references)
        if flags["missing_word_suspected"]:
            best *= 0.8
        return round(best, 4)

    return None


def build_env_for_case(args: argparse.Namespace, case: BenchmarkCase) -> dict[str, str]:
    env = {
        "USE_MOCK_ENGINE": "true" if args.mock else "false",
        "FAST_PATH_MODE": case.fast_path_mode,
        "LLAMA_FAST_MODEL_DIR": args.fast_model or os.getenv("LLAMA_FAST_MODEL_DIR", ""),
        "QWEN_AUTOCOMPLETE_MODEL_DIR": args.qwen_model
        or os.getenv("QWEN_AUTOCOMPLETE_MODEL_DIR", ""),
        "GRANITE_FIM_MODEL_DIR": args.granite_fim_model
        or os.getenv("GRANITE_FIM_MODEL_DIR", ""),
        "LLAMA_REWRITE_TARGET_MODEL_DIR": args.target_model
        or os.getenv("LLAMA_REWRITE_TARGET_MODEL_DIR", ""),
        "GENERATION_TEMPERATURE": str(case.temperature),
        "GENERATION_TOP_P": str(case.top_p),
        "GENERATION_TOP_K": str(case.top_k),
        "GENERATION_REPETITION_PENALTY": str(case.repetition_penalty),
        "REWRITE_TEMPERATURE": str(case.temperature),
        "REWRITE_TOP_P": str(case.top_p),
        "REWRITE_TOP_K": str(case.top_k),
        "REWRITE_REPETITION_PENALTY": str(case.repetition_penalty),
        "REWRITE_MAX_NEW_TOKENS": str(case.max_new_tokens),
        "AUTOCOMPLETE_MAX_NEW_TOKENS": str(case.max_new_tokens),
        "CORRECTION_MAX_NEW_TOKENS": str(case.max_new_tokens),
        "ENABLE_SPECULATIVE_REWRITE": "true" if case.speculative_enabled else "false",
        "SPECULATIVE_DRAFT_TOKENS": str(case.speculative_draft_tokens or 1),
    }
    if case.draft_model_path:
        env["LLAMA_REWRITE_DRAFT_MODEL_DIR"] = case.draft_model_path
    elif args.mock:
        env["LLAMA_REWRITE_DRAFT_MODEL_DIR"] = "mock-draft"
    return env


def expand_cases(
    config: dict[str, Any],
    scenarios: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[BenchmarkCase]:
    repetitions = args.repetitions if args.repetitions is not None else int(config["repetitions"])
    allowed_tasks = set(args.tasks.split(",")) if args.tasks else None
    allowed_draft_modes = set(args.draft_modes.split(",")) if args.draft_modes else None
    allowed_fast_path_modes = (
        set(args.fast_path_modes.split(",")) if args.fast_path_modes else None
    )
    draft_modes = [
        mode
        for mode in config.get("draft_modes", DEFAULT_CONFIG["draft_modes"])
        if allowed_draft_modes is None or mode in allowed_draft_modes
    ]
    fast_path_modes = [
        mode
        for mode in config.get("fast_path_modes", DEFAULT_CONFIG["fast_path_modes"])
        if allowed_fast_path_modes is None or mode in allowed_fast_path_modes
    ]
    draft_token_values = [int(value) for value in ensure_list(config["speculative_draft_tokens"])]
    cases: list[BenchmarkCase] = []

    for scenario in scenarios:
        task = str(scenario["task"])
        if allowed_tasks is not None and task not in allowed_tasks:
            continue

        task_grid = config.get(task, {})
        temperatures = [float(value) for value in ensure_list(task_grid.get("temperature", [0.2]))]
        top_ps = [float(value) for value in ensure_list(task_grid.get("top_p", [0.9]))]
        top_ks = [int(value) for value in ensure_list(task_grid.get("top_k", [40]))]
        penalties = [
            float(value)
            for value in ensure_list(task_grid.get("repetition_penalty", [1.05]))
        ]
        max_tokens = [
            int(value)
            for value in ensure_list(task_grid.get("max_new_tokens", [128 if task == "rewrite" else 12]))
        ]

        if task == "rewrite":
            mode_values = draft_modes
        else:
            mode_values = fast_path_modes if task == "autocomplete" else ["llama_instruct"]

        for repetition in range(1, repetitions + 1):
            for mode in mode_values:
                token_values: list[int | None] = (
                    draft_token_values if mode in {"draft_1b", "draft_3b"} else [None]
                )
                for temp, top_p, top_k, penalty, max_new_tokens, draft_tokens in itertools.product(
                    temperatures,
                    top_ps,
                    top_ks,
                    penalties,
                    max_tokens,
                    token_values,
                ):
                    if mode == "draft_1b":
                        draft_path = args.draft_1b_model or os.getenv(
                            "LLAMA_REWRITE_DRAFT_MODEL_DIR",
                            "",
                        )
                    elif mode == "draft_3b":
                        draft_path = args.draft_3b_model or args.fast_model or os.getenv(
                            "LLAMA_FAST_MODEL_DIR",
                            "",
                        )
                    else:
                        draft_path = ""
                    if task == "autocomplete":
                        fast_path_mode = mode
                    else:
                        fast_path_mode = "llama_instruct"

                    if fast_path_mode == "qwen_fim":
                        fast_path_model_path = args.qwen_model or os.getenv(
                            "QWEN_AUTOCOMPLETE_MODEL_DIR",
                            "",
                        )
                    elif fast_path_mode == "granite_fim":
                        fast_path_model_path = args.granite_fim_model or os.getenv(
                            "GRANITE_FIM_MODEL_DIR",
                            "",
                        )
                    else:
                        fast_path_model_path = args.fast_model or os.getenv(
                            "LLAMA_FAST_MODEL_DIR",
                            "",
                        )
                    cases.append(
                        BenchmarkCase(
                            task=task,
                            scenario=scenario,
                            repetition=repetition,
                            draft_model_label=mode,
                            draft_model_path=draft_path,
                            fast_path_mode=fast_path_mode,
                            fast_path_model_path=fast_path_model_path,
                            speculative_enabled=mode in {"draft_1b", "draft_3b"},
                            speculative_draft_tokens=draft_tokens,
                            temperature=temp,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=penalty,
                            max_new_tokens=max_new_tokens,
                        )
                    )

    if args.limit is not None:
        return cases[: args.limit]
    return cases


def resolve_cursor(scenario: dict[str, Any]) -> int:
    text = str(scenario.get("document_text", ""))
    cursor = scenario.get("cursor")
    if cursor is None:
        return len(text)
    return min(max(int(cursor), 0), len(text))


def prompt_info(engine: Any, case: BenchmarkCase) -> tuple[int | None, int | None]:
    try:
        from engine import (
            build_llama_autocomplete_prompt,
            build_llama_correction_prompt,
            build_qwen_fim_prompt,
            build_llama_rewrite_prompt,
        )

        if case.task == "rewrite":
            prompt = build_llama_rewrite_prompt(
                str(case.scenario["text"]),
                str(case.scenario["instruction"]),
            )
            tokenizer = getattr(getattr(engine, "llama_target", None), "tokenizer", None)
        elif case.task == "autocomplete":
            text = str(case.scenario["document_text"])
            if case.fast_path_mode in {"qwen_fim", "granite_fim"}:
                prompt = build_qwen_fim_prompt(text, resolve_cursor(case.scenario))
                tokenizer_attr = "qwen" if case.fast_path_mode == "qwen_fim" else "granite_fim"
                tokenizer = getattr(getattr(engine, tokenizer_attr, None), "tokenizer", None)
            else:
                prompt = build_llama_autocomplete_prompt(text, resolve_cursor(case.scenario))
                tokenizer = getattr(getattr(engine, "fast_llama", None), "tokenizer", None)
        else:
            text = str(case.scenario["document_text"])
            prompt, _, _ = build_llama_correction_prompt(text, resolve_cursor(case.scenario))
            tokenizer = getattr(getattr(engine, "fast_llama", None), "tokenizer", None)

        prompt_tokens = None
        if tokenizer is not None and hasattr(engine, "_encode_prompt"):
            prompt_tokens = len(engine._encode_prompt(tokenizer, prompt))
        return len(prompt), prompt_tokens
    except Exception:
        return None, None


def run_case(args: argparse.Namespace, case: BenchmarkCase) -> tuple[dict[str, Any], str, str]:
    env = build_env_for_case(args, case)
    started_at = time.perf_counter()
    output = ""
    error = ""
    log_buffer = io.StringIO()
    prompt_chars: int | None = None
    prompt_tokens: int | None = None

    try:
        with patched_environ(env):
            import engine as engine_module

            # Construct inside the patched environment so each run gets isolated
            # model paths and decoding settings.
            benchmark_engine = engine_module.get_engine()
            prompt_chars, prompt_tokens = prompt_info(benchmark_engine, case)
            cancel_event = CancelEvent()
            with contextlib.redirect_stdout(log_buffer):
                if case.task == "rewrite":
                    chunks = list(
                        benchmark_engine.apply_rewrite(
                            str(case.scenario["text"]),
                            str(case.scenario["instruction"]),
                            cancel_event,
                            case.max_new_tokens,
                        )
                    )
                    output = "".join(chunks)
                elif case.task == "autocomplete":
                    text = str(case.scenario["document_text"])
                    chunks = list(
                        benchmark_engine.generate_autocomplete_stream(
                            text,
                            resolve_cursor(case.scenario),
                            cancel_event,
                            case.max_new_tokens,
                        )
                    )
                    output = "".join(chunks)
                elif case.task == "correction":
                    text = str(case.scenario["document_text"])
                    corrections = benchmark_engine.generate_corrections(
                        text,
                        resolve_cursor(case.scenario),
                        cancel_event,
                    )
                    output = json.dumps(corrections, ensure_ascii=True, separators=(",", ":"))
                else:
                    error = f"unknown_task:{case.task}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log_buffer.write(traceback.format_exc())

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    captured_logs = log_buffer.getvalue()
    telemetry = parse_telemetry(captured_logs)
    flags = quality_flags(case.task, output, error)
    quality_flag_names = [name for name, enabled in flags.items() if enabled]
    score = quality_score(case, output, error, flags)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "row_id": "",
        "task": case.task,
        "scenario_id": case.scenario["id"],
        "scenario_name": case.scenario.get("name", case.scenario["id"]),
        "repetition": case.repetition,
        "target_model": env.get("LLAMA_REWRITE_TARGET_MODEL_DIR", ""),
        "draft_model_label": case.draft_model_label,
        "draft_model_path": case.draft_model_path,
        "fast_path_mode": case.fast_path_mode,
        "fast_path_model_path": case.fast_path_model_path,
        "fast_model": env.get("LLAMA_FAST_MODEL_DIR", ""),
        "speculative_enabled": bool_csv(case.speculative_enabled),
        "speculative_draft_tokens": number_or_empty(case.speculative_draft_tokens),
        "temperature": case.temperature,
        "top_p": case.top_p,
        "top_k": case.top_k,
        "repetition_penalty": case.repetition_penalty,
        "max_new_tokens": case.max_new_tokens,
        "prompt_chars": number_or_empty(prompt_chars),
        "prompt_tokens": number_or_empty(prompt_tokens),
        "output_chars": len(output),
        "output_tokens": number_or_empty(telemetry["output_tokens"]),
        "ttft_ms": number_or_empty(telemetry["ttft_ms"]),
        "total_ms": number_or_empty(telemetry["total_ms"] or elapsed_ms),
        "tokens_per_second": number_or_empty(telemetry["tokens_per_second"]),
        "accepted_draft_tokens": number_or_empty(telemetry["accepted_draft_tokens"]),
        "rejected_draft_tokens": number_or_empty(telemetry["rejected_draft_tokens"]),
        "draft_acceptance_rate": number_or_empty(telemetry["draft_acceptance_rate"]),
        "stop_reason": telemetry["stop_reason"],
        "cancelled": "false",
        "error": error,
        "empty_output": bool_csv(flags["empty_output"]),
        "role_leak": bool_csv(flags["role_leak"]),
        "code_leak": bool_csv(flags["code_leak"]),
        "missing_word_suspected": bool_csv(flags["missing_word_suspected"]),
        "malformed_json": bool_csv(flags["malformed_json"]),
        "quality_flags": "|".join(quality_flag_names) if quality_flag_names else "none",
        "quality_score": number_or_empty(score),
        "output_preview": preview(output),
        "raw_output": output if args.raw_output else "",
    }
    return row, captured_logs, output


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        file.flush()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def grouped_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["task"],
            row["draft_model_label"],
            row["speculative_draft_tokens"],
            row["scenario_id"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (task, draft_label, draft_tokens, scenario_id), group_rows in sorted(groups.items()):
        errors = sum(1 for row in group_rows if row.get("error"))
        flagged = sum(
            1
            for row in group_rows
            if row.get("quality_flags") not in ("", "none", None)
        )
        avg_ttft = mean(row.get("ttft_ms") for row in group_rows)
        avg_total = mean(row.get("total_ms") for row in group_rows)
        avg_tps = mean(row.get("tokens_per_second") for row in group_rows)
        avg_acceptance = mean(row.get("draft_acceptance_rate") for row in group_rows)
        avg_quality = mean(row.get("quality_score") for row in group_rows)
        summary_rows.append(
            {
                "task": task,
                "draft_model_label": draft_label,
                "speculative_draft_tokens": draft_tokens,
                "scenario_id": scenario_id,
                "runs": len(group_rows),
                "errors": errors,
                "avg_ttft_ms": number_or_empty(round(avg_ttft, 3) if avg_ttft is not None else None),
                "avg_total_ms": number_or_empty(round(avg_total, 3) if avg_total is not None else None),
                "avg_tps": number_or_empty(round(avg_tps, 3) if avg_tps is not None else None),
                "avg_acceptance_rate": number_or_empty(round(avg_acceptance, 4) if avg_acceptance is not None else None),
                "quality_flag_rate": round(flagged / len(group_rows), 4) if group_rows else 0,
                "avg_quality_score": number_or_empty(round(avg_quality, 4) if avg_quality is not None else None),
            }
        )
    return summary_rows


def write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)


def try_import_matplotlib() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def numeric_rows(rows: list[dict[str, str]], key: str) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for row in rows:
        value = safe_float(row.get(key))
        if value is None:
            continue
        label = f"{row['draft_model_label']}:{row.get('speculative_draft_tokens') or '-'}"
        values.append((label, value))
    return values


def average_by_label(rows: list[dict[str, str]], metric: str, label_keys: tuple[str, ...]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = safe_float(row.get(metric))
        if value is None:
            continue
        label = " | ".join(str(row.get(key, "")) or "-" for key in label_keys)
        groups.setdefault(label, []).append(value)
    return {label: statistics.fmean(values) for label, values in groups.items() if values}


def bar_plot(plt: Any, output_path: Path, title: str, ylabel: str, values: dict[str, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        write_placeholder_plot(output_path, title, "No numeric data available")
        return
    labels = list(values.keys())
    data = [values[label] for label in labels]
    width = max(8, min(18, len(labels) * 1.1))
    plt.figure(figsize=(width, 5))
    plt.bar(labels, data, color="#3f3bbd")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_placeholder_plot(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="420">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        f'<text x="40" y="80" font-family="Arial" font-size="28" fill="#151823">{title}</text>'
        f'<text x="40" y="140" font-family="Arial" font-size="18" fill="#5b6275">{message}</text>'
        "</svg>"
    )
    path.with_suffix(".svg").write_text(svg, encoding="utf-8")


def generate_plots(results_csv: Path, output_dir: Path) -> list[Path]:
    rows = read_csv_rows(results_csv)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = try_import_matplotlib()
    generated: list[Path] = []

    plot_specs = [
        (
            "latency_by_model.png",
            "Average Total Latency by Model Mode",
            "Total latency (ms)",
            "total_ms",
            ("task", "draft_model_label"),
        ),
        (
            "ttft_by_model.png",
            "Average TTFT by Model Mode",
            "TTFT (ms)",
            "ttft_ms",
            ("task", "draft_model_label"),
        ),
        (
            "tps_by_model.png",
            "Average TPS by Model Mode",
            "Tokens per second",
            "tokens_per_second",
            ("task", "draft_model_label"),
        ),
        (
            "acceptance_by_draft_tokens.png",
            "Draft Acceptance by Draft Token Count",
            "Acceptance rate",
            "draft_acceptance_rate",
            ("draft_model_label", "speculative_draft_tokens"),
        ),
        (
            "output_tokens_by_scenario.png",
            "Output Tokens by Scenario",
            "Output tokens",
            "output_tokens",
            ("task", "scenario_id"),
        ),
        (
            "quality_score_by_model.png",
            "Average Quality Score by Model Mode",
            "Quality score",
            "quality_score",
            ("task", "draft_model_label"),
        ),
    ]

    for filename, title, ylabel, metric, label_keys in plot_specs:
        output_path = plots_dir / filename
        values = average_by_label(rows, metric, label_keys)
        if plt is None:
            write_placeholder_plot(output_path, title, "Install matplotlib to generate PNG plots")
            generated.append(output_path.with_suffix(".svg"))
            continue
        bar_plot(plt, output_path, title, ylabel, values)
        generated.append(output_path)

    flag_counts: dict[str, float] = {}
    for flag in ("empty_output", "role_leak", "code_leak", "missing_word_suspected", "malformed_json"):
        flag_counts[flag] = sum(1 for row in rows if row.get(flag) == "true")
    flag_plot = plots_dir / "quality_flag_counts.png"
    if plt is None:
        write_placeholder_plot(flag_plot, "Quality Flag Counts", "Install matplotlib to generate PNG plots")
        generated.append(flag_plot.with_suffix(".svg"))
    else:
        bar_plot(plt, flag_plot, "Quality Flag Counts", "Count", flag_counts)
        generated.append(flag_plot)

    error_values = average_by_label(
        [
            {
                **row,
                "error_rate": "1" if row.get("error") else "0",
            }
            for row in rows
        ],
        "error_rate",
        ("task", "draft_model_label"),
    )
    error_plot = plots_dir / "error_rate_by_model.png"
    if plt is None:
        write_placeholder_plot(error_plot, "Error Rate by Model", "Install matplotlib to generate PNG plots")
        generated.append(error_plot.with_suffix(".svg"))
    else:
        bar_plot(plt, error_plot, "Error Rate by Model", "Error rate", error_values)
        generated.append(error_plot)

    return generated


def write_summary_md(path: Path, rows: list[dict[str, str]], summary_rows: list[dict[str, Any]], plots: list[Path]) -> None:
    total = len(rows)
    errors = sum(1 for row in rows if row.get("error"))
    flagged = sum(1 for row in rows if row.get("quality_flags") not in ("", "none", None))
    avg_total = mean(row.get("total_ms") for row in rows)
    avg_ttft = mean(row.get("ttft_ms") for row in rows)
    avg_quality = mean(row.get("quality_score") for row in rows)
    lines = [
        "# Benchmark Summary",
        "",
        f"- Runs: {total}",
        f"- Errors: {errors}",
        f"- Runs with quality flags: {flagged}",
        f"- Average TTFT ms: {avg_ttft:.2f}" if avg_ttft is not None else "- Average TTFT ms: unavailable",
        f"- Average total ms: {avg_total:.2f}" if avg_total is not None else "- Average total ms: unavailable",
        f"- Average quality score: {avg_quality:.3f}" if avg_quality is not None else "- Average quality score: unavailable",
        "",
        "## Generated Files",
        "",
        "- `results.csv`",
        "- `summary.csv`",
        "- `events.jsonl`",
        "- `run_config.json`",
        "- `plots/`",
        "",
        "## Plots",
        "",
    ]
    for plot in plots:
        lines.append(f"- `{plot.relative_to(path.parent)}`")
    lines.extend(["", "## Grouped Summary", ""])
    for summary in summary_rows[:30]:
        lines.append(
            "- "
            f"{summary['task']} | {summary['scenario_id']} | {summary['draft_model_label']} | "
            f"runs={summary['runs']} errors={summary['errors']} "
            f"avg_total_ms={summary['avg_total_ms']} avg_quality={summary['avg_quality_score']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def quality_label_for_row(row: dict[str, str]) -> str:
    if row.get("task") == "autocomplete":
        return str(row.get("fast_path_mode", ""))
    return str(row.get("draft_model_label", ""))


def compare_quality_against_baseline(
    rows: list[dict[str, str]],
    baseline_label: str,
    candidate_label: str,
) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        label = quality_label_for_row(row)
        if label not in {baseline_label, candidate_label}:
            continue
        score = safe_float(row.get("quality_score"))
        if score is None:
            continue
        key = (str(row.get("task", "")), str(row.get("scenario_id", "")))
        group = by_group.setdefault(key, {baseline_label: [], candidate_label: []})
        group[label].append(score)

    comparisons: list[dict[str, Any]] = []
    for (task, scenario_id), grouped_scores in sorted(by_group.items()):
        baseline_mean = mean(grouped_scores.get(baseline_label, []))
        candidate_mean = mean(grouped_scores.get(candidate_label, []))
        if baseline_mean is None or candidate_mean is None:
            continue
        comparisons.append(
            {
                "task": task,
                "scenario_id": scenario_id,
                "baseline_label": baseline_label,
                "candidate_label": candidate_label,
                "baseline_quality": round(baseline_mean, 4),
                "candidate_quality": round(candidate_mean, 4),
                "quality_delta": round(candidate_mean - baseline_mean, 4),
            }
        )
    return comparisons


def write_quality_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=QUALITY_COMPARISON_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_quality_comparison_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Quality Comparison",
        "",
    ]
    if not rows:
        lines.extend(
            [
                "- No comparable rows were found.",
                "- Ensure both baseline and candidate labels exist in `results.csv` with non-empty quality scores.",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    avg_delta = mean(row["quality_delta"] for row in rows)
    lines.extend(
        [
            f"- Scenarios compared: {len(rows)}",
            (
                f"- Average quality delta (candidate - baseline): {avg_delta:.4f}"
                if avg_delta is not None
                else "- Average quality delta (candidate - baseline): unavailable"
            ),
            "",
            "| task | scenario_id | baseline | candidate | baseline_quality | candidate_quality | delta |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['scenario_id']} | {row['baseline_label']} | "
            f"{row['candidate_label']} | {row['baseline_quality']} | "
            f"{row['candidate_quality']} | {row['quality_delta']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_write_quality_comparison(
    output_dir: Path,
    rows: list[dict[str, str]],
    baseline_label: str,
    candidate_label: str,
) -> None:
    comparisons = compare_quality_against_baseline(rows, baseline_label, candidate_label)
    write_quality_comparison_csv(output_dir / "quality_comparison.csv", comparisons)
    write_quality_comparison_md(output_dir / "quality_comparison.md", comparisons)


def write_run_config(path: Path, args: argparse.Namespace, config: dict[str, Any], scenarios: list[dict[str, Any]], cases: list[BenchmarkCase]) -> None:
    payload = {
        "run_id": args.run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mock": args.mock,
        "dry_run": args.dry_run,
        "raw_output": args.raw_output,
        "target_model": args.target_model or os.getenv("LLAMA_REWRITE_TARGET_MODEL_DIR", ""),
        "draft_1b_model": args.draft_1b_model or os.getenv("LLAMA_REWRITE_DRAFT_MODEL_DIR", ""),
        "draft_3b_model": args.draft_3b_model or args.fast_model or os.getenv("LLAMA_FAST_MODEL_DIR", ""),
        "fast_model": args.fast_model or os.getenv("LLAMA_FAST_MODEL_DIR", ""),
        "qwen_model": args.qwen_model or os.getenv("QWEN_AUTOCOMPLETE_MODEL_DIR", ""),
        "granite_fim_model": args.granite_fim_model or os.getenv("GRANITE_FIM_MODEL_DIR", ""),
        "quality_baseline_label": args.quality_baseline_label,
        "quality_candidate_label": args.quality_candidate_label,
        "config": config,
        "scenario_count": len(scenarios),
        "case_count": len(cases),
        "cases": [
            {
                "task": case.task,
                "scenario_id": case.scenario["id"],
                "repetition": case.repetition,
                "draft_model_label": case.draft_model_label,
                "fast_path_mode": case.fast_path_mode,
                "speculative_draft_tokens": case.speculative_draft_tokens,
                "temperature": case.temperature,
                "top_p": case.top_p,
                "top_k": case.top_k,
                "repetition_penalty": case.repetition_penalty,
                "max_new_tokens": case.max_new_tokens,
            }
            for case in cases
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def print_case_header(index: int, total: int, case: BenchmarkCase) -> None:
    draft_tokens = case.speculative_draft_tokens if case.speculative_draft_tokens is not None else "-"
    print(
        f"[{index:03d}/{total:03d}] {case.task} | "
        f"scenario={case.scenario['id']} | draft={case.draft_model_label} | "
        f"draft_tokens={draft_tokens} | temp={case.temperature}"
    )


def print_case_result(row: dict[str, Any]) -> None:
    status = "error" if row["error"] else "ok"
    acceptance = safe_float(row["draft_acceptance_rate"])
    quality = safe_float(row["quality_score"])
    acceptance_text = f"{acceptance * 100:.0f}%" if acceptance is not None else "n/a"
    quality_text = f"{quality:.2f}" if quality is not None else "n/a"
    flags = row["quality_flags"] or "none"
    ttft_text = f"{row['ttft_ms']}ms" if row["ttft_ms"] not in ("", None) else "n/a"
    total_text = f"{row['total_ms']}ms" if row["total_ms"] not in ("", None) else "n/a"
    print(
        f"  {status} | ttft={ttft_text} | "
        f"total={total_text} | tps={row['tokens_per_second'] or 'n/a'} | "
        f"acceptance={acceptance_text} | quality={quality_text} | flags={flags}"
    )
    if row["error"]:
        print(f"  error={row['error']}")


def run_benchmarks(args: argparse.Namespace) -> Path:
    config = load_json(Path(args.config), DEFAULT_CONFIG)
    scenarios = load_json(Path(args.scenarios), DEFAULT_SCENARIOS)
    cases = expand_cases(config, scenarios, args)
    output_dir = Path(args.output_root) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    write_run_config(output_dir / "run_config.json", args, config, scenarios, cases)

    print(f"run_id={args.run_id}")
    print(f"output_dir={output_dir}")
    print(f"cases={len(cases)}")

    if args.dry_run:
        for index, case in enumerate(cases, start=1):
            print_case_header(index, len(cases), case)
        print("dry_run=true")
        return output_dir

    results_csv = output_dir / "results.csv"
    events_jsonl = output_dir / "events.jsonl"

    for index, case in enumerate(cases, start=1):
        print_case_header(index, len(cases), case)
        row, captured_logs, full_output = run_case(args, case)
        row["row_id"] = index
        write_csv_row(results_csv, row)
        append_jsonl(
            events_jsonl,
            {
                "event": "benchmark_case",
                "row": row,
                "full_output": full_output,
                "engine_logs": captured_logs.splitlines(),
            },
        )
        print_case_result(row)

    rows = read_csv_rows(results_csv)
    summary_rows = grouped_summary(rows)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    plots = generate_plots(results_csv, output_dir)
    write_summary_md(output_dir / "summary.md", rows, summary_rows, plots)
    if args.quality_baseline_label and args.quality_candidate_label:
        maybe_write_quality_comparison(
            output_dir=output_dir,
            rows=rows,
            baseline_label=args.quality_baseline_label,
            candidate_label=args.quality_candidate_label,
        )

    print(f"results_csv={results_csv}")
    print(f"summary_csv={output_dir / 'summary.csv'}")
    print(f"events_jsonl={events_jsonl}")
    print(f"plots_dir={output_dir / 'plots'}")
    return output_dir


def plot_only(args: argparse.Namespace) -> Path:
    results_csv = Path(args.plot_only).resolve()
    if not results_csv.exists():
        raise SystemExit(f"results CSV not found: {results_csv}")
    output_dir = results_csv.parent
    rows = read_csv_rows(results_csv)
    summary_rows = grouped_summary(rows)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    plots = generate_plots(results_csv, output_dir)
    write_summary_md(output_dir / "summary.md", rows, summary_rows, plots)
    if args.quality_baseline_label and args.quality_candidate_label:
        maybe_write_quality_comparison(
            output_dir=output_dir,
            rows=rows,
            baseline_label=args.quality_baseline_label,
            candidate_label=args.quality_candidate_label,
        )
    print(f"plot_only=true")
    print(f"results_csv={results_csv}")
    print(f"plots_dir={output_dir / 'plots'}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command benchmark runner for editor NLP/PDC inference experiments."
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS_PATH))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--run-id", default=f"benchmark_{timestamp}_{uuid.uuid4().hex[:6]}")
    parser.add_argument("--target-model", default=os.getenv("LLAMA_REWRITE_TARGET_MODEL_DIR", ""))
    parser.add_argument("--draft-1b-model", default=os.getenv("LLAMA_REWRITE_DRAFT_MODEL_DIR", ""))
    parser.add_argument("--draft-3b-model", default=os.getenv("LLAMA_FAST_MODEL_DIR", ""))
    parser.add_argument("--fast-model", default=os.getenv("LLAMA_FAST_MODEL_DIR", ""))
    parser.add_argument("--qwen-model", default=os.getenv("QWEN_AUTOCOMPLETE_MODEL_DIR", ""))
    parser.add_argument("--granite-fim-model", default=os.getenv("GRANITE_FIM_MODEL_DIR", ""))
    parser.add_argument(
        "--quality-baseline-label",
        default="",
        help="Baseline label for quality comparison output (e.g. draft_3b, qwen_fim).",
    )
    parser.add_argument(
        "--quality-candidate-label",
        default="",
        help="Candidate label for quality comparison output (e.g. draft_1b, granite_fim).",
    )
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--tasks", default="", help="Comma-separated subset: rewrite,autocomplete,correction")
    parser.add_argument("--draft-modes", default="", help="Comma-separated subset: target_only,draft_1b,draft_3b")
    parser.add_argument(
        "--fast-path-modes",
        default="",
        help="Comma-separated subset for autocomplete: llama_instruct,qwen_fim,granite_fim",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--raw-output", action="store_true")
    parser.add_argument("--plot-only", default="", help="Regenerate plots from an existing results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        plot_only(args)
        return
    run_benchmarks(args)


if __name__ == "__main__":
    main()
