"""
Hardware abstraction layer for editor inference engines.

Set USE_MOCK_ENGINE=true for local development without CUDA/ExLlamaV2. Real
mode imports ExLlamaV2 lazily so merely importing this module is safe on
non-GPU machines.
"""

from __future__ import annotations

import os
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, TypedDict


TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
DEFAULT_MAX_SEQ_LEN = 8192
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_REWRITE_TEMPERATURE = 0.2
DEFAULT_REWRITE_TOP_P = 0.8
DEFAULT_REWRITE_TOP_K = 40
DEFAULT_REWRITE_REPETITION_PENALTY = 1.05
DEFAULT_REWRITE_MAX_NEW_TOKENS = 128
DEFAULT_SPECULATIVE_DRAFT_TOKENS = 1
DEFAULT_ENABLE_SPECULATIVE_REWRITE = True
DEFAULT_FAST_PATH_MODE = "llama_instruct"
DEFAULT_AUTOCOMPLETE_MAX_NEW_TOKENS = 12
DEFAULT_CORRECTION_MAX_NEW_TOKENS = 256
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_PAD = "<|fim_pad|>"
QWEN_FIM_STOP_STRINGS = (
    FIM_PREFIX,
    FIM_SUFFIX,
    FIM_MIDDLE,
    FIM_PAD,
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "</s>",
)
LLAMA_REWRITE_STOP_STRINGS = (
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|reserved_special_token",
)
LLAMA3_STOP_TOKEN_IDS = (
    128001,  # <|end_of_text|>
    128009,  # <|eot_id|>
)
LLAMA_FAST_STOP_STRINGS = (
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|reserved_special_token",
    "\n",
)
AUTOCOMPLETE_CODE_PATTERNS = (
    "```",
    "import ",
    "from ",
    "def ",
    "class ",
    "function ",
    "const ",
    "let ",
    "var ",
    "{",
    "}",
    "# ",
    "##",
)
AUTOCOMPLETE_PREFACE_PATTERNS = (
    "here is",
    "here's",
    "sure",
    "certainly",
    "the continuation",
)
AUTOCOMPLETE_PROMPT_LABEL_PATTERNS = (
    "before cursor",
    "after cursor",
    "return only",
    "json",
    "text segment",
)
CorrectionReason = Literal["grammar", "typo", "punctuation", "clarity"]


class CorrectionSuggestion(TypedDict):
    start: int
    end: int
    replacement: str
    reason: CorrectionReason


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip().lower() in TRUTHY_ENV_VALUES


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def clamp_cursor_index(document_text: str, cursor_char_index: int) -> int:
    return min(max(cursor_char_index, 0), len(document_text))


def build_qwen_fim_prompt(document_text: str, cursor_char_index: int) -> str:
    cursor_char_index = clamp_cursor_index(document_text, cursor_char_index)
    prefix = document_text[:cursor_char_index]
    suffix = document_text[cursor_char_index:]
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


def build_llama_autocomplete_prompt(
    document_text: str,
    cursor_char_index: int,
    *,
    prefix_chars: int = 1200,
    suffix_chars: int = 300,
) -> str:
    cursor_char_index = clamp_cursor_index(document_text, cursor_char_index)
    prefix = document_text[max(0, cursor_char_index - prefix_chars):cursor_char_index]
    suffix = document_text[cursor_char_index:cursor_char_index + suffix_chars]
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an editor autocomplete engine for normal prose. Continue the "
        "text at the cursor with 1 to 8 words. Return only the continuation. "
        "Do not write code, markdown, labels, explanations, alternatives, or a "
        "complete paragraph. The continuation must fit naturally before the "
        "suffix if one is provided."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Before cursor:\n{prefix}\n\n"
        f"After cursor:\n{suffix}\n\n"
        "Return only the next words after the cursor."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_llama_correction_prompt(
    document_text: str,
    cursor_char_index: int,
    *,
    window_chars: int = 1800,
) -> tuple[str, int, str]:
    cursor_char_index = clamp_cursor_index(document_text, cursor_char_index)
    start = max(0, cursor_char_index - window_chars // 2)
    end = min(len(document_text), cursor_char_index + window_chars // 2)
    segment = document_text[start:end]
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a careful prose correction engine. Find only clear typos, "
        "grammar, punctuation, or clarity issues in the provided text segment. "
        "Return strict JSON only: an array of objects with start, end, "
        "replacement, and reason. start and end must be character offsets "
        "relative to the provided segment. reason must be one of grammar, typo, "
        "punctuation, clarity. Return [] if no correction is needed."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Text segment:\n{segment}\n\n"
        "JSON:"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt, start, segment


def build_llama_rewrite_prompt(text: str, instruction: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a precise text rewriting engine. Rewrite the selected text "
        "according to the user's instruction. Output only the final rewritten "
        "text. Do not add a preface, label, explanation, quote marks, markdown, "
        "or alternate version. Preserve the original meaning and fill in all "
        "necessary words so the sentence is grammatical. The output must be "
        "one complete sentence or paragraph, not a fragment."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Instruction: {instruction}\n\n"
        f"Selected text:\n{text}\n\n"
        "Return only the rewritten selected text."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


@dataclass
class ModelBundle:
    model_dir: str
    config: Any
    model: Any
    cache: Any
    tokenizer: Any | None = None


@dataclass
class TelemetryStats:
    label: str
    started_at: float
    first_token_at: Optional[float] = None
    tokens: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0

    def observe_chunk(self, chunk: str, token_count: int = 1) -> None:
        if not chunk:
            return
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()
        self.tokens += max(1, token_count)

    def observe_speculative_result(self, result: dict[str, Any]) -> None:
        draft_tokens = _first_int_value(
            result,
            (
                "draft_tokens",
                "num_draft_tokens",
                "drafted_tokens",
                "attempted_draft_tokens",
            ),
        )
        accepted_tokens = _first_int_value(
            result,
            (
                "accepted_tokens",
                "num_accepted_tokens",
                "accepted_draft_tokens",
                "draft_tokens_accepted",
            ),
        )
        if draft_tokens is not None:
            self.draft_tokens += draft_tokens
        if accepted_tokens is not None:
            self.accepted_tokens += accepted_tokens

    def observe_speculative_job(self, job: Any) -> None:
        accepted_tokens = _int_attr(job, "accepted_draft_tokens")
        rejected_tokens = _int_attr(job, "rejected_draft_tokens")
        if accepted_tokens is None and rejected_tokens is None:
            return

        self.accepted_tokens = accepted_tokens or 0
        self.draft_tokens = self.accepted_tokens + (rejected_tokens or 0)

    def acceptance_summary(self) -> str:
        if self.draft_tokens <= 0:
            return "unavailable"
        return f"{self.accepted_tokens}/{self.draft_tokens}"

    def log(self) -> None:
        ended_at = time.perf_counter()
        total_ms = int((ended_at - self.started_at) * 1000)
        ttft_ms = (
            int((self.first_token_at - self.started_at) * 1000)
            if self.first_token_at is not None
            else total_ms
        )
        generation_seconds = (
            ended_at - self.first_token_at
            if self.first_token_at is not None
            else 0.0
        )
        tps = self.tokens / generation_seconds if generation_seconds > 0 else 0.0

        if self.label == "HEAVY-PATH":
            acceptance = (
                f"{(self.accepted_tokens / self.draft_tokens) * 100:.0f}%"
                if self.draft_tokens > 0
                else "unavailable"
            )
            print(
                f"[{self.label}] TTFT: {ttft_ms}ms | TPS: {tps:.1f} | "
                f"Draft Acceptance: {acceptance} | "
                f"Total Time: {total_ms}ms | Tokens: {self.tokens}",
                flush=True,
            )
            return

        print(
            f"[{self.label}] TTFT: {ttft_ms}ms | TPS: {tps:.1f} | "
            f"Total Time: {total_ms}ms | Tokens: {self.tokens}",
            flush=True,
        )


def _first_int_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return None


def _int_attr(obj: Any, name: str) -> Optional[int]:
    value = getattr(obj, name, None)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _first_present_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def trim_at_stop_strings(text: str, stop_strings: tuple[str, ...]) -> str:
    stop_index: Optional[int] = None
    for stop_string in stop_strings:
        index = text.find(stop_string)
        if index != -1 and (stop_index is None or index < stop_index):
            stop_index = index
    return text if stop_index is None else text[:stop_index]


def sanitize_autocomplete_completion(
    raw_completion: str,
    document_text: str,
    cursor_char_index: int,
) -> str:
    if any(pattern in raw_completion for pattern in AUTOCOMPLETE_CODE_PATTERNS):
        return ""
    completion = trim_at_stop_strings(raw_completion, LLAMA_FAST_STOP_STRINGS)
    completion = completion.replace("\r", " ").replace("\n", " ").strip()
    completion = completion.strip("\"'` ")
    if not completion:
        return ""

    completion = strip_autocomplete_prompt_label(completion)
    if not completion:
        return ""

    lower_completion = completion.lower()
    if any(pattern in lower_completion for pattern in AUTOCOMPLETE_PREFACE_PATTERNS):
        return ""
    if starts_with_autocomplete_prompt_label(lower_completion):
        return ""
    if any(pattern in completion for pattern in AUTOCOMPLETE_CODE_PATTERNS):
        return ""

    # Keep ghost text short and phrase-like.
    sentence_match = re.search(r"([.!?])\s+", completion)
    if sentence_match:
        completion = completion[: sentence_match.end(1)]
    words = completion.split()
    if len(words) > 8:
        completion = " ".join(words[:8])
    if completion.count(".") + completion.count("!") + completion.count("?") > 1:
        return ""

    cursor_char_index = clamp_cursor_index(document_text, cursor_char_index)
    prefix_tail = document_text[max(0, cursor_char_index - 120):cursor_char_index].lower()
    suffix_head = document_text[cursor_char_index:cursor_char_index + 120].lower()
    if completion.lower() and completion.lower() in prefix_tail:
        return ""
    if suffix_head.strip() and suffix_head.lstrip().startswith(completion.lower().strip()):
        return ""
    return completion


def starts_with_autocomplete_prompt_label(lower_completion: str) -> bool:
    return any(
        lower_completion.startswith(pattern)
        for pattern in AUTOCOMPLETE_PROMPT_LABEL_PATTERNS
    )


def strip_autocomplete_prompt_label(completion: str) -> str:
    stripped = completion.strip()
    match = re.match(
        r"^(before cursor|after cursor|return only(?: the next words after the cursor)?|json|text segment)\s*:?\s*",
        stripped,
        flags=re.IGNORECASE,
    )
    if not match:
        return stripped
    return stripped[match.end() :].strip("\"'` ")


def parse_correction_suggestions(
    raw_completion: str,
    segment_start: int,
    segment_text: str,
) -> list[CorrectionSuggestion]:
    text = trim_at_stop_strings(raw_completion, LLAMA_FAST_STOP_STRINGS).strip()
    if not text:
        return []
    start_index = text.find("[")
    end_index = text.rfind("]")
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return []
    try:
        parsed = json.loads(text[start_index : end_index + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    suggestions: list[CorrectionSuggestion] = []
    seen_spans: set[tuple[int, int, str]] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        replacement = item.get("replacement")
        reason = item.get("reason")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if not isinstance(replacement, str) or not replacement.strip():
            continue
        if reason not in {"grammar", "typo", "punctuation", "clarity"}:
            continue
        if start < 0 or end <= start or end > len(segment_text):
            continue
        if segment_text[start:end] == replacement:
            continue
        absolute_start = segment_start + start
        absolute_end = segment_start + end
        key = (absolute_start, absolute_end, replacement)
        if key in seen_spans:
            continue
        seen_spans.add(key)
        suggestions.append(
            {
                "start": absolute_start,
                "end": absolute_end,
                "replacement": replacement,
                "reason": reason,
            }
        )
    return suggestions[:5]


class MockExLlamaEngine:
    """CPU-only stand-in with API parity for local WebSocket development."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.gpu_lock = threading.Lock()
        self.current_seq_len = 0
        self.prefill_history: list[list[int]] = []
        self.truncation_history: list[int] = []
        self.requires_full_prefill = False
        self.rewrite_max_new_tokens = DEFAULT_REWRITE_MAX_NEW_TOKENS
        self.autocomplete_max_new_tokens = DEFAULT_AUTOCOMPLETE_MAX_NEW_TOKENS

    def reset(self) -> None:
        with self.gpu_lock:
            self.current_seq_len = 0
            self.prefill_history.clear()
            self.truncation_history.clear()
            self.requires_full_prefill = False

    def truncate(self, safe_token_index: int) -> None:
        with self.gpu_lock:
            if safe_token_index < 0:
                raise ValueError(
                    f"truncate index must be >= 0, got {safe_token_index}"
                )
            if safe_token_index > self.current_seq_len:
                raise ValueError(
                    f"Cannot truncate to {safe_token_index}; current_seq_len "
                    f"is only {self.current_seq_len}"
                )
            self.truncation_history.append(safe_token_index)
            self.current_seq_len = safe_token_index
            self.requires_full_prefill = False

    def forward_prefill(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("forward_prefill called with an empty token list")
        with self.gpu_lock:
            self.prefill_history.append(list(token_ids))
            self.current_seq_len += len(token_ids)
            self.requires_full_prefill = False

    def generate_stream(
        self,
        cancel_event: Any,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Iterator[str]:
        token = " word"
        for _ in range(max_new_tokens):
            if cancel_event.is_set():
                break
            time.sleep(0.05)
            if cancel_event.is_set():
                break
            yield token

    def generate_autocomplete_stream(
        self,
        document_text: str,
        cursor_char_index: int,
        cancel_event: Any,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Iterator[str]:
        suggestion = sanitize_autocomplete_completion(
            " next phrase",
            document_text,
            cursor_char_index,
        )
        if suggestion and not cancel_event.is_set():
            time.sleep(0.05)
            yield suggestion

    def generate_corrections(
        self,
        document_text: str,
        cursor_char_index: int,
        cancel_event: Any,
    ) -> list[CorrectionSuggestion]:
        if cancel_event.is_set():
            return []
        return []

    def apply_rewrite(
        self,
        text: str,
        instruction: str,
        cancel_event: Any,
        max_new_tokens: int = DEFAULT_REWRITE_MAX_NEW_TOKENS,
    ) -> Iterator[str]:
        chunks = [f"Rewritten ({instruction}): ", text]
        for chunk in chunks:
            if cancel_event.is_set():
                break
            time.sleep(0.05)
            if cancel_event.is_set():
                break
            yield chunk


class RealExLlamaEngine:
    """ExLlamaV2-backed multi-model engine with serialized GPU access."""

    def __init__(
        self,
        qwen_model_dir: str,
        llama_fast_model_dir: str | None,
        llama_target_model_dir: str,
        llama_draft_model_dir: str | None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.gpu_lock = threading.Lock()
        self.max_seq_len = max_seq_len
        self.default_max_new_tokens = max_new_tokens
        self.current_seq_len = 0
        self._qwen_last_token_id: Optional[int] = None
        self._qwen_next_logits: Optional[Any] = None
        self.requires_full_prefill = False
        self.fast_path_mode = os.getenv("FAST_PATH_MODE", DEFAULT_FAST_PATH_MODE).strip().lower()
        if self.fast_path_mode not in {"llama_instruct", "qwen_fim"}:
            self.fast_path_mode = DEFAULT_FAST_PATH_MODE
        self.autocomplete_max_new_tokens = max(
            1,
            _env_int(
                "AUTOCOMPLETE_MAX_NEW_TOKENS",
                DEFAULT_AUTOCOMPLETE_MAX_NEW_TOKENS,
            ),
        )
        self.correction_max_new_tokens = max(
            1,
            _env_int(
                "CORRECTION_MAX_NEW_TOKENS",
                DEFAULT_CORRECTION_MAX_NEW_TOKENS,
            ),
        )
        self.enable_speculative_rewrite = _env_bool(
            "ENABLE_SPECULATIVE_REWRITE",
            DEFAULT_ENABLE_SPECULATIVE_REWRITE,
        )
        self.debug_speculative_rewrite = _env_bool(
            "DEBUG_SPECULATIVE_REWRITE",
            False,
        )
        self.rewrite_max_new_tokens = max(
            1,
            _env_int("REWRITE_MAX_NEW_TOKENS", DEFAULT_REWRITE_MAX_NEW_TOKENS),
        )
        self.speculative_draft_tokens = _env_int(
            "SPECULATIVE_DRAFT_TOKENS",
            DEFAULT_SPECULATIVE_DRAFT_TOKENS,
        )
        self.qwen_temperature = max(
            0.0,
            _env_float("GENERATION_TEMPERATURE", DEFAULT_TEMPERATURE),
        )
        self.qwen_top_p = min(
            1.0,
            max(0.0, _env_float("GENERATION_TOP_P", DEFAULT_TOP_P)),
        )
        self.qwen_top_k = max(0, _env_int("GENERATION_TOP_K", DEFAULT_TOP_K))
        self.qwen_repetition_penalty = max(
            1.0,
            _env_float("GENERATION_REPETITION_PENALTY", DEFAULT_REPETITION_PENALTY),
        )

        self.rewrite_temperature = max(
            0.0,
            _env_float("REWRITE_TEMPERATURE", DEFAULT_REWRITE_TEMPERATURE),
        )
        self.rewrite_top_p = min(
            1.0,
            max(0.0, _env_float("REWRITE_TOP_P", DEFAULT_REWRITE_TOP_P)),
        )
        self.rewrite_top_k = max(0, _env_int("REWRITE_TOP_K", DEFAULT_REWRITE_TOP_K))
        self.rewrite_repetition_penalty = max(
            1.0,
            _env_float(
                "REWRITE_REPETITION_PENALTY",
                DEFAULT_REWRITE_REPETITION_PENALTY,
            ),
        )

        try:
            import torch
            from exllamav2 import (
                ExLlamaV2,
                ExLlamaV2Cache_8bit,
                ExLlamaV2Cache_Q4,
                ExLlamaV2Config,
                ExLlamaV2Tokenizer,
            )
            from exllamav2.generator import (
                ExLlamaV2DynamicGenerator,
                ExLlamaV2DynamicJob,
                ExLlamaV2Sampler,
                ExLlamaV2StreamingGenerator,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "RealExLlamaEngine requires exllamav2 and torch. Set "
                "USE_MOCK_ENGINE=true for local non-CUDA development."
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "This ExLlamaV2 build does not expose the dynamic/speculative "
                "generator. Upgrade ExLlamaV2 before enabling Phase 6 rewrite."
            ) from exc

        self.torch = torch
        self.qwen_cache_cls = ExLlamaV2Cache_8bit
        self.rewrite_cache_cls = ExLlamaV2Cache_Q4
        self.config_cls = ExLlamaV2Config
        self.model_cls = ExLlamaV2
        self.tokenizer_cls = ExLlamaV2Tokenizer
        self.sampler_cls = ExLlamaV2Sampler
        self.streaming_generator_cls = ExLlamaV2StreamingGenerator
        self.dynamic_generator_cls = ExLlamaV2DynamicGenerator
        self.dynamic_job_cls = ExLlamaV2DynamicJob

        # Load largest first to reduce VRAM fragmentation.
        self.llama_target = self._load_model_bundle(
            "LLAMA_REWRITE_TARGET_MODEL_DIR",
            llama_target_model_dir,
            load_tokenizer=True,
            cache_cls=self.rewrite_cache_cls,
            cache_name="Q4 rewrite target",
        )
        self.qwen: ModelBundle | None = None
        self.fast_llama: ModelBundle | None = None
        if self.fast_path_mode == "qwen_fim":
            self.qwen = self._load_model_bundle(
                "QWEN_AUTOCOMPLETE_MODEL_DIR",
                qwen_model_dir,
                load_tokenizer=True,
                cache_cls=self.qwen_cache_cls,
                cache_name="8-bit Qwen autocomplete",
            )
        else:
            if not llama_fast_model_dir:
                raise RuntimeError(
                    "LLAMA_FAST_MODEL_DIR is required when FAST_PATH_MODE is "
                    "llama_instruct"
                )
            self.fast_llama = self._load_model_bundle(
                "LLAMA_FAST_MODEL_DIR",
                llama_fast_model_dir,
                load_tokenizer=True,
                cache_cls=self.qwen_cache_cls,
                cache_name="8-bit Llama fast path",
            )
        self.llama_draft: ModelBundle | None = None
        if self.enable_speculative_rewrite:
            if not llama_draft_model_dir:
                raise RuntimeError(
                    "LLAMA_REWRITE_DRAFT_MODEL_DIR is required when "
                    "ENABLE_SPECULATIVE_REWRITE is true"
                )
            self.llama_draft = self._load_model_bundle(
                "LLAMA_REWRITE_DRAFT_MODEL_DIR",
                llama_draft_model_dir,
                load_tokenizer=self.debug_speculative_rewrite,
                cache_cls=self.rewrite_cache_cls,
                cache_name="Q4 rewrite draft",
            )

        if self.qwen is not None:
            self.qwen_generator = self.streaming_generator_cls(
                self.qwen.model,
                self.qwen.cache,
                self.qwen.tokenizer,
            )
        else:
            assert self.fast_llama is not None
            self.fast_generator = self.streaming_generator_cls(
                self.fast_llama.model,
                self.fast_llama.cache,
                self.fast_llama.tokenizer,
            )
        self.fast_sampling_settings = self._build_sampling_settings(
            self.qwen_temperature,
            self.qwen_top_p,
            self.qwen_top_k,
            self.qwen_repetition_penalty,
        )
        self.qwen_sampling_settings = self.fast_sampling_settings

        if self.enable_speculative_rewrite:
            assert self.llama_draft is not None
            self.rewrite_generator = self.dynamic_generator_cls(
                model=self.llama_target.model,
                cache=self.llama_target.cache,
                tokenizer=self.llama_target.tokenizer,
                max_seq_len=max_seq_len,
                draft_model=self.llama_draft.model,
                draft_cache=self.llama_draft.cache,
                num_draft_tokens=self.speculative_draft_tokens,
            )
        else:
            self.rewrite_generator = self.streaming_generator_cls(
                self.llama_target.model,
                self.llama_target.cache,
                self.llama_target.tokenizer,
            )
        self.rewrite_sampling_settings = self._build_sampling_settings(
            self.rewrite_temperature,
            self.rewrite_top_p,
            self.rewrite_top_k,
            self.rewrite_repetition_penalty,
        )
        self._log_speculative_startup_debug()

    def _load_model_bundle(
        self,
        env_name: str,
        model_dir: str,
        *,
        load_tokenizer: bool,
        cache_cls: Any,
        cache_name: str,
    ) -> ModelBundle:
        resolved_model_dir = str(Path(model_dir).expanduser())
        if not Path(resolved_model_dir).exists():
            raise FileNotFoundError(f"{env_name} does not exist: {resolved_model_dir}")

        config = self.config_cls(resolved_model_dir)
        if hasattr(config, "prepare"):
            config.prepare()
        if hasattr(config, "arch_compat_overrides"):
            config.arch_compat_overrides()

        model = self.model_cls(config)
        cache = self._build_cache(model, cache_cls, cache_name)
        if hasattr(model, "load_autosplit"):
            model.load_autosplit(cache)
        else:
            model.load()

        tokenizer = self._build_tokenizer(config) if load_tokenizer else None
        return ModelBundle(
            model_dir=resolved_model_dir,
            config=config,
            model=model,
            cache=cache,
            tokenizer=tokenizer,
        )

    def _build_cache(self, model: Any, cache_cls: Any, cache_name: str) -> Any:
        candidates = (
            {"batch_size": 1, "max_seq_len": self.max_seq_len, "lazy": True},
            {"max_seq_len": self.max_seq_len, "lazy": True},
            {"batch_size": 1, "max_seq_len": self.max_seq_len},
            {"max_seq_len": self.max_seq_len},
        )
        last_error: Optional[TypeError] = None

        for kwargs in candidates:
            try:
                return cache_cls(model, **kwargs)
            except TypeError as exc:
                last_error = exc

        raise RuntimeError(f"Could not initialize ExLlamaV2 {cache_name} cache") from last_error

    def _build_tokenizer(self, config: Any) -> Any:
        try:
            return self.tokenizer_cls(config)
        except TypeError:
            return self.tokenizer_cls(config.model_dir)

    def _build_sampling_settings(
        self,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Any:
        settings = self.sampler_cls.Settings()
        settings.temperature = temperature
        settings.top_p = top_p
        settings.top_k = top_k
        settings.token_repetition_penalty = repetition_penalty
        if hasattr(settings, "token_repetition_range"):
            settings.token_repetition_range = -1
        if hasattr(settings, "dry_multiplier"):
            settings.dry_multiplier = _env_float("GENERATION_DRY_MULTIPLIER", 0.35)
        if hasattr(settings, "dry_allowed_length"):
            settings.dry_allowed_length = _env_int("GENERATION_DRY_ALLOWED_LENGTH", 2)
        if hasattr(settings, "dry_base"):
            settings.dry_base = _env_float("GENERATION_DRY_BASE", 1.75)
        if hasattr(settings, "dry_range"):
            settings.dry_range = _env_int("GENERATION_DRY_RANGE", 512)
        return settings

    def reset(self) -> None:
        with self.gpu_lock:
            if self.qwen is not None:
                self.qwen.cache.current_seq_len = 0
            if self.fast_llama is not None:
                self.fast_llama.cache.current_seq_len = 0
            self.current_seq_len = 0
            self._qwen_last_token_id = None
            self._qwen_next_logits = None
            self.requires_full_prefill = False

    def truncate(self, safe_token_index: int) -> None:
        with self.gpu_lock:
            if safe_token_index < 0:
                raise ValueError(
                    f"truncate index must be >= 0, got {safe_token_index}"
                )
            if safe_token_index > self.current_seq_len:
                raise ValueError(
                    f"Cannot truncate to {safe_token_index}; current_seq_len "
                    f"is only {self.current_seq_len}"
                )
            if self.qwen is not None:
                self.qwen.cache.current_seq_len = safe_token_index
            if self.fast_llama is not None:
                self.fast_llama.cache.current_seq_len = 0
            self.current_seq_len = safe_token_index
            if safe_token_index == 0:
                self._qwen_last_token_id = None
            self._qwen_next_logits = None
            self.requires_full_prefill = False

    def forward_prefill(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("forward_prefill called with an empty token list")

        with self.gpu_lock, self.torch.inference_mode():
            if self.qwen is None:
                self.current_seq_len += len(token_ids)
                self.requires_full_prefill = False
                return
            input_ids = self.torch.tensor([token_ids], dtype=self.torch.long)
            self._qwen_next_logits = self.qwen.model.forward(input_ids, self.qwen.cache)
            self.current_seq_len = self.qwen.cache.current_seq_len
            self._qwen_last_token_id = int(token_ids[-1])
            self.requires_full_prefill = False

    def generate_autocomplete_stream(
        self,
        document_text: str,
        cursor_char_index: int,
        cancel_event: Any,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Iterator[str]:
        if self.fast_path_mode == "llama_instruct":
            yield from self._generate_llama_autocomplete_stream(
                document_text,
                cursor_char_index,
                cancel_event,
                min(max_new_tokens, self.autocomplete_max_new_tokens),
            )
            return

        telemetry = TelemetryStats("FAST-PATH", time.perf_counter())
        assert self.qwen is not None
        prompt = build_qwen_fim_prompt(document_text, cursor_char_index)
        prompt_token_ids = self._encode_prompt(self.qwen.tokenizer, prompt)
        if not prompt_token_ids:
            telemetry.log()
            return

        input_ids = self.torch.tensor([prompt_token_ids], dtype=self.torch.long)

        try:
            with self.gpu_lock, self.torch.inference_mode():
                self.qwen_generator.set_stop_conditions(self._autocomplete_stop_conditions())
                self.qwen_generator.begin_stream_ex(
                    input_ids,
                    self.qwen_sampling_settings,
                    # FIM prompts must end on the literal <|fim_middle|> token.
                    # Token healing can rewrite that boundary token and make Qwen
                    # continue with newline/EOS instead of filling the middle.
                    token_healing=False,
                )
                self.current_seq_len = self.qwen.cache.current_seq_len
                self._qwen_last_token_id = int(prompt_token_ids[-1])
                self._qwen_next_logits = None
                self.requires_full_prefill = True

            for _ in range(max_new_tokens):
                if cancel_event.is_set():
                    break

                with self.gpu_lock, self.torch.inference_mode():
                    result = self.qwen_generator.stream_ex()
                    self.current_seq_len = self.qwen.cache.current_seq_len

                chunk = result.get("chunk", "")
                if chunk:
                    telemetry.observe_chunk(
                        chunk,
                        self._result_token_count(result),
                    )
                    yield chunk
                if result.get("eos", False):
                    break
        finally:
            telemetry.log()

    def _generate_llama_autocomplete_stream(
        self,
        document_text: str,
        cursor_char_index: int,
        cancel_event: Any,
        max_new_tokens: int,
    ) -> Iterator[str]:
        telemetry = TelemetryStats("FAST-PATH", time.perf_counter())
        assert self.fast_llama is not None
        prompt = build_llama_autocomplete_prompt(document_text, cursor_char_index)
        prompt_token_ids = self._encode_prompt(self.fast_llama.tokenizer, prompt)
        if not prompt_token_ids:
            telemetry.log()
            return

        input_ids = self.torch.tensor([prompt_token_ids], dtype=self.torch.long)
        raw_completion = ""
        try:
            with self.gpu_lock, self.torch.inference_mode():
                self.fast_generator.set_stop_conditions(self._llama_fast_stop_conditions())
                self.fast_generator.begin_stream_ex(
                    input_ids,
                    self.fast_sampling_settings,
                    token_healing=False,
                )

            for _ in range(max_new_tokens):
                if cancel_event.is_set():
                    break

                with self.gpu_lock, self.torch.inference_mode():
                    result = self.fast_generator.stream_ex()

                chunk = str(result.get("chunk") or result.get("text") or "")
                if chunk:
                    raw_completion += chunk
                    telemetry.observe_chunk(chunk, self._result_token_count(result))
                if result.get("eos", False):
                    break
                if any(stop in raw_completion for stop in LLAMA_FAST_STOP_STRINGS):
                    break

            suggestion = sanitize_autocomplete_completion(
                raw_completion,
                document_text,
                cursor_char_index,
            )
            if suggestion and not cancel_event.is_set():
                yield suggestion
        finally:
            telemetry.log()

    def generate_corrections(
        self,
        document_text: str,
        cursor_char_index: int,
        cancel_event: Any,
    ) -> list[CorrectionSuggestion]:
        if self.fast_path_mode != "llama_instruct":
            return []
        telemetry = TelemetryStats("CORRECTION-PATH", time.perf_counter())
        assert self.fast_llama is not None
        prompt, segment_start, segment_text = build_llama_correction_prompt(
            document_text,
            cursor_char_index,
        )
        prompt_token_ids = self._encode_prompt(self.fast_llama.tokenizer, prompt)
        if not prompt_token_ids:
            telemetry.log()
            return []

        input_ids = self.torch.tensor([prompt_token_ids], dtype=self.torch.long)
        raw_completion = ""
        try:
            with self.gpu_lock, self.torch.inference_mode():
                self.fast_generator.set_stop_conditions(self._llama_fast_stop_conditions())
                self.fast_generator.begin_stream_ex(
                    input_ids,
                    self.fast_sampling_settings,
                    token_healing=False,
                )

            for _ in range(self.correction_max_new_tokens):
                if cancel_event.is_set():
                    return []

                with self.gpu_lock, self.torch.inference_mode():
                    result = self.fast_generator.stream_ex()

                chunk = str(result.get("chunk") or result.get("text") or "")
                if chunk:
                    raw_completion += chunk
                    telemetry.observe_chunk(chunk, self._result_token_count(result))
                if result.get("eos", False):
                    break
                if "<|eot_id|>" in raw_completion or "<|end_of_text|>" in raw_completion:
                    break
            return parse_correction_suggestions(
                raw_completion,
                segment_start,
                segment_text,
            )
        finally:
            telemetry.log()

    def apply_rewrite(
        self,
        text: str,
        instruction: str,
        cancel_event: Any,
        max_new_tokens: int | None = None,
    ) -> Iterator[str]:
        max_new_tokens = max_new_tokens or self.rewrite_max_new_tokens
        telemetry = TelemetryStats("HEAVY-PATH", time.perf_counter())
        prompt = build_llama_rewrite_prompt(text, instruction)
        input_ids = self._encode_prompt(self.llama_target.tokenizer, prompt)
        if not input_ids:
            telemetry.log()
            return
        input_tensor = self.torch.tensor([input_ids], dtype=self.torch.long)
        if self.debug_speculative_rewrite:
            self._log_rewrite_request_debug(
                prompt,
                input_ids,
                max_new_tokens,
            )

        if not self.enable_speculative_rewrite:
            try:
                yield from self._consume_target_rewrite_stream(
                    input_tensor,
                    cancel_event,
                    telemetry,
                    max_new_tokens,
                )
            finally:
                telemetry.log()
            return

        job = self._build_rewrite_job(input_tensor, max_new_tokens)

        try:
            yield from self._consume_dynamic_job(
                job,
                cancel_event,
                telemetry,
                max_new_tokens,
            )
        finally:
            telemetry.observe_speculative_job(job)
            telemetry.log()

    def _consume_target_rewrite_stream(
        self,
        input_tensor: Any,
        cancel_event: Any,
        telemetry: TelemetryStats,
        max_new_tokens: int,
    ) -> Iterator[str]:
        with self.gpu_lock, self.torch.inference_mode():
            self.rewrite_generator.set_stop_conditions(self._rewrite_stop_conditions())
            self.rewrite_generator.begin_stream_ex(
                input_tensor,
                self.rewrite_sampling_settings,
                token_healing=False,
            )

        for _ in range(max_new_tokens):
            if cancel_event.is_set():
                self._log_rewrite_stop("cancelled")
                break

            with self.gpu_lock, self.torch.inference_mode():
                result = self.rewrite_generator.stream_ex()

            chunk = str(result.get("chunk") or result.get("text") or "")
            if chunk:
                trimmed_chunk, stop_string = self._trim_at_rewrite_stop(chunk)
                if trimmed_chunk:
                    telemetry.observe_chunk(
                        trimmed_chunk,
                        self._result_token_count(result),
                    )
                    yield trimmed_chunk
                if stop_string:
                    self._log_rewrite_stop(f"text stop {stop_string!r}", result)
                    return

            if result.get("eos", False):
                self._log_rewrite_stop("eos", result)
                return

    def _build_rewrite_job(self, input_tensor: Any, max_new_tokens: int) -> Any:
        job_candidates = (
            {
                "input_ids": input_tensor,
                "stop_conditions": self._rewrite_stop_conditions(),
                "gen_settings": self.rewrite_sampling_settings,
                "max_new_tokens": max_new_tokens,
            },
            {
                "input_ids": input_tensor,
                "banned_strings": list(LLAMA_REWRITE_STOP_STRINGS),
                "gen_settings": self.rewrite_sampling_settings,
                "max_new_tokens": max_new_tokens,
            },
            {
                "input_ids": input_tensor,
                "gen_settings": self.rewrite_sampling_settings,
                "max_new_tokens": max_new_tokens,
            },
        )

        last_error: Optional[TypeError] = None
        for kwargs in job_candidates:
            try:
                return self.dynamic_job_cls(**kwargs)
            except TypeError as exc:
                last_error = exc

        raise RuntimeError("Could not initialize ExLlamaV2 dynamic rewrite job") from last_error

    def _consume_dynamic_job(
        self,
        job: Any,
        cancel_event: Any,
        telemetry: TelemetryStats,
        max_new_tokens: int,
    ) -> Iterator[str]:
        with self.gpu_lock, self.torch.inference_mode():
            self.rewrite_generator.enqueue(job)

        emitted_tokens = 0
        emitted_text = ""
        iteration = 0
        empty_iterations = 0
        try:
            while not cancel_event.is_set():
                iteration += 1
                with self.gpu_lock, self.torch.inference_mode():
                    results = self.rewrite_generator.iterate()

                if results:
                    empty_iterations = 0
                else:
                    empty_iterations += 1

                if self.debug_speculative_rewrite and (
                    results or empty_iterations == 1 or empty_iterations % 100 == 0
                ):
                    print(
                        "[HEAVY-PATH][spec] "
                        f"iterate={iteration} results={len(results) if results else 0} "
                        f"empty_iterations={empty_iterations} "
                        f"job={self._job_debug_state(job)}",
                        flush=True,
                    )

                if not results:
                    if self._job_is_complete(job):
                        self._log_rewrite_stop("generator completion without eos")
                        return
                    continue

                for result_index, result in enumerate(results):
                    result_job = result.get("job")
                    if result_job is not None and result_job is not job:
                        if self.debug_speculative_rewrite:
                            print(
                                "[HEAVY-PATH][spec] "
                                f"iterate={iteration} result={result_index} skipped foreign job",
                                flush=True,
                            )
                        continue

                    telemetry.observe_speculative_result(result)
                    token_count = self._result_token_count(result)
                    chunk = self._dynamic_result_text_delta(result)
                    if self.debug_speculative_rewrite:
                        self._log_dynamic_result_debug(
                            result,
                            iteration,
                            result_index,
                            token_count,
                            telemetry,
                            emitted_text,
                        )
                    if not chunk:
                        if result.get("eos", False):
                            self._log_rewrite_stop("eos", result)
                            return
                        continue

                    trimmed_chunk, stop_string = self._trim_at_rewrite_stop(chunk)
                    if trimmed_chunk:
                        telemetry.observe_chunk(trimmed_chunk, token_count)
                        emitted_tokens += token_count
                        emitted_text += trimmed_chunk
                        if self.debug_speculative_rewrite:
                            print(
                                "[HEAVY-PATH][spec] emit "
                                f"tokens={emitted_tokens} chars={len(emitted_text)} "
                                f"delta={trimmed_chunk!r}",
                                flush=True,
                            )
                        yield trimmed_chunk
                    if stop_string:
                        self._log_rewrite_stop(f"text stop {stop_string!r}", result)
                        return
                    if result.get("eos", False):
                        self._log_rewrite_stop("eos", result)
                        return
                    if emitted_tokens >= max_new_tokens:
                        self._log_rewrite_stop("max_new_tokens", result)
                        return
        finally:
            if cancel_event.is_set() and hasattr(job, "cancel"):
                self._log_rewrite_stop("cancelled")
                job.cancel()
            if self.debug_speculative_rewrite:
                print(
                    "[HEAVY-PATH][spec] final emitted "
                    f"tokens={emitted_tokens} chars={len(emitted_text)} "
                    f"text={emitted_text!r} job={self._job_debug_state(job)}",
                    flush=True,
                )

    def _dynamic_result_text_delta(self, result: dict[str, Any]) -> str:
        chunk = result.get("chunk")
        text = result.get("text")
        return str(text if text is not None else chunk or "")

    def _log_speculative_startup_debug(self) -> None:
        if not self.debug_speculative_rewrite:
            return
        print(
            "[HEAVY-PATH][spec] startup "
            f"enabled={self.enable_speculative_rewrite} "
            f"draft_tokens={self.speculative_draft_tokens} "
            f"target={self.llama_target.model_dir} "
            f"draft={self.llama_draft.model_dir if self.llama_draft else None}",
            flush=True,
        )
        self._log_model_fingerprint("target", self.llama_target)
        if self.llama_draft is not None:
            self._log_model_fingerprint("draft", self.llama_draft)
            self._log_tokenizer_file_comparison(self.llama_target, self.llama_draft)

    def _log_rewrite_request_debug(
        self,
        prompt: str,
        input_ids: list[int],
        max_new_tokens: int,
    ) -> None:
        print(
            "[HEAVY-PATH][spec] rewrite request "
            f"prompt_chars={len(prompt)} prompt_tokens={len(input_ids)} "
            f"max_new_tokens={max_new_tokens} "
            f"temperature={self.rewrite_temperature} top_p={self.rewrite_top_p} "
            f"top_k={self.rewrite_top_k} repetition_penalty={self.rewrite_repetition_penalty}",
            flush=True,
        )
        print(
            "[HEAVY-PATH][spec] prompt preview "
            f"{prompt[:500]!r}",
            flush=True,
        )

    def _log_model_fingerprint(self, label: str, bundle: ModelBundle) -> None:
        config = bundle.config
        tokenizer = bundle.tokenizer
        print(
            "[HEAVY-PATH][spec] model "
            f"{label} dir={bundle.model_dir} "
            f"vocab_size={getattr(config, 'vocab_size', None)} "
            f"bos={getattr(config, 'bos_token_id', None)} "
            f"eos={getattr(config, 'eos_token_id', None)} "
            f"tokenizer_eos={getattr(tokenizer, 'eos_token_id', None) if tokenizer else None} "
            f"tokenizer_hash={self._model_file_hash(bundle.model_dir, 'tokenizer.json')} "
            f"tokenizer_config_hash={self._model_file_hash(bundle.model_dir, 'tokenizer_config.json')}",
            flush=True,
        )

    def _log_tokenizer_file_comparison(
        self,
        target: ModelBundle,
        draft: ModelBundle,
    ) -> None:
        for filename in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            target_hash = self._model_file_hash(target.model_dir, filename)
            draft_hash = self._model_file_hash(draft.model_dir, filename)
            print(
                "[HEAVY-PATH][spec] file compare "
                f"{filename} target={target_hash} draft={draft_hash} "
                f"match={target_hash == draft_hash and target_hash != 'missing'}",
                flush=True,
            )

    def _model_file_hash(self, model_dir: str, filename: str) -> str:
        path = Path(model_dir) / filename
        if not path.exists():
            return "missing"
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()[:16]

    def _log_dynamic_result_debug(
        self,
        result: dict[str, Any],
        iteration: int,
        result_index: int,
        token_count: int,
        telemetry: TelemetryStats,
        emitted_text: str,
    ) -> None:
        keys = ",".join(sorted(str(key) for key in result.keys()))
        text = result.get("text")
        chunk = result.get("chunk")
        print(
            "[HEAVY-PATH][spec] result "
            f"iterate={iteration} index={result_index} keys={keys} "
            f"eos={result.get('eos', None)} token_count={token_count} "
            f"acceptance={telemetry.acceptance_summary()} "
            f"text={self._preview_debug_value(text)} "
            f"chunk={self._preview_debug_value(chunk)} "
            f"tokens={self._result_tokens_debug(result)} "
            f"draft={self._result_draft_debug(result)} "
            f"emitted_tail={emitted_text[-120:]!r}",
            flush=True,
        )

    def _result_tokens_debug(self, result: dict[str, Any]) -> str:
        parts = []
        for key in (
            "token",
            "token_id",
            "tokens",
            "token_ids",
            "ids",
            "new_token_ids",
            "generated_token_ids",
        ):
            if key in result:
                parts.append(f"{key}={self._preview_debug_value(result[key])}")
        return "; ".join(parts) or "unavailable"

    def _result_draft_debug(self, result: dict[str, Any]) -> str:
        parts = []
        for key in (
            "draft_tokens",
            "num_draft_tokens",
            "drafted_tokens",
            "attempted_draft_tokens",
            "accepted_tokens",
            "num_accepted_tokens",
            "accepted_draft_tokens",
            "draft_tokens_accepted",
            "rejected_tokens",
            "rejected_draft_tokens",
            "draft_token_ids",
            "draft_ids",
            "draft_text",
        ):
            if key in result:
                parts.append(f"{key}={self._preview_debug_value(result[key])}")
        if parts:
            return "; ".join(parts)
        return "raw draft proposal not exposed by result"

    def _preview_debug_value(self, value: Any, limit: int = 160) -> str:
        if value is None:
            return "None"
        if hasattr(value, "tolist"):
            try:
                value = value.tolist()
            except Exception:
                pass
        rendered = repr(value)
        if len(rendered) > limit:
            rendered = f"{rendered[:limit]}..."
        return rendered

    def _job_debug_state(self, job: Any) -> str:
        parts = []
        for attr_name in (
            "eos",
            "finished",
            "done",
            "completed",
            "max_new_tokens",
            "new_tokens",
            "generated_tokens",
            "accepted_draft_tokens",
            "rejected_draft_tokens",
        ):
            if hasattr(job, attr_name):
                value = getattr(job, attr_name)
                if callable(value):
                    try:
                        value = value()
                    except TypeError:
                        value = "<callable>"
                parts.append(f"{attr_name}={value!r}")
        return " ".join(parts) or "unavailable"

    def _job_is_complete(self, job: Any) -> bool:
        for attr_name in ("eos", "finished", "done", "completed", "is_complete"):
            value = getattr(job, attr_name, None)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            if isinstance(value, bool) and value:
                return True
        return False

    def _trim_at_rewrite_stop(self, chunk: str) -> tuple[str, Optional[str]]:
        stop_index: Optional[int] = None
        matched_stop: Optional[str] = None
        for stop_string in LLAMA_REWRITE_STOP_STRINGS:
            index = chunk.find(stop_string)
            if index != -1 and (stop_index is None or index < stop_index):
                stop_index = index
                matched_stop = stop_string

        if stop_index is None:
            return chunk, None
        return chunk[:stop_index], matched_stop

    def _log_rewrite_stop(
        self,
        reason: str,
        result: Optional[dict[str, Any]] = None,
    ) -> None:
        details = ""
        if result is not None:
            token_id = _first_present_value(
                result,
                ("token", "token_id", "id", "ids"),
            )
            if token_id is not None:
                details = f" token={token_id!r}"
            eos_reason = result.get("eos_reason")
            if eos_reason is not None:
                details += f" eos_reason={eos_reason!r}"
            full_completion = result.get("full_completion")
            if full_completion is not None:
                details += f" full_completion={self._preview_debug_value(full_completion, 240)}"
        print(f"[HEAVY-PATH] Stop reason: {reason}{details}", flush=True)

    def _rewrite_stop_conditions(self) -> list[Any]:
        stop_conditions: list[Any] = list(LLAMA_REWRITE_STOP_STRINGS)
        eos_token_id = getattr(self.llama_target.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple, set)):
                stop_conditions.extend(eos_token_id)
            else:
                stop_conditions.append(eos_token_id)
        stop_conditions.extend(LLAMA3_STOP_TOKEN_IDS)
        return stop_conditions

    def _result_token_count(self, result: dict[str, Any]) -> int:
        token_ids = _first_present_value(
            result,
            ("token_ids", "tokens_ids", "ids", "token"),
        )
        if hasattr(token_ids, "numel"):
            return int(token_ids.numel())
        if isinstance(token_ids, (list, tuple)):
            if (
                len(token_ids) == 1
                and isinstance(token_ids[0], (list, tuple))
            ):
                return max(1, len(token_ids[0]))
            return max(1, len(token_ids))
        if isinstance(token_ids, int):
            return 1

        explicit_count = _first_int_value(
            result,
            (
                "tokens",
                "num_tokens",
                "new_tokens",
                "generated_tokens",
                "token_count",
            ),
        )
        if explicit_count is not None and explicit_count > 0:
            return explicit_count

        return 1

    def generate_stream(
        self,
        cancel_event: Any,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        limit = max_new_tokens or self.default_max_new_tokens
        generated_token_ids: list[int] = []
        emitted_text = ""

        if self.qwen is None or self._qwen_last_token_id is None:
            return

        input_token: Optional[Any] = None

        for _ in range(limit):
            if cancel_event.is_set():
                break

            with self.gpu_lock, self.torch.inference_mode():
                if self._qwen_next_logits is not None:
                    logits = self._qwen_next_logits
                    self._qwen_next_logits = None
                else:
                    if input_token is None:
                        break
                    if self.qwen.cache.current_seq_len >= self.max_seq_len:
                        break
                    logits = self.qwen.model.forward(input_token, self.qwen.cache)
                next_token_id = self._sample_next_token(logits, generated_token_ids)
                self.current_seq_len = self.qwen.cache.current_seq_len
                self._qwen_last_token_id = next_token_id

            if self._is_eos(self.qwen.tokenizer, next_token_id):
                break
            if self._is_degenerate_repeat(generated_token_ids, next_token_id):
                break

            generated_token_ids.append(next_token_id)
            chunk = self._decode_complete_chunk(
                self.qwen.tokenizer,
                generated_token_ids,
                emitted_text,
            )
            if chunk:
                emitted_text += chunk
                yield chunk

            input_token = self.torch.tensor(
                [[next_token_id]],
                dtype=self.torch.long,
            )

    def _encode_prompt(self, tokenizer: Any, prompt: str) -> list[int]:
        encode_attempts = (
            {"add_bos": False, "add_eos": False, "encode_special_tokens": True},
            {"add_bos": False, "add_eos": False},
            {},
        )

        encoded: Any = None
        for kwargs in encode_attempts:
            try:
                encoded = tokenizer.encode(prompt, **kwargs)
                break
            except TypeError:
                continue

        if encoded is None:
            encoded = tokenizer.encode(prompt)

        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]

    def _autocomplete_stop_conditions(self) -> list[Any]:
        assert self.qwen is not None
        stop_conditions: list[Any] = list(QWEN_FIM_STOP_STRINGS)
        eos_token_id = getattr(self.qwen.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_conditions.append(eos_token_id)
        return stop_conditions

    def _llama_fast_stop_conditions(self) -> list[Any]:
        assert self.fast_llama is not None
        stop_conditions: list[Any] = list(LLAMA_FAST_STOP_STRINGS)
        eos_token_id = getattr(self.fast_llama.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            if isinstance(eos_token_id, (list, tuple, set)):
                stop_conditions.extend(eos_token_id)
            else:
                stop_conditions.append(eos_token_id)
        stop_conditions.extend(LLAMA3_STOP_TOKEN_IDS)
        return stop_conditions

    def _sample_next_token(self, logits: Any, generated_token_ids: list[int]) -> int:
        scores = logits[:, -1, :].float().squeeze(0)

        if self.qwen_repetition_penalty > 1.0 and generated_token_ids:
            for token_id in set(generated_token_ids[-128:]):
                if scores[token_id] < 0:
                    scores[token_id] *= self.qwen_repetition_penalty
                else:
                    scores[token_id] /= self.qwen_repetition_penalty

        if self.qwen_temperature <= 0:
            return int(self.torch.argmax(scores).item())

        scores = scores / self.qwen_temperature
        scores = self._apply_top_k(scores)
        scores = self._apply_top_p(scores)
        probabilities = self.torch.softmax(scores, dim=-1)

        if not self.torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
            return int(self.torch.argmax(scores).item())

        token = self.torch.multinomial(probabilities, num_samples=1)
        return int(token.item())

    def _apply_top_k(self, scores: Any) -> Any:
        if self.qwen_top_k <= 0 or self.qwen_top_k >= scores.shape[-1]:
            return scores

        values, indexes = self.torch.topk(scores, self.qwen_top_k)
        filtered = self.torch.full_like(scores, float("-inf"))
        return filtered.scatter(0, indexes, values)

    def _apply_top_p(self, scores: Any) -> Any:
        if self.qwen_top_p <= 0 or self.qwen_top_p >= 1:
            return scores

        sorted_scores, sorted_indexes = self.torch.sort(scores, descending=True)
        sorted_probabilities = self.torch.softmax(sorted_scores, dim=-1)
        cumulative_probabilities = self.torch.cumsum(sorted_probabilities, dim=-1)
        remove_mask = cumulative_probabilities > self.qwen_top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False
        sorted_scores = sorted_scores.masked_fill(remove_mask, float("-inf"))

        filtered = self.torch.full_like(scores, float("-inf"))
        return filtered.scatter(0, sorted_indexes, sorted_scores)

    def _is_eos(self, tokenizer: Any, token_id: int) -> bool:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            return False
        if isinstance(eos_token_id, (list, tuple, set)):
            return token_id in eos_token_id
        return token_id == eos_token_id

    def _is_degenerate_repeat(
        self,
        generated_token_ids: list[int],
        next_token_id: int,
    ) -> bool:
        if len(generated_token_ids) < 3:
            return False
        return all(token_id == next_token_id for token_id in generated_token_ids[-3:])

    def _coerce_decoded_text(self, decoded: Any) -> str:
        if isinstance(decoded, str):
            return decoded
        if isinstance(decoded, list):
            return "".join(str(part) for part in decoded)
        return str(decoded)

    def _decode_tokens(self, tokenizer: Any, token_ids: list[int]) -> str:
        for candidate in (token_ids, self.torch.tensor([token_ids])):
            try:
                return self._coerce_decoded_text(tokenizer.decode(candidate))
            except (AssertionError, TypeError):
                continue
        return self._coerce_decoded_text(
            tokenizer.decode(self.torch.tensor([token_ids]))
        )

    def _decode_complete_chunk(
        self,
        tokenizer: Any,
        token_ids: list[int],
        emitted_text: str,
    ) -> str:
        text = self._decode_tokens(tokenizer, token_ids)
        if not text:
            return ""
        if "\ufffd" in text:
            return ""
        if not emitted_text:
            return text

        shared_length = 0
        for previous_char, next_char in zip(emitted_text, text):
            if previous_char != next_char:
                break
            shared_length += 1

        if shared_length < len(emitted_text):
            return ""

        return text[shared_length:]


def get_engine() -> MockExLlamaEngine | RealExLlamaEngine:
    if _env_truthy("USE_MOCK_ENGINE"):
        return MockExLlamaEngine()

    enable_speculative_rewrite = _env_bool(
        "ENABLE_SPECULATIVE_REWRITE",
        DEFAULT_ENABLE_SPECULATIVE_REWRITE,
    )
    fast_path_mode = os.getenv("FAST_PATH_MODE", DEFAULT_FAST_PATH_MODE).strip().lower()
    qwen_model_dir = os.getenv("QWEN_AUTOCOMPLETE_MODEL_DIR") or os.getenv(
        "EXLLAMA_MODEL_DIR"
    )
    llama_fast_model_dir = os.getenv("LLAMA_FAST_MODEL_DIR")
    llama_target_model_dir = os.getenv("LLAMA_REWRITE_TARGET_MODEL_DIR")
    llama_draft_model_dir = os.getenv("LLAMA_REWRITE_DRAFT_MODEL_DIR")

    missing_env_vars = []
    if fast_path_mode == "qwen_fim" and not qwen_model_dir:
        missing_env_vars.append("QWEN_AUTOCOMPLETE_MODEL_DIR")
    if fast_path_mode != "qwen_fim" and not llama_fast_model_dir:
        missing_env_vars.append("LLAMA_FAST_MODEL_DIR")
    if not llama_target_model_dir:
        missing_env_vars.append("LLAMA_REWRITE_TARGET_MODEL_DIR")
    if enable_speculative_rewrite and not llama_draft_model_dir:
        missing_env_vars.append("LLAMA_REWRITE_DRAFT_MODEL_DIR")

    if missing_env_vars:
        raise RuntimeError(
            "Missing required model path environment variables when "
            f"USE_MOCK_ENGINE is not true: {', '.join(missing_env_vars)}"
        )

    return RealExLlamaEngine(
        qwen_model_dir=qwen_model_dir or "",
        llama_fast_model_dir=llama_fast_model_dir,
        llama_target_model_dir=llama_target_model_dir,
        llama_draft_model_dir=llama_draft_model_dir,
        max_seq_len=_env_int("MAX_SEQ_LEN", DEFAULT_MAX_SEQ_LEN),
    )
