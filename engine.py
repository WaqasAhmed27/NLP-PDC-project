"""
Hardware abstraction layer for editor inference engines.

Set USE_MOCK_ENGINE=true for local development without CUDA/ExLlamaV2. Real
mode imports ExLlamaV2 lazily so merely importing this module is safe on
non-GPU machines.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional


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
DEFAULT_SPECULATIVE_DRAFT_TOKENS = 4
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
    "<|end_header_id|>",
    "<|reserved_special_token",
    "\nassistant",
    "\nAssistant",
    ".assistant",
    "assistant\n",
    "Here is the rewritten:",
    "Here is the text:",
    "Here is text",
)
LLAMA3_STOP_TOKEN_IDS = (
    128001,  # <|end_of_text|>
    128009,  # <|eot_id|>
)


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


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


class MockExLlamaEngine:
    """CPU-only stand-in with API parity for local WebSocket development."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.gpu_lock = threading.Lock()
        self.current_seq_len = 0
        self.prefill_history: list[list[int]] = []
        self.truncation_history: list[int] = []
        self.requires_full_prefill = False

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
        yield from self.generate_stream(cancel_event, max_new_tokens)

    def apply_rewrite(
        self,
        text: str,
        instruction: str,
        cancel_event: Any,
        max_new_tokens: int = 512,
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
        llama_target_model_dir: str,
        llama_draft_model_dir: str,
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
        self.qwen = self._load_model_bundle(
            "QWEN_AUTOCOMPLETE_MODEL_DIR",
            qwen_model_dir,
            load_tokenizer=True,
            cache_cls=self.qwen_cache_cls,
            cache_name="8-bit Qwen autocomplete",
        )
        self.llama_draft = self._load_model_bundle(
            "LLAMA_REWRITE_DRAFT_MODEL_DIR",
            llama_draft_model_dir,
            load_tokenizer=False,
            cache_cls=self.rewrite_cache_cls,
            cache_name="Q4 rewrite draft",
        )

        self.qwen_generator = self.streaming_generator_cls(
            self.qwen.model,
            self.qwen.cache,
            self.qwen.tokenizer,
        )
        self.qwen_sampling_settings = self._build_sampling_settings(
            self.qwen_temperature,
            self.qwen_top_p,
            self.qwen_top_k,
            self.qwen_repetition_penalty,
        )

        self.rewrite_generator = self.dynamic_generator_cls(
            model=self.llama_target.model,
            cache=self.llama_target.cache,
            tokenizer=self.llama_target.tokenizer,
            max_seq_len=max_seq_len,
            draft_model=self.llama_draft.model,
            draft_cache=self.llama_draft.cache,
            num_draft_tokens=self.speculative_draft_tokens,
        )
        self.rewrite_sampling_settings = self._build_sampling_settings(
            self.rewrite_temperature,
            self.rewrite_top_p,
            self.rewrite_top_k,
            self.rewrite_repetition_penalty,
        )

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
            self.qwen.cache.current_seq_len = 0
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
            self.qwen.cache.current_seq_len = safe_token_index
            self.current_seq_len = safe_token_index
            if safe_token_index == 0:
                self._qwen_last_token_id = None
            self._qwen_next_logits = None
            self.requires_full_prefill = False

    def forward_prefill(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("forward_prefill called with an empty token list")

        with self.gpu_lock, self.torch.inference_mode():
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
        telemetry = TelemetryStats("FAST-PATH", time.perf_counter())
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

    def apply_rewrite(
        self,
        text: str,
        instruction: str,
        cancel_event: Any,
        max_new_tokens: int = 512,
    ) -> Iterator[str]:
        telemetry = TelemetryStats("HEAVY-PATH", time.perf_counter())
        prompt = build_llama_rewrite_prompt(text, instruction)
        input_ids = self._encode_prompt(self.llama_target.tokenizer, prompt)
        if not input_ids:
            telemetry.log()
            return
        input_tensor = self.torch.tensor([input_ids], dtype=self.torch.long)

        job = self._build_rewrite_job(input_tensor, max_new_tokens)

        try:
            yield from self._consume_dynamic_job(job, cancel_event, telemetry)
        finally:
            telemetry.log()

    def _build_rewrite_job(self, input_tensor: Any, max_new_tokens: int) -> Any:
        job_candidates = (
            {
                "input_ids": input_tensor,
                "banned_strings": list(LLAMA_REWRITE_STOP_STRINGS),
                "gen_settings": self.rewrite_sampling_settings,
                "max_new_tokens": max_new_tokens,
            },
            {
                "input_ids": input_tensor,
                "stop_conditions": self._rewrite_stop_conditions(),
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
    ) -> Iterator[str]:
        with self.gpu_lock, self.torch.inference_mode():
            self.rewrite_generator.enqueue(job)

        try:
            while not cancel_event.is_set():
                with self.gpu_lock, self.torch.inference_mode():
                    results = self.rewrite_generator.iterate()

                if not results:
                    continue

                for result in results:
                    result_job = result.get("job")
                    if result_job is not None and result_job is not job:
                        continue

                    telemetry.observe_speculative_result(result)
                    chunk = str(result.get("text") or result.get("chunk") or "")
                    if not chunk:
                        if result.get("eos", False):
                            self._log_rewrite_stop("eos", result)
                            return
                        continue

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
        finally:
            if cancel_event.is_set() and hasattr(job, "cancel"):
                self._log_rewrite_stop("cancelled")
                job.cancel()

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
            token_id = (
                result.get("token")
                or result.get("token_id")
                or result.get("id")
                or result.get("ids")
            )
            if token_id is not None:
                details = f" token={token_id!r}"
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

        token_ids = (
            result.get("token_ids")
            or result.get("tokens_ids")
            or result.get("ids")
            or result.get("token")
        )
        if hasattr(token_ids, "numel"):
            return int(token_ids.numel())
        if isinstance(token_ids, (list, tuple)):
            return max(1, len(token_ids))
        if isinstance(token_ids, int):
            return 1

        return 1

    def generate_stream(
        self,
        cancel_event: Any,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        limit = max_new_tokens or self.default_max_new_tokens
        generated_token_ids: list[int] = []
        emitted_text = ""

        if self._qwen_last_token_id is None:
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
        stop_conditions: list[Any] = list(QWEN_FIM_STOP_STRINGS)
        eos_token_id = getattr(self.qwen.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            stop_conditions.append(eos_token_id)
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

    qwen_model_dir = os.getenv("QWEN_AUTOCOMPLETE_MODEL_DIR") or os.getenv(
        "EXLLAMA_MODEL_DIR"
    )
    llama_target_model_dir = os.getenv("LLAMA_REWRITE_TARGET_MODEL_DIR")
    llama_draft_model_dir = os.getenv("LLAMA_REWRITE_DRAFT_MODEL_DIR")

    missing_env_vars = []
    if not qwen_model_dir:
        missing_env_vars.append("QWEN_AUTOCOMPLETE_MODEL_DIR")
    if not llama_target_model_dir:
        missing_env_vars.append("LLAMA_REWRITE_TARGET_MODEL_DIR")
    if not llama_draft_model_dir:
        missing_env_vars.append("LLAMA_REWRITE_DRAFT_MODEL_DIR")

    if missing_env_vars:
        raise RuntimeError(
            "Missing required model path environment variables when "
            f"USE_MOCK_ENGINE is not true: {', '.join(missing_env_vars)}"
        )

    return RealExLlamaEngine(
        qwen_model_dir=qwen_model_dir,
        llama_target_model_dir=llama_target_model_dir,
        llama_draft_model_dir=llama_draft_model_dir,
        max_seq_len=_env_int("MAX_SEQ_LEN", DEFAULT_MAX_SEQ_LEN),
    )
