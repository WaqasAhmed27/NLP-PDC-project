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
from pathlib import Path
from typing import Any, Iterator, Optional


TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
DEFAULT_MAX_SEQ_LEN = 8192
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_TEMPERATURE = 0.75
DEFAULT_TOP_P = 0.92
DEFAULT_TOP_K = 40
DEFAULT_REPETITION_PENALTY = 1.12


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


class MockExLlamaEngine:
    """CPU-only stand-in with API parity for local WebSocket development."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.gpu_lock = threading.Lock()
        self.current_seq_len = 0
        self.prefill_history: list[list[int]] = []
        self.truncation_history: list[int] = []

    def reset(self) -> None:
        with self.gpu_lock:
            self.current_seq_len = 0
            self.prefill_history.clear()
            self.truncation_history.clear()

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

    def forward_prefill(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("forward_prefill called with an empty token list")
        with self.gpu_lock:
            self.prefill_history.append(list(token_ids))
            self.current_seq_len += len(token_ids)

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


class RealExLlamaEngine:
    """ExLlamaV2-backed engine with serialized GPU access."""

    def __init__(
        self,
        model_dir: str,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.gpu_lock = threading.Lock()
        self.model_dir = str(Path(model_dir).expanduser())
        self.max_seq_len = max_seq_len
        self.default_max_new_tokens = max_new_tokens
        self.current_seq_len = 0
        self._last_token_id: Optional[int] = None
        self._next_logits: Optional[Any] = None
        self.temperature = max(0.0, _env_float("GENERATION_TEMPERATURE", DEFAULT_TEMPERATURE))
        self.top_p = min(1.0, max(0.0, _env_float("GENERATION_TOP_P", DEFAULT_TOP_P)))
        self.top_k = max(0, _env_int("GENERATION_TOP_K", DEFAULT_TOP_K))
        self.repetition_penalty = max(
            1.0,
            _env_float("GENERATION_REPETITION_PENALTY", DEFAULT_REPETITION_PENALTY),
        )

        if not Path(self.model_dir).exists():
            raise FileNotFoundError(
                f"EXLLAMA_MODEL_DIR does not exist: {self.model_dir}"
            )

        try:
            import torch
            from exllamav2 import (
                ExLlamaV2,
                ExLlamaV2Cache_8bit,
                ExLlamaV2Config,
                ExLlamaV2Tokenizer,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "RealExLlamaEngine requires exllamav2 and torch. Set "
                "USE_MOCK_ENGINE=true for local non-CUDA development."
            ) from exc

        self.torch = torch
        self.config = ExLlamaV2Config(self.model_dir)
        if hasattr(self.config, "prepare"):
            self.config.prepare()
        if hasattr(self.config, "arch_compat_overrides"):
            self.config.arch_compat_overrides()

        self.model = ExLlamaV2(self.config)
        self.cache = self._build_cache(ExLlamaV2Cache_8bit, max_seq_len)
        if hasattr(self.model, "load_autosplit"):
            self.model.load_autosplit(self.cache)
        else:
            self.model.load()
        self.tokenizer = self._build_tokenizer(ExLlamaV2Tokenizer)

    def _build_cache(self, cache_cls: Any, max_seq_len: int) -> Any:
        candidates = (
            {"batch_size": 1, "max_seq_len": max_seq_len, "lazy": True},
            {"max_seq_len": max_seq_len, "lazy": True},
            {"batch_size": 1, "max_seq_len": max_seq_len},
            {"max_seq_len": max_seq_len},
        )
        last_error: Optional[TypeError] = None

        for kwargs in candidates:
            try:
                return cache_cls(self.model, **kwargs)
            except TypeError as exc:
                last_error = exc

        raise RuntimeError("Could not initialize ExLlamaV2 cache") from last_error

    def _build_tokenizer(self, tokenizer_cls: Any) -> Any:
        try:
            return tokenizer_cls(self.config)
        except TypeError:
            return tokenizer_cls(self.model_dir)

    def reset(self) -> None:
        with self.gpu_lock:
            self.cache.current_seq_len = 0
            self.current_seq_len = 0
            self._last_token_id = None
            self._next_logits = None

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
            self.cache.current_seq_len = safe_token_index
            self.current_seq_len = safe_token_index
            if safe_token_index == 0:
                self._last_token_id = None
            self._next_logits = None

    def forward_prefill(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("forward_prefill called with an empty token list")

        with self.gpu_lock, self.torch.inference_mode():
            input_ids = self.torch.tensor([token_ids], dtype=self.torch.long)
            self._next_logits = self.model.forward(input_ids, self.cache)
            self.current_seq_len = self.cache.current_seq_len
            self._last_token_id = int(token_ids[-1])

    def generate_stream(
        self,
        cancel_event: Any,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        limit = max_new_tokens or self.default_max_new_tokens
        generated_token_ids: list[int] = []
        emitted_text = ""

        if self._last_token_id is None:
            return

        input_token: Optional[Any] = None

        for _ in range(limit):
            if cancel_event.is_set():
                break

            with self.gpu_lock, self.torch.inference_mode():
                if self._next_logits is not None:
                    logits = self._next_logits
                    self._next_logits = None
                else:
                    if input_token is None:
                        break
                    if self.cache.current_seq_len >= self.max_seq_len:
                        break
                    logits = self.model.forward(input_token, self.cache)
                next_token_id = self._sample_next_token(logits, generated_token_ids)
                self.current_seq_len = self.cache.current_seq_len
                self._last_token_id = next_token_id

            if self._is_eos(next_token_id):
                break
            if self._is_degenerate_repeat(generated_token_ids, next_token_id):
                break

            generated_token_ids.append(next_token_id)
            chunk = self._decode_complete_chunk(
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

    def _sample_next_token(self, logits: Any, generated_token_ids: list[int]) -> int:
        scores = logits[:, -1, :].float().squeeze(0)

        if self.repetition_penalty > 1.0 and generated_token_ids:
            for token_id in set(generated_token_ids[-128:]):
                if scores[token_id] < 0:
                    scores[token_id] *= self.repetition_penalty
                else:
                    scores[token_id] /= self.repetition_penalty

        if self.temperature <= 0:
            return int(self.torch.argmax(scores).item())

        scores = scores / self.temperature
        scores = self._apply_top_k(scores)
        scores = self._apply_top_p(scores)
        probabilities = self.torch.softmax(scores, dim=-1)

        if not self.torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
            return int(self.torch.argmax(scores).item())

        token = self.torch.multinomial(probabilities, num_samples=1)
        return int(token.item())

    def _apply_top_k(self, scores: Any) -> Any:
        if self.top_k <= 0 or self.top_k >= scores.shape[-1]:
            return scores

        values, indexes = self.torch.topk(scores, self.top_k)
        filtered = self.torch.full_like(scores, float("-inf"))
        return filtered.scatter(0, indexes, values)

    def _apply_top_p(self, scores: Any) -> Any:
        if self.top_p <= 0 or self.top_p >= 1:
            return scores

        sorted_scores, sorted_indexes = self.torch.sort(scores, descending=True)
        sorted_probabilities = self.torch.softmax(sorted_scores, dim=-1)
        cumulative_probabilities = self.torch.cumsum(sorted_probabilities, dim=-1)
        remove_mask = cumulative_probabilities > self.top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False
        sorted_scores = sorted_scores.masked_fill(remove_mask, float("-inf"))

        filtered = self.torch.full_like(scores, float("-inf"))
        return filtered.scatter(0, sorted_indexes, sorted_scores)

    def _is_eos(self, token_id: int) -> bool:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
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

    def _decode_tokens(self, token_ids: list[int]) -> str:
        for candidate in (token_ids, self.torch.tensor([token_ids])):
            try:
                return self._coerce_decoded_text(self.tokenizer.decode(candidate))
            except (AssertionError, TypeError):
                continue
        return self._coerce_decoded_text(
            self.tokenizer.decode(self.torch.tensor([token_ids]))
        )

    def _decode_complete_chunk(
        self,
        token_ids: list[int],
        emitted_text: str,
    ) -> str:
        text = self._decode_tokens(token_ids)
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

    model_dir = os.getenv("EXLLAMA_MODEL_DIR")
    if not model_dir:
        raise RuntimeError(
            "EXLLAMA_MODEL_DIR must be set when USE_MOCK_ENGINE is not true."
        )
    return RealExLlamaEngine(model_dir=model_dir, max_seq_len=DEFAULT_MAX_SEQ_LEN)
