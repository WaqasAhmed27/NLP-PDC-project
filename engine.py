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


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


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
        try:
            self.cache = ExLlamaV2Cache_8bit(
                self.model,
                batch_size=1,
                max_seq_len=max_seq_len,
            )
        except TypeError:
            self.cache = ExLlamaV2Cache_8bit(
                self.model,
                max_seq_len=max_seq_len,
            )
        self.model.load()
        self.tokenizer = self._build_tokenizer(ExLlamaV2Tokenizer)

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

    def forward_prefill(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("forward_prefill called with an empty token list")

        with self.gpu_lock, self.torch.inference_mode():
            input_ids = self.torch.tensor([token_ids], dtype=self.torch.long)
            self.model.forward(input_ids, self.cache)
            self.current_seq_len = self.cache.current_seq_len
            self._last_token_id = int(token_ids[-1])

    def generate_stream(
        self,
        cancel_event: Any,
        max_new_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        limit = max_new_tokens or self.default_max_new_tokens
        buffered_token_ids: list[int] = []
        emitted_text = ""

        if self._last_token_id is None:
            return

        input_token = self.torch.tensor(
            [[self._last_token_id]],
            dtype=self.torch.long,
        )

        for _ in range(limit):
            if cancel_event.is_set():
                break

            with self.gpu_lock, self.torch.inference_mode():
                if self.cache.current_seq_len >= self.max_seq_len:
                    break
                logits = self.model.forward(input_token, self.cache)
                next_token_id = self._sample_greedy(logits)
                self.current_seq_len = self.cache.current_seq_len
                self._last_token_id = next_token_id

            if self._is_eos(next_token_id):
                break

            buffered_token_ids.append(next_token_id)
            chunk = self._decode_complete_chunk(
                buffered_token_ids,
                emitted_text,
            )
            if chunk:
                emitted_text += chunk
                buffered_token_ids.clear()
                yield chunk

            input_token = self.torch.tensor(
                [[next_token_id]],
                dtype=self.torch.long,
            )

        if not cancel_event.is_set() and buffered_token_ids:
            chunk = self._decode_complete_chunk(
                buffered_token_ids,
                emitted_text,
            )
            if chunk:
                yield chunk

    def _sample_greedy(self, logits: Any) -> int:
        token = self.torch.argmax(logits[:, -1, :], dim=-1)
        return int(token.item())

    def _is_eos(self, token_id: int) -> bool:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            return False
        if isinstance(eos_token_id, (list, tuple, set)):
            return token_id in eos_token_id
        return token_id == eos_token_id

    def _decode_tokens(self, token_ids: list[int]) -> str:
        for candidate in (token_ids, self.torch.tensor([token_ids])):
            try:
                return self.tokenizer.decode(candidate)
            except TypeError:
                continue
        return self.tokenizer.decode(token_ids)

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
        if emitted_text and text.startswith(emitted_text):
            return text[len(emitted_text):]
        return text


def get_engine() -> MockExLlamaEngine | RealExLlamaEngine:
    if _env_truthy("USE_MOCK_ENGINE"):
        return MockExLlamaEngine()

    model_dir = os.getenv("EXLLAMA_MODEL_DIR")
    if not model_dir:
        raise RuntimeError(
            "EXLLAMA_MODEL_DIR must be set when USE_MOCK_ENGINE is not true."
        )
    return RealExLlamaEngine(model_dir=model_dir, max_seq_len=DEFAULT_MAX_SEQ_LEN)
