"""
MockExLlamaV2Engine — CPU-only simulation of an ExLlamaV2 inference engine.

This module provides a lightweight stand-in for the real ExLlamaV2 engine so that
the EditorStateManager can be developed and tested on any machine without a GPU.
The mock faithfully tracks the KV-cache sequence length and validates all
truncation / prefill operations against the same invariants the real engine
enforces.

Design notes
------------
* ``current_seq_len`` mirrors ``cache.current_seq_len`` in the real engine.
* ``truncate(index)`` is O(1) — it simply resets the length counter.
* ``forward_prefill(new_tokens)`` increments the counter by the number of
  tokens passed in, simulating the real engine writing those tokens into
  the KV-cache.
"""

from __future__ import annotations

from typing import List


class MockExLlamaV2Engine:
    """Simulates an ExLlamaV2 inference engine operating in batch-size-1 mode.

    Attributes
    ----------
    current_seq_len : int
        The number of tokens currently held in the (simulated) KV cache.
    prefill_history : list[list[int]]
        Debug log — every ``forward_prefill`` call appends its token list here
        so that tests can assert on exactly which tokens were sent to the
        engine.
    truncation_history : list[int]
        Debug log — every ``truncate`` call appends its target index here.
    """

    __slots__ = ("current_seq_len", "prefill_history", "truncation_history")

    def __init__(self) -> None:
        self.current_seq_len: int = 0
        self.prefill_history: List[List[int]] = []
        self.truncation_history: List[int] = []

    # ------------------------------------------------------------------ #
    #  Public API (mirrors real ExLlamaV2)                                 #
    # ------------------------------------------------------------------ #

    def truncate(self, index: int) -> None:
        """Discard all KV-cache entries at positions >= *index*.

        Parameters
        ----------
        index : int
            The token position at which to truncate.  After this call,
            ``current_seq_len == index``.

        Raises
        ------
        ValueError
            If *index* is negative or exceeds the current sequence length.
        """
        if index < 0:
            raise ValueError(
                f"truncate index must be >= 0, got {index}"
            )
        if index > self.current_seq_len:
            raise ValueError(
                f"Cannot truncate to {index}; current_seq_len is only "
                f"{self.current_seq_len}"
            )
        self.truncation_history.append(index)
        self.current_seq_len = index

    def forward_prefill(self, new_tokens: List[int]) -> None:
        """Write *new_tokens* into the KV cache and advance ``current_seq_len``.

        In the real engine this triggers a single CUDA kernel launch that
        computes attention for the new tokens against the existing cache.

        Parameters
        ----------
        new_tokens : list[int]
            Token IDs to prefill.

        Raises
        ------
        ValueError
            If *new_tokens* is empty.
        """
        if not new_tokens:
            raise ValueError("forward_prefill called with an empty token list")
        self.prefill_history.append(list(new_tokens))
        self.current_seq_len += len(new_tokens)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Hard-reset the engine to its initial state (empty cache)."""
        self.current_seq_len = 0
        self.prefill_history.clear()
        self.truncation_history.clear()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MockExLlamaV2Engine(current_seq_len={self.current_seq_len})"
        )
