"""
EditorStateManager — Rewind & Re-tokenize algorithm for real-time AI editing.

This module implements the core state-management logic that allows an
ExLlamaV2-backed text editor to handle arbitrary character-level edits with
sub-40 ms Time-To-First-Token latency.

Algorithm overview
------------------
1. **Map Index** — Given a character position where the edit occurred, binary-
   search the stored ``token_offsets`` to find the enclosing token index *N*.
2. **Safe Rewind** — Compute ``safe_token_index = max(0, N - SAFETY_MARGIN)``
   to guard against BPE boundary merges that can ripple backwards.
3. **Cache Truncation** — Tell the engine to discard everything from
   ``safe_token_index`` onward (O(1) in ExLlamaV2's linear KV cache).
4. **Extract Suffix** — Use the character offset of ``safe_token_index`` to
   slice the *new* text from that point to the end of the document.
5. **Re-tokenize & Prefill** — Tokenize only the suffix, feed the resulting
   token IDs to the engine, and update internal bookkeeping.

Performance notes
-----------------
* The tokenizer object is instantiated **once** at construction time.
* ``apply_edit`` performs exactly **one** tokenizer call (on the suffix only).
* Token-offset lookup uses ``bisect`` for O(log N) search.
* No heavy allocations inside the hot path.
"""

from __future__ import annotations

import bisect
from typing import List, Optional, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from mock_engine import MockExLlamaV2Engine


# -------------------------------------------------------------------- #
#  Constants                                                            #
# -------------------------------------------------------------------- #

# Number of tokens to rewind *before* the edit token to guarantee we never
# break a BPE merge boundary.  4 is conservative; most BPE merge chains
# span at most 2-3 tokens.
SAFETY_MARGIN: int = 4

# Default model whose tokenizer we load.
DEFAULT_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"


# -------------------------------------------------------------------- #
#  Offset type alias                                                    #
# -------------------------------------------------------------------- #

# Each entry is (char_start, char_end) — half-open interval [start, end).
TokenOffset = Tuple[int, int]


class EditorStateManager:
    """Tracks document text & token-offset state for an ExLlamaV2 cache.

    Parameters
    ----------
    engine : MockExLlamaV2Engine
        The (mock or real) inference engine whose KV cache we manage.
    model_id : str, optional
        HuggingFace model identifier used to load the fast tokenizer.
    safety_margin : int, optional
        How many tokens before the edit token to rewind.  Defaults to
        ``SAFETY_MARGIN`` (4).

    Attributes
    ----------
    current_text : str
        The full document text currently reflected in the KV cache.
    token_ids : list[int]
        Token IDs for ``current_text``.
    token_offsets : list[TokenOffset]
        Character-level ``(start, end)`` spans for each token.
    """

    __slots__ = (
        "engine",
        "tokenizer",
        "safety_margin",
        "current_text",
        "token_ids",
        "token_offsets",
        "_offset_starts",  # cached list of start positions for bisect
    )

    def __init__(
        self,
        engine: MockExLlamaV2Engine,
        model_id: str = DEFAULT_MODEL_ID,
        safety_margin: int = SAFETY_MARGIN,
    ) -> None:
        self.engine: MockExLlamaV2Engine = engine
        self.safety_margin: int = safety_margin

        # Load the fast tokenizer once.
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_id, use_fast=True
        )

        # Internal state — starts empty.
        self.current_text: str = ""
        self.token_ids: List[int] = []
        self.token_offsets: List[TokenOffset] = []
        self._offset_starts: List[int] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def initialize(self, text: str) -> None:
        """Tokenize *text* from scratch and fully prefill the engine cache.

        Use this for the very first document load.  Subsequent edits should
        go through ``apply_edit``.

        Parameters
        ----------
        text : str
            The initial document content.
        """
        self.engine.reset()
        ids, offsets = self._tokenize(text)
        self.current_text = text
        self.token_ids = ids
        self.token_offsets = offsets
        self._offset_starts = [o[0] for o in offsets]

        if ids:
            self.engine.forward_prefill(ids)

    def apply_edit(
        self,
        new_text: str,
        edit_char_index: int,
    ) -> "EditResult":
        """Process a character-level edit and surgically update the KV cache.

        Parameters
        ----------
        new_text : str
            The full document text *after* the edit has been applied.
        edit_char_index : int
            The character index in the *old* text where the edit began.
            For appends this equals ``len(old_text)``.

        Returns
        -------
        EditResult
            Diagnostic dataclass containing rewind / prefill details.

        Raises
        ------
        ValueError
            If ``edit_char_index`` is negative or exceeds the length of the
            current text.
        """
        if edit_char_index < 0:
            raise ValueError(
                f"edit_char_index must be >= 0, got {edit_char_index}"
            )
        if edit_char_index > len(self.current_text):
            raise ValueError(
                f"edit_char_index out of bounds: {edit_char_index} > "
                f"len(current_text) = {len(self.current_text)}"
            )

        # Step 1 — Map the character index to a token index.
        token_index: int = self._char_to_token_index(edit_char_index)

        # Step 2 — Compute the safe rewind point.
        safe_token_index: int = self._calculate_safe_rewind(token_index)

        # Step 3 — Truncate the engine's KV cache.
        self.engine.truncate(safe_token_index)

        # Step 4 — Extract the suffix from the *new* text.
        if safe_token_index < len(self.token_offsets):
            suffix_char_start: int = self.token_offsets[safe_token_index][0]
        else:
            # Edit is past all existing tokens (pure append).
            suffix_char_start = len(self.current_text)

        suffix_text: str = new_text[suffix_char_start:]

        # Step 5 — Re-tokenize the suffix.
        suffix_ids: List[int]
        suffix_offsets: List[TokenOffset]
        suffix_ids, suffix_offsets = self._tokenize(suffix_text)

        # Shift the suffix offsets so they are relative to the full document.
        shifted_offsets: List[TokenOffset] = [
            (s + suffix_char_start, e + suffix_char_start)
            for s, e in suffix_offsets
        ]

        # Prefill the engine with the new tokens.
        if suffix_ids:
            self.engine.forward_prefill(suffix_ids)

        # Update internal state: keep everything *before* the rewind point
        # and replace the rest with the newly tokenized suffix.
        self.token_ids = self.token_ids[:safe_token_index] + suffix_ids
        self.token_offsets = self.token_offsets[:safe_token_index] + shifted_offsets
        self._offset_starts = [o[0] for o in self.token_offsets]
        self.current_text = new_text

        return EditResult(
            token_index=token_index,
            safe_token_index=safe_token_index,
            suffix_char_start=suffix_char_start,
            suffix_text=suffix_text,
            suffix_token_count=len(suffix_ids),
            total_cache_len=self.engine.current_seq_len,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_safe_rewind(self, token_index: int) -> int:
        """Return the safe rewind token position given the edit's token index.

        The safety margin protects against BPE merge boundaries shifting
        when characters are inserted or deleted.

        Parameters
        ----------
        token_index : int
            The token index that contains (or is closest to) the edit.

        Returns
        -------
        int
            ``max(0, token_index - self.safety_margin)``
        """
        return max(0, token_index - self.safety_margin)

    def _char_to_token_index(self, char_index: int) -> int:
        """Map a character index to the enclosing token index.

        Uses ``bisect_right`` on the cached list of token start positions
        for O(log N) lookup.

        Parameters
        ----------
        char_index : int
            A character offset into ``current_text``.

        Returns
        -------
        int
            The index into ``token_ids`` / ``token_offsets`` whose span
            contains *char_index*.  If *char_index* is past the end of
            all tokens (e.g. pure append), returns ``len(token_ids)``.
        """
        if not self._offset_starts:
            return 0

        # bisect_right gives us the first token whose start is > char_index.
        # The enclosing token is one position before that.
        pos: int = bisect.bisect_right(self._offset_starts, char_index) - 1
        if pos < 0:
            return 0
        return pos

    def _tokenize(self, text: str) -> Tuple[List[int], List[TokenOffset]]:
        """Tokenize *text* and return (token_ids, offsets).

        Special tokens are **not** added — we deal with raw document text
        only.

        Parameters
        ----------
        text : str
            The string to tokenize.

        Returns
        -------
        tuple[list[int], list[TokenOffset]]
            Token IDs and their character spans.
        """
        if not text:
            return [], []

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        ids: List[int] = encoding["input_ids"]
        raw_offsets = encoding["offset_mapping"]

        # Filter out any (0, 0) placeholder offsets that some tokenizers emit
        # for special/padding tokens.
        filtered_ids = [tid for tid, o in zip(ids, raw_offsets) if o != (0, 0)]
        offsets = [o for o in raw_offsets if o != (0, 0)]

        return filtered_ids, offsets

    def _verify_offset_partition(self, text: str) -> None:
        """Verify that token_offsets form a perfect partition of the text.

        Raises AssertionError if:
        * Offsets have gaps or overlaps
        * Total span != len(text)
        * Offsets are not in order
        """
        total_span = sum(end - start for start, end in self.token_offsets)
        assert total_span == len(text), (
            f"Offset partition does not cover entire text: {total_span} != {len(text)}"
        )

        for i in range(len(self.token_offsets) - 1):
            end_i = self.token_offsets[i][1]
            start_next = self.token_offsets[i + 1][0]
            assert end_i == start_next, (
                f"Gap/overlap between token {i} and {i+1}: "
                f"token[{i}] ends at {end_i}, token[{i+1}] starts at {start_next}"
            )

    def _verify_no_offset_gaps(self) -> bool:
        """Check for offset gaps and return True if valid."""
        for i in range(len(self.token_offsets) - 1):
            if self.token_offsets[i][1] != self.token_offsets[i + 1][0]:
                return False
        return True

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EditorStateManager("
            f"text_len={len(self.current_text)}, "
            f"tokens={len(self.token_ids)}, "
            f"cache_len={self.engine.current_seq_len})"
        )


# -------------------------------------------------------------------- #
#  Result dataclass                                                     #
# -------------------------------------------------------------------- #

class EditResult:
    """Diagnostic container returned by ``apply_edit``.

    Attributes
    ----------
    token_index : int
        Token index that contains the edit position.
    safe_token_index : int
        Token index we actually rewound to (with safety margin).
    suffix_char_start : int
        Character offset where the re-tokenized suffix begins.
    suffix_text : str
        The substring that was re-tokenized.
    suffix_token_count : int
        Number of tokens produced by re-tokenizing the suffix.
    total_cache_len : int
        Engine ``current_seq_len`` after the operation.
    """

    __slots__ = (
        "token_index",
        "safe_token_index",
        "suffix_char_start",
        "suffix_text",
        "suffix_token_count",
        "total_cache_len",
    )

    def __init__(
        self,
        token_index: int,
        safe_token_index: int,
        suffix_char_start: int,
        suffix_text: str,
        suffix_token_count: int,
        total_cache_len: int,
    ) -> None:
        self.token_index = token_index
        self.safe_token_index = safe_token_index
        self.suffix_char_start = suffix_char_start
        self.suffix_text = suffix_text
        self.suffix_token_count = suffix_token_count
        self.total_cache_len = total_cache_len

    def __repr__(self) -> str:
        return (
            f"EditResult("
            f"token_idx={self.token_index}, "
            f"safe_idx={self.safe_token_index}, "
            f"suffix_start={self.suffix_char_start}, "
            f"suffix_tokens={self.suffix_token_count}, "
            f"cache_len={self.total_cache_len})"
        )
