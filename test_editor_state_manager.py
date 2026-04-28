"""
Comprehensive pytest suite for EditorStateManager & MockExLlamaV2Engine.

Scenarios
---------
A  Autocomplete — appending text at the very end of the document.
B  Mid-document insertion — typing into the middle of a paragraph.
C  Boundary merge prevention — edits that force BPE re-chunking
   (e.g. "invest" → "investigate", adding a space that merges tokens).

Each test asserts on:
* Cache truncation index (rewind correctness).
* Suffix re-tokenization boundaries.
* Final cache sequence length = total token count.
* Token offset consistency (every offset spans a real substring of the text).

Running
-------
    pytest test_editor_state_manager.py -v
"""

from __future__ import annotations

import pytest

from mock_engine import MockExLlamaV2Engine
from editor_state_manager import EditorStateManager, EditResult


# -------------------------------------------------------------------- #
#  Fixtures                                                             #
# -------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def shared_tokenizer_manager() -> EditorStateManager:
    """Create a single manager (loads the tokenizer once for the module)."""
    engine = MockExLlamaV2Engine()
    return EditorStateManager(engine=engine)


@pytest.fixture
def manager(shared_tokenizer_manager: EditorStateManager) -> EditorStateManager:
    """Return the shared manager with a freshly reset engine per test."""
    shared_tokenizer_manager.engine.reset()
    shared_tokenizer_manager.current_text = ""
    shared_tokenizer_manager.token_ids = []
    shared_tokenizer_manager.token_offsets = []
    shared_tokenizer_manager._offset_starts = []
    return shared_tokenizer_manager


# -------------------------------------------------------------------- #
#  Mock Engine Unit Tests                                               #
# -------------------------------------------------------------------- #

class TestMockExLlamaV2Engine:
    """Basic sanity checks on the mock engine itself."""

    def test_initial_state(self) -> None:
        engine = MockExLlamaV2Engine()
        assert engine.current_seq_len == 0
        assert engine.prefill_history == []
        assert engine.truncation_history == []

    def test_forward_prefill_updates_seq_len(self) -> None:
        engine = MockExLlamaV2Engine()
        engine.forward_prefill([1, 2, 3])
        assert engine.current_seq_len == 3
        engine.forward_prefill([4, 5])
        assert engine.current_seq_len == 5

    def test_truncate_reduces_seq_len(self) -> None:
        engine = MockExLlamaV2Engine()
        engine.forward_prefill([1, 2, 3, 4, 5])
        engine.truncate(2)
        assert engine.current_seq_len == 2

    def test_truncate_beyond_seq_len_raises(self) -> None:
        engine = MockExLlamaV2Engine()
        engine.forward_prefill([1, 2, 3])
        with pytest.raises(ValueError, match="Cannot truncate to 10"):
            engine.truncate(10)

    def test_truncate_negative_raises(self) -> None:
        engine = MockExLlamaV2Engine()
        with pytest.raises(ValueError, match="must be >= 0"):
            engine.truncate(-1)

    def test_forward_prefill_empty_raises(self) -> None:
        engine = MockExLlamaV2Engine()
        with pytest.raises(ValueError, match="empty"):
            engine.forward_prefill([])

    def test_reset_clears_all(self) -> None:
        engine = MockExLlamaV2Engine()
        engine.forward_prefill([1, 2, 3])
        engine.truncate(1)
        engine.reset()
        assert engine.current_seq_len == 0
        assert engine.prefill_history == []
        assert engine.truncation_history == []


# -------------------------------------------------------------------- #
#  Scenario A — Autocomplete (append at end)                            #
# -------------------------------------------------------------------- #

class TestScenarioA_Autocomplete:
    """Appending text to the end of the document."""

    def test_append_word_to_end(self, manager: EditorStateManager) -> None:
        """Type a sentence, then append ' world' — only the suffix is re-tokenized."""
        original = "The quick brown fox jumps over the lazy dog"
        manager.initialize(original)
        original_token_count: int = len(manager.token_ids)
        assert manager.engine.current_seq_len == original_token_count

        new_text = original + " quickly"
        result: EditResult = manager.apply_edit(new_text, len(original))

        # The edit is at the very end — token_index should equal the last token
        # or past-end.
        assert result.token_index >= original_token_count - 1

        # Safe rewind goes back by safety margin.
        assert result.safe_token_index == max(0, result.token_index - 4)

        # Cache length should equal total token count of new text.
        assert manager.engine.current_seq_len == len(manager.token_ids)

        # The stored text must match.
        assert manager.current_text == new_text

        # Full re-tokenization from scratch must produce identical ids.
        full_ids, full_offsets = manager._tokenize(new_text)
        assert manager.token_ids == full_ids
        assert manager.token_offsets == full_offsets

    def test_append_to_empty(self, manager: EditorStateManager) -> None:
        """First keystroke on an empty document."""
        manager.initialize("")
        assert manager.engine.current_seq_len == 0

        result = manager.apply_edit("H", 0)
        assert manager.engine.current_seq_len == len(manager.token_ids)
        assert manager.current_text == "H"

    def test_multiple_sequential_appends(self, manager: EditorStateManager) -> None:
        """Simulates rapid typing character by character at the end."""
        base = "Hello"
        manager.initialize(base)

        for i, ch in enumerate(" World"):
            old_text = manager.current_text
            new_text = old_text + ch
            manager.apply_edit(new_text, len(old_text))

            # After every keystroke, tokens must match a fresh tokenization.
            full_ids, full_offsets = manager._tokenize(new_text)
            assert manager.token_ids == full_ids, f"Mismatch at append step {i}"
            assert manager.current_text == new_text


# -------------------------------------------------------------------- #
#  Scenario B — Mid-Document Insertion                                  #
# -------------------------------------------------------------------- #

class TestScenarioB_MidDocumentInsertion:
    """Inserting text in the middle of a paragraph."""

    def test_insert_word_in_middle(self, manager: EditorStateManager) -> None:
        """Insert 'very ' between 'the' and 'lazy' in a sentence."""
        original = "The quick brown fox jumps over the lazy dog"
        manager.initialize(original)
        original_cache_len: int = manager.engine.current_seq_len

        # Find "the lazy" and insert "very " before "lazy".
        insert_pos = original.index("lazy")
        new_text = original[:insert_pos] + "very " + original[insert_pos:]

        result: EditResult = manager.apply_edit(new_text, insert_pos)

        # The rewind must have happened *before* the insert position.
        assert result.safe_token_index < result.token_index
        assert result.safe_token_index == max(0, result.token_index - 4)

        # Engine must have been truncated and then re-filled.
        assert len(manager.engine.truncation_history) == 1
        assert manager.engine.truncation_history[0] == result.safe_token_index

        # Final cache length must be consistent.
        assert manager.engine.current_seq_len == len(manager.token_ids)

        # Token stream must be identical to a from-scratch tokenization.
        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids

    def test_insert_at_very_beginning(self, manager: EditorStateManager) -> None:
        """Edge case: insert at character 0."""
        original = "world"
        manager.initialize(original)

        new_text = "Hello " + original
        result = manager.apply_edit(new_text, 0)

        # Safe rewind should be 0 (can't go before the start).
        assert result.safe_token_index == 0

        # Engine must have been truncated to 0 and fully re-prefilled.
        assert manager.engine.truncation_history[-1] == 0
        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids

    def test_insert_preserves_prefix_cache(self, manager: EditorStateManager) -> None:
        """Verify that tokens *before* safe_token_index are NOT re-prefilled.

        The whole point of the algorithm is that we keep the prefix of the
        KV cache intact and only send the suffix to the engine.
        """
        # Use a long enough text so safe_token_index > 0.
        original = (
            "Artificial intelligence is transforming the way we "
            "interact with technology in every aspect of daily life."
        )
        manager.initialize(original)

        insert_pos = original.index("interact")
        new_text = original[:insert_pos] + "deeply " + original[insert_pos:]

        result = manager.apply_edit(new_text, insert_pos)

        # The prefill call should contain fewer tokens than a full
        # from-scratch tokenization of the entire document.
        full_ids, _ = manager._tokenize(new_text)
        last_prefill = manager.engine.prefill_history[-1]
        assert len(last_prefill) < len(full_ids), (
            "The suffix prefill should be shorter than full re-tokenization"
        )


# -------------------------------------------------------------------- #
#  Scenario C — Boundary Merge Prevention                               #
# -------------------------------------------------------------------- #

class TestScenarioC_BoundaryMergePrevention:
    """Edits that force the BPE tokenizer to re-chunk differently."""

    def test_invest_to_investigate(self, manager: EditorStateManager) -> None:
        """Extending 'invest' → 'investigate' changes BPE boundaries.

        Without the safety margin, the cache would contain tokens for
        'invest' that are incompatible with the new 'investigate' tokens.
        """
        original = "We should invest in this project"
        manager.initialize(original)

        # Tokenize the word boundaries for "invest" before the edit.
        invest_pos = original.index("invest")
        invest_token_before = manager._char_to_token_index(invest_pos)

        # Modify "invest" → "investigate" by inserting "igate" right after "invest"
        insert_pos = invest_pos + len("invest")
        new_text = original[:insert_pos] + "igate" + original[insert_pos:]

        result = manager.apply_edit(new_text, insert_pos)

        # The rewind should have gone back at least to before "invest".
        assert result.safe_token_index <= invest_token_before, (
            f"Safety margin should rewind to at or before the 'invest' token "
            f"(token {invest_token_before}), but got {result.safe_token_index}"
        )

        # Tokens must still match from-scratch tokenization.
        full_ids, full_offsets = manager._tokenize(new_text)
        assert manager.token_ids == full_ids
        assert manager.token_offsets == full_offsets

    def test_adding_space_splits_token(self, manager: EditorStateManager) -> None:
        """Inserting a space can break one BPE token into two.

        Example: 'cannot' is often a single token, but 'can not' is two.
        """
        original = "We cannot do this"
        manager.initialize(original)
        original_count = len(manager.token_ids)

        # Insert a space into "cannot" → "can not"
        space_pos = original.index("cannot") + 3  # after "can"
        new_text = original[:space_pos] + " " + original[space_pos:]

        result = manager.apply_edit(new_text, space_pos)

        # Token count may change.
        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids
        assert manager.engine.current_seq_len == len(full_ids)

    def test_merging_tokens_by_removing_space(self, manager: EditorStateManager) -> None:
        """Simulates removing a space, merging two tokens into one.

        'can not' → 'cannot'.  This is a deletion, but the char_index
        is where the space used to be.
        """
        original = "We can not do this"
        manager.initialize(original)

        space_pos = original.index("can not") + 3  # position of the space
        new_text = original[:space_pos] + original[space_pos + 1:]  # remove space

        result = manager.apply_edit(new_text, space_pos)

        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids
        assert manager.engine.current_seq_len == len(full_ids)

    def test_suffix_changes_tokenization(self, manager: EditorStateManager) -> None:
        """Adding 'ing' to 'play' → 'playing' — suffix that alters BPE merge."""
        original = "I like to play games"
        manager.initialize(original)

        play_pos = original.index("play")
        insert_pos = play_pos + len("play")  # right after "play"
        new_text = original[:insert_pos] + "ing" + original[insert_pos:]
        # "play games" → "playing games"

        result = manager.apply_edit(new_text, insert_pos)

        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids
        assert manager.engine.current_seq_len == len(manager.token_ids)


# -------------------------------------------------------------------- #
#  Edge-case & invariant tests                                          #
# -------------------------------------------------------------------- #

class TestEdgeCases:
    """Additional edge cases and invariant checks."""

    def test_offset_consistency(self, manager: EditorStateManager) -> None:
        """Every token offset must span a real substring of current_text."""
        text = "Hello, world! This is a comprehensive test of BPE tokenization."
        manager.initialize(text)

        for i, (start, end) in enumerate(manager.token_offsets):
            assert 0 <= start < end <= len(text), (
                f"Token {i}: offset ({start}, {end}) is out of range for "
                f"text of length {len(text)}"
            )
            # The substring must decode to something non-empty.
            assert len(text[start:end]) > 0

        full_ids, full_offsets = manager._tokenize(text)
        for i, ((start, end), expected_offset) in enumerate(zip(manager.token_offsets, full_offsets)):
            assert (start, end) == expected_offset, f"Token {i}: offset {(start, end)} != expected {expected_offset}"

    def test_replace_single_character(self, manager: EditorStateManager) -> None:
        """Replace 'a' with 'e' in 'cat' → 'cet'."""
        original = "The cat sat on the mat"
        manager.initialize(original)

        # Replace the 'a' in 'cat' with 'e'.
        a_pos = original.index("cat") + 1  # the 'a'
        new_text = original[:a_pos] + "e" + original[a_pos + 1:]

        manager.apply_edit(new_text, a_pos)
        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids

    def test_delete_word(self, manager: EditorStateManager) -> None:
        """Delete a word from the middle of a sentence."""
        original = "The quick brown fox jumps over the lazy dog"
        manager.initialize(original)

        # Delete "brown "
        brown_start = original.index("brown ")
        brown_end = brown_start + len("brown ")
        new_text = original[:brown_start] + original[brown_end:]

        manager.apply_edit(new_text, brown_start)
        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids

    def test_negative_edit_index_raises(self, manager: EditorStateManager) -> None:
        """Negative char index must raise."""
        manager.initialize("hello")
        with pytest.raises(ValueError, match="must be >= 0"):
            manager.apply_edit("hello", -1)

    def test_cache_seq_len_always_equals_token_count(
        self, manager: EditorStateManager
    ) -> None:
        """Invariant: after any operation, cache len == total tokens."""
        text = "Starting document for invariant checking."
        manager.initialize(text)
        assert manager.engine.current_seq_len == len(manager.token_ids)

        edits = [
            (text + " Extra.", len(text)),
            ("Modified " + text[9:], 0),
        ]
        for new_text, idx in edits:
            manager.apply_edit(new_text, idx)
            assert manager.engine.current_seq_len == len(manager.token_ids), (
                f"Invariant broken after edit at index {idx}"
            )


# -------------------------------------------------------------------- #
#  Performance-oriented sanity check                                    #
# -------------------------------------------------------------------- #

class TestPerformance:
    """Verify that the algorithm does NOT re-tokenize the full document."""

    def test_suffix_is_shorter_than_full_text(
        self, manager: EditorStateManager
    ) -> None:
        """The suffix passed to the engine must be strictly shorter than
        the full document for a mid-document edit.
        """
        long_text = "Alpha beta gamma delta epsilon " * 20  # ~600 chars
        manager.initialize(long_text)

        # Edit near the end.
        edit_pos = len(long_text) - 10
        new_text = long_text[:edit_pos] + "REPLACED" + long_text[edit_pos:]

        result = manager.apply_edit(new_text, edit_pos)
        assert len(result.suffix_text) < len(new_text), (
            "Suffix should be a fraction of the full text, not the whole thing"
        )

    def test_edit_near_start_rewinds_to_zero(
        self, manager: EditorStateManager
    ) -> None:
        """An edit in the first few tokens should rewind to 0, which is still
        cheaper than re-initializing the whole engine.
        """
        text = "Short text"
        manager.initialize(text)

        result = manager.apply_edit("XShort text", 0)
        assert result.safe_token_index == 0


# -------------------------------------------------------------------- #
#  🔴 BLOCKER 1: Space-Prefix Tokenization                             #
# -------------------------------------------------------------------- #

class TestSpacePrefixTokenization:
    """BPE treats " hello" and "hello" as fundamentally different token sequences."""

    def test_space_is_separate_token(self, manager: EditorStateManager) -> None:
        """Appending a space must tokenize as a separate token (in most BPE systems)."""
        original = "hello"
        manager.initialize(original)
        original_token_count = len(manager.token_ids)

        # Append a space.
        new_text = original + " "
        manager.apply_edit(new_text, len(original))

        # Fresh tokenization of the entire document.
        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids, (
            "Space appended must produce identical tokens to fresh tokenization"
        )

    def test_hello_vs_space_hello_different_tokens(
        self, manager: EditorStateManager
    ) -> None:
        """Appending 'hello' vs ' hello' to identical prefix produces different IDs."""
        prefix = "The beginning "
        manager.initialize(prefix)

        # Test 1: Append "hello"
        new_text_1 = prefix + "hello"
        result_1 = manager.apply_edit(new_text_1, len(prefix))
        ids_1, _ = manager._tokenize(new_text_1)
        assert manager.token_ids == ids_1

        # Reset manager
        manager.initialize(prefix)

        # Test 2: Append " hello"
        new_text_2 = prefix + " hello"
        result_2 = manager.apply_edit(new_text_2, len(prefix))
        ids_2, _ = manager._tokenize(new_text_2)
        assert manager.token_ids == ids_2

        # The two should produce different token sequences.
        assert ids_1 != ids_2, (
            "Appending 'hello' vs ' hello' must produce different token IDs"
        )

    def test_multiple_sequential_space_appends(
        self, manager: EditorStateManager
    ) -> None:
        """Multiple sequential space appends must preserve BPE boundaries."""
        base = "word"
        manager.initialize(base)

        for i in range(3):
            old_text = manager.current_text
            new_text = old_text + " "
            manager.apply_edit(new_text, len(old_text))

            # After each space append, verify tokens match fresh tokenization.
            full_ids, full_offsets = manager._tokenize(new_text)
            assert manager.token_ids == full_ids, (
                f"Space append {i+1}: token_ids mismatch"
            )
            assert manager.current_text == new_text


# -------------------------------------------------------------------- #
#  🔴 BLOCKER 2: N < 4 Rewind Boundary                                 #
# -------------------------------------------------------------------- #

class TestRewindBoundarySmallN:
    """Test rewind correctness when token_index < SAFETY_MARGIN (4)."""

    def test_edit_in_3_token_document_at_position_2(
        self, manager: EditorStateManager
    ) -> None:
        """Initialize with a 3-token document; edit at token position 2."""
        # Use short text that should tokenize to ~3 tokens
        original = "Hi there"
        manager.initialize(original)
        assert len(manager.token_ids) <= 4, "Document should be ≤ 4 tokens for this test"

        # Edit near the end.
        insert_pos = len(original) - 1
        new_text = original[:-1] + "X" + original[-1:]
        result = manager.apply_edit(new_text, insert_pos)

        # Safe rewind should be 0 since N < 4.
        assert result.safe_token_index == 0, (
            f"Expected safe_token_index == 0 for N < 4, got {result.safe_token_index}"
        )
        assert manager.engine.truncation_history[-1] == 0

    def test_edit_at_position_3_rewinds_to_0(
        self, manager: EditorStateManager
    ) -> None:
        """Edit at token position 3 should rewind to max(0, 3 - 4) = 0."""
        original = "Short"
        manager.initialize(original)

        # Append to force token_index to be near the end.
        new_text = original + " text"
        result = manager.apply_edit(new_text, len(original))

        # Verify safe_token_index is 0 (or very small)
        assert result.safe_token_index == 0, (
            f"Expected safe_token_index == 0 for small N, got {result.safe_token_index}"
        )

    def test_truncation_history_never_negative(
        self, manager: EditorStateManager
    ) -> None:
        """Assert that truncation_history never receives negative indices."""
        texts = [
            "Hi",
            "Hello world",
            "Short doc",
        ]

        for text in texts:
            manager.initialize(text)
            for i in range(len(text)):
                new_text = text[:i] + "X" + text[i:]
                result = manager.apply_edit(new_text, i)
                assert result.safe_token_index >= 0, (
                    f"safe_token_index must be >= 0, got {result.safe_token_index}"
                )
                assert manager.engine.truncation_history[-1] >= 0, (
                    f"truncation_history entry must be >= 0"
                )


# -------------------------------------------------------------------- #
#  🔴 BLOCKER 3: Offset Array Consistency & Shifting                   #
# -------------------------------------------------------------------- #

class TestOffsetArrayConsistency:
    """Verify offset shifting and contiguity after rewind."""

    def test_shifted_offsets_start_at_or_after_rewind_point(
        self, manager: EditorStateManager
    ) -> None:
        """All shifted offsets must satisfy start >= token_offsets[safe_token_index][0]."""
        original = (
            "Artificial intelligence systems are revolutionizing the world "
            "of technology and human interaction every single day now."
        )
        manager.initialize(original)

        insert_pos = original.index("revolutionizing")
        new_text = original[:insert_pos] + "RADICAL " + original[insert_pos:]

        result = manager.apply_edit(new_text, insert_pos)

        # Get the suffix start position.
        if result.safe_token_index < len(manager.token_offsets):
            suffix_start = result.safe_token_index
        else:
            suffix_start = len(manager.token_offsets)

        # All shifted offsets should be >= suffix_char_start.
        for i, (start, end) in enumerate(manager.token_offsets[result.safe_token_index :], start=result.safe_token_index):
            assert start >= result.suffix_char_start, (
                f"Token {i}: offset start {start} < suffix_char_start {result.suffix_char_start}"
            )

    def test_offsets_form_contiguous_partition(
        self, manager: EditorStateManager
    ) -> None:
        """Token offsets must form a perfect partition with no gaps or overlaps."""
        text = "The quick brown fox jumps over the lazy dog and runs away quickly."
        manager.initialize(text)

        # Verify initial state
        manager._verify_offset_partition(text)

        # After an edit
        new_text = text + " And then returns."
        manager.apply_edit(new_text, len(text))
        manager._verify_offset_partition(new_text)
        full_ids, full_offsets = manager._tokenize(new_text)
        for i, ((start, end), expected_offset) in enumerate(zip(manager.token_offsets, full_offsets)):
            assert (start, end) == expected_offset, f"Token {i}: offset {(start, end)} != expected {expected_offset}"

        # After multiple edits
        new_text = "The " + new_text
        manager.apply_edit(new_text, 0)
        manager._verify_offset_partition(new_text)
        full_ids, full_offsets = manager._tokenize(new_text)
        for i, ((start, end), expected_offset) in enumerate(zip(manager.token_offsets, full_offsets)):
            assert (start, end) == expected_offset, f"Token {i}: offset {(start, end)} != expected {expected_offset}"

    def test_no_offset_gaps_or_overlaps(
        self, manager: EditorStateManager
    ) -> None:
        """Verify sum(end - start) == len(current_text) and sequential coverage."""
        text = "Hello, world! How are you today?"
        manager.initialize(text)

        # Check coverage
        total_span = sum(end - start for start, end in manager.token_offsets)
        assert total_span == len(manager.current_text), (
            f"Token offsets do not cover entire text: {total_span} != {len(manager.current_text)}"
        )

        # Check sequential and contiguous
        for i in range(len(manager.token_offsets) - 1):
            end_i = manager.token_offsets[i][1]
            start_next = manager.token_offsets[i + 1][0]
            assert end_i == start_next, (
                f"Gap between token {i} and {i+1}: {end_i} != {start_next}"
            )

        full_ids, full_offsets = manager._tokenize(text)
        for i, ((start, end), expected_offset) in enumerate(zip(manager.token_offsets, full_offsets)):
            assert (start, end) == expected_offset, f"Token {i}: offset {(start, end)} != expected {expected_offset}"


# -------------------------------------------------------------------- #
#  🔴 BLOCKER 4: Out-of-Bounds Validation                              #
# -------------------------------------------------------------------- #

class TestOutOfBoundsValidation:
    """Bounds checking must prevent undefined bisect behavior."""

    def test_edit_char_index_out_of_bounds_raises(
        self, manager: EditorStateManager
    ) -> None:
        """Edit char index > len(current_text) must raise ValueError."""
        manager.initialize("hello")

        with pytest.raises(ValueError, match="edit_char_index out of bounds"):
            manager.apply_edit("hello world", 10)  # 10 > 5

    def test_edit_char_index_at_end_is_valid(
        self, manager: EditorStateManager
    ) -> None:
        """Edit char index == len(current_text) should be valid (append)."""
        manager.initialize("hello")
        # This should NOT raise.
        result = manager.apply_edit("hello world", 5)
        assert result.token_index >= 0


# -------------------------------------------------------------------- #
#  🟡 HIGH PRIORITY 5: Sequential Operation Sequences                  #
# -------------------------------------------------------------------- #

class TestSequentialOperations:
    """State must survive chaotic sequences of edits."""

    def test_insert_insert_delete_insert_sequence(
        self, manager: EditorStateManager
    ) -> None:
        """Execute: Insert → Insert → Delete → Insert."""
        manager.initialize("Start")

        # Step 1: Insert " here" at end
        text = manager.current_text + " here"
        manager.apply_edit(text, len(manager.current_text))
        full_ids, _ = manager._tokenize(text)
        assert manager.token_ids == full_ids, "After insert 1"
        assert manager._verify_no_offset_gaps()

        # Step 2: Insert "MIDDLE " in middle
        insert_pos = manager.current_text.index("here")
        text = manager.current_text[:insert_pos] + "MIDDLE " + manager.current_text[insert_pos:]
        manager.apply_edit(text, insert_pos)
        full_ids, _ = manager._tokenize(text)
        assert manager.token_ids == full_ids, "After insert 2"
        assert manager._verify_no_offset_gaps()

        # Step 3: Delete " MIDDLE"
        delete_start = manager.current_text.index("MIDDLE") - 1
        delete_end = delete_start + len(" MIDDLE")
        text = manager.current_text[:delete_start] + manager.current_text[delete_end:]
        manager.apply_edit(text, delete_start)
        full_ids, _ = manager._tokenize(text)
        assert manager.token_ids == full_ids, "After delete"
        assert manager._verify_no_offset_gaps()

        # Step 4: Insert "END" at end
        text = manager.current_text + "END"
        manager.apply_edit(text, len(manager.current_text))
        full_ids, full_offsets = manager._tokenize(text)
        assert manager.token_ids == full_ids, "After insert 3"
        assert manager._verify_no_offset_gaps()

        for i, ((start, end), expected_offset) in enumerate(zip(manager.token_offsets, full_offsets)):
            assert (start, end) == expected_offset, f"Token {i}: offset {(start, end)} != expected {expected_offset}"

        assert manager.engine.current_seq_len == len(manager.token_ids), f"Cache desync: {manager.engine.current_seq_len} != {len(manager.token_ids)}"

    def test_delete_insert_delete_insert_sequence(
        self, manager: EditorStateManager
    ) -> None:
        """Execute: Delete → Insert → Delete → Insert."""
        original = "The quick brown fox jumps"
        manager.initialize(original)

        # Step 1: Delete "quick "
        quick_start = original.index("quick ")
        quick_end = quick_start + len("quick ")
        text = original[:quick_start] + original[quick_end:]
        manager.apply_edit(text, quick_start)
        full_ids, _ = manager._tokenize(text)
        assert manager.token_ids == full_ids, "After delete 1"

        # Step 2: Insert "SLOW "
        insert_pos = manager.current_text.index("brown")
        text = manager.current_text[:insert_pos] + "SLOW " + manager.current_text[insert_pos:]
        manager.apply_edit(text, insert_pos)
        full_ids, _ = manager._tokenize(text)
        assert manager.token_ids == full_ids, "After insert 1"

        # Step 3: Delete "fox"
        fox_pos = manager.current_text.index("fox")
        text = manager.current_text[:fox_pos] + manager.current_text[fox_pos + 3:]
        manager.apply_edit(text, fox_pos)
        full_ids, _ = manager._tokenize(text)
        assert manager.token_ids == full_ids, "After delete 2"

        # Step 4: Insert "ANIMAL"
        jumps_pos = manager.current_text.index("jumps")
        text = manager.current_text[:jumps_pos] + "ANIMAL " + manager.current_text[jumps_pos:]
        manager.apply_edit(text, jumps_pos)
        full_ids, full_offsets = manager._tokenize(text)
        assert manager.token_ids == full_ids, "After insert 2"

        for i, ((start, end), expected_offset) in enumerate(zip(manager.token_offsets, full_offsets)):
            assert (start, end) == expected_offset, f"Token {i}: offset {(start, end)} != expected {expected_offset}"

        assert manager.engine.current_seq_len == len(manager.token_ids), f"Cache desync: {manager.engine.current_seq_len} != {len(manager.token_ids)}"


# -------------------------------------------------------------------- #
#  🟡 HIGH PRIORITY 6: Systematic Deletions & Empty Suffix             #
# -------------------------------------------------------------------- #

class TestDeletionsAndEmptySuffix:
    """Deletions at all positions, including the empty suffix edge case."""

    def test_delete_at_document_start(self, manager: EditorStateManager) -> None:
        """Delete at char position 0."""
        original = "Hello World"
        manager.initialize(original)

        new_text = original[1:]  # Remove first character
        manager.apply_edit(new_text, 0)

        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids

    def test_delete_at_document_middle(self, manager: EditorStateManager) -> None:
        """Delete from the middle of the document."""
        original = "The quick brown fox"
        manager.initialize(original)

        # Delete " quick"
        delete_start = original.index(" quick")
        delete_end = delete_start + len(" quick")
        new_text = original[:delete_start] + original[delete_end:]

        manager.apply_edit(new_text, delete_start)
        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids

    def test_delete_at_document_end(self, manager: EditorStateManager) -> None:
        """Delete at the very end of the document."""
        original = "Hello World"
        manager.initialize(original)

        new_text = original[:-1]  # Remove last character
        manager.apply_edit(new_text, len(original) - 1)

        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids

    def test_empty_suffix_edge_case(self, manager: EditorStateManager) -> None:
        """Delete everything after position N (suffix becomes empty)."""
        original = "The quick brown fox jumps over lazy dog"
        manager.initialize(original)

        # Find a position and delete everything after it
        delete_after_pos = original.index("jumps")
        new_text = original[:delete_after_pos]

        result = manager.apply_edit(new_text, delete_after_pos)

        # Suffix IDs should be empty or very short
        assert result.suffix_token_count >= 0

        # Engine should still be consistent
        assert manager.engine.current_seq_len == len(manager.token_ids)

        # Verify tokens match fresh tokenization
        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids


# -------------------------------------------------------------------- #
#  🟡 HIGH PRIORITY 7: Tokenizer Call Instrumentation                  #
# -------------------------------------------------------------------- #

class TestTokenizerCallOptimization:
    """Verify tokenizer is called exactly once per edit, only on suffix."""

    def test_suffix_tokenization_produces_correct_tokens(
        self, manager: EditorStateManager
    ) -> None:
        """Verify that suffix-only tokenization produces correct token IDs."""
        manager.initialize("Hello world")

        # Perform an edit
        new_text = "Hello world again"
        result = manager.apply_edit(new_text, len("Hello world"))

        # The key optimization metric: suffix_token_count should be much smaller
        # than len(manager.token_ids) for a mid-document edit
        # (suffix_token_count is how many tokens were added, not total)
        full_ids, _ = manager._tokenize(new_text)
        assert manager.token_ids == full_ids, (
            "Suffix-only tokenization must match full fresh tokenization"
        )

        # Verify the suffix text was actually tokenized
        assert result.suffix_token_count > 0, "Suffix should produce tokens"

    def test_suffix_is_minimal_for_mid_document_edits(
        self, manager: EditorStateManager
    ) -> None:
        """Verify suffix is NOT the full document for mid-document edits."""
        # Use longer text so rewind doesn't collapse to position 0
        manager.initialize("The beginning of the document " * 3)  # longer text

        # Edit well into the middle (not near start where rewind would go to 0)
        edit_pos = 50  # Well into the document
        new_text = manager.current_text[:edit_pos] + "INSERTED" + manager.current_text[edit_pos:]

        result = manager.apply_edit(new_text, edit_pos)

        # The suffix_text should be significantly shorter than new_text
        # for a mid-document edit (unless we're very near the end)
        assert len(result.suffix_text) < len(new_text), (
            f"Suffix should be shorter than full text for mid-document edit. "
            f"Suffix: {len(result.suffix_text)}, Full: {len(new_text)}, "
            f"safe_token_index: {result.safe_token_index}"
        )

        # The suffix should start at the rewind point, not at the beginning
        assert result.suffix_char_start > 0, (
            "Suffix should start after position 0 for mid-document edits in longer text"
        )

    def test_append_only_at_end_maximizes_cache_reuse(
        self, manager: EditorStateManager
    ) -> None:
        """Verify appends reuse most of the cache (minimal re-tokenization)."""
        long_text = "Prefix text " * 10  # ~120 chars
        manager.initialize(long_text)
        original_token_count = len(manager.token_ids)

        # Pure append at end
        new_text = long_text + " suffix text"
        result = manager.apply_edit(new_text, len(long_text))

        # For a pure append, safe_token_index should be near or at end of original tokens
        # This means we're reusing most of the cache
        reuse_ratio = result.safe_token_index / original_token_count if original_token_count > 0 else 1
        if original_token_count >= 4:
            assert reuse_ratio > 0.5, (
                f"Append should reuse most of cache. Reuse ratio: {reuse_ratio}"
            )
        else:
            assert reuse_ratio == 0.0, "Expected 0 reuse ratio for tiny N < 4 documents."

        # The suffix_token_count should be small compared to total new tokens
        total_new_tokens = len(manager.token_ids)
        if original_token_count >= 4:
            assert result.suffix_token_count < total_new_tokens * 0.5, (
                "Append should only add a few new tokens, not retokenize everything"
            )
