from engine import (
    MockExLlamaEngine,
    build_llama_autocomplete_prompt,
    build_llama_rewrite_prompt,
    build_qwen_fim_prompt,
    parse_correction_suggestions,
    sanitize_autocomplete_completion,
)


def test_fim_prompt_cursor_at_end() -> None:
    assert (
        build_qwen_fim_prompt("def add(a, b):", len("def add(a, b):"))
        == "<|fim_prefix|>def add(a, b):<|fim_suffix|><|fim_middle|>"
    )


def test_fim_prompt_cursor_in_middle() -> None:
    assert (
        build_qwen_fim_prompt("return value", 6)
        == "<|fim_prefix|>return<|fim_suffix|> value<|fim_middle|>"
    )


def test_fim_prompt_empty_document() -> None:
    assert (
        build_qwen_fim_prompt("", 0)
        == "<|fim_prefix|><|fim_suffix|><|fim_middle|>"
    )


def test_fim_prompt_multiline_code() -> None:
    document = "def add(a, b):\n    return a + b\n"
    cursor = document.index("return")

    assert build_qwen_fim_prompt(document, cursor) == (
        "<|fim_prefix|>def add(a, b):\n    "
        "<|fim_suffix|>return a + b\n"
        "<|fim_middle|>"
    )


def test_fim_prompt_clamps_out_of_range_cursor() -> None:
    assert (
        build_qwen_fim_prompt("abc", 99)
        == "<|fim_prefix|>abc<|fim_suffix|><|fim_middle|>"
    )
    assert (
        build_qwen_fim_prompt("abc", -5)
        == "<|fim_prefix|><|fim_suffix|>abc<|fim_middle|>"
    )


def test_llama_rewrite_prompt_uses_llama_chat_template() -> None:
    prompt = build_llama_rewrite_prompt(
        "this is too casual",
        "Make this more professional",
    )

    assert "<|begin_of_text|>" in prompt
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt
    assert "Make this more professional" in prompt
    assert "this is too casual" in prompt
    assert "Output only the final rewritten text" in prompt
    assert "Do not add a preface" in prompt
    assert "one complete sentence or paragraph" in prompt
    assert "<|eot_id|>" in prompt
    assert "<|fim_prefix|>" not in prompt
    assert "<|fim_suffix|>" not in prompt
    assert "<|fim_middle|>" not in prompt


def test_llama_autocomplete_prompt_uses_prefix_suffix_context() -> None:
    prompt = build_llama_autocomplete_prompt("Hello brave world", 11)

    assert "Continue the text at the cursor with 1 to 8 words" in prompt
    assert "Before cursor:\nHello brave" in prompt
    assert "After cursor:\n world" in prompt
    assert "<|fim_prefix|>" not in prompt
    assert "<|fim_suffix|>" not in prompt


def test_autocomplete_sanitizer_rejects_code_and_prefaces() -> None:
    assert sanitize_autocomplete_completion("```python\nprint(1)", "Hello", 5) == ""
    assert sanitize_autocomplete_completion("Here is the continuation", "Hello", 5) == ""
    assert sanitize_autocomplete_completion("After cursor:", "Hello", 5) == ""
    assert sanitize_autocomplete_completion("Before cursor:", "Hello", 5) == ""
    assert sanitize_autocomplete_completion(" gentle useful words for today", "Hello", 5) == (
        "gentle useful words for today"
    )


def test_autocomplete_sanitizer_strips_prompt_label_when_content_follows() -> None:
    assert sanitize_autocomplete_completion("After cursor: should proceed carefully", "The team", 8) == (
        "should proceed carefully"
    )


def test_autocomplete_sanitizer_limits_phrase_length() -> None:
    completion = " one two three four five six seven eight nine ten"

    assert sanitize_autocomplete_completion(completion, "Hello", 5) == (
        "one two three four five six seven eight"
    )


def test_parse_correction_suggestions_validates_and_shifts_offsets() -> None:
    raw = '[{"start": 4, "end": 7, "replacement": "are", "reason": "grammar"}]'

    assert parse_correction_suggestions(raw, 10, "You is kind") == [
        {"start": 14, "end": 17, "replacement": "are", "reason": "grammar"}
    ]


def test_parse_correction_suggestions_accepts_multiline_json() -> None:
    raw = """
[
  {"start": 4, "end": 7, "replacement": "are", "reason": "grammar"}
]
"""

    assert parse_correction_suggestions(raw, 0, "You is kind") == [
        {"start": 4, "end": 7, "replacement": "are", "reason": "grammar"}
    ]


def test_parse_correction_suggestions_rejects_malformed_json() -> None:
    assert parse_correction_suggestions("not json", 0, "You is kind") == []
    assert (
        parse_correction_suggestions(
            '[{"start": 4, "end": 99, "replacement": "are", "reason": "grammar"}]',
            0,
            "You is kind",
        )
        == []
    )


def test_mock_engine_correction_path_returns_obvious_suggestions() -> None:
    engine = MockExLlamaEngine()
    suggestions = engine.generate_corrections(
        "The manager were happy with the report, but it contain several typo and missing punctuation",
        83,
        cancel_event=type("CancelEvent", (), {"is_set": lambda self: False})(),
    )

    replacements = {suggestion["replacement"] for suggestion in suggestions}
    assert {"manager was", "it contains", "several typos", "punctuation."} <= replacements
