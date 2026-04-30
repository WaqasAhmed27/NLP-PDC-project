from engine import build_qwen_fim_prompt


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
