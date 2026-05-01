from engine import build_llama_rewrite_prompt, build_qwen_fim_prompt


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
    assert "<|eot_id|>" in prompt
    assert "<|fim_prefix|>" not in prompt
    assert "<|fim_suffix|>" not in prompt
    assert "<|fim_middle|>" not in prompt
