import asyncio
import os

import pytest
from pydantic import ValidationError

os.environ["USE_MOCK_ENGINE"] = "true"

from server import EditorPayload, GenerationTaskManager


def test_rewrite_payload_accepts_text_and_prompt_without_edit_fields() -> None:
    payload = EditorPayload(
        request_id="rewrite-1",
        action="rewrite",
        text="hello there",
        prompt="Make this more formal",
    )

    assert payload.text == "hello there"
    assert payload.prompt == "Make this more formal"
    assert payload.new_text is None
    assert payload.edit_char_index is None


def test_rewrite_payload_requires_text_and_prompt() -> None:
    with pytest.raises(ValidationError):
        EditorPayload(request_id="rewrite-1", action="rewrite", text="hello")


def test_autocomplete_payload_requires_editor_fields() -> None:
    with pytest.raises(ValidationError):
        EditorPayload(request_id="auto-1", action="autocomplete", new_text="hello")


def test_correct_payload_requires_editor_fields() -> None:
    with pytest.raises(ValidationError):
        EditorPayload(request_id="correct-1", action="correct", new_text="hello")


class FakeWebSocket:
    def __init__(self) -> None:
        self.payloads = []

    async def send_json(self, payload: dict) -> None:
        self.payloads.append(payload)


class FakeEngine:
    def apply_rewrite(self, text, instruction, cancel_event, max_new_tokens):
        yield f"{instruction}: {text}"


class FakeManager:
    def __init__(self) -> None:
        self.engine = FakeEngine()
        self.current_text = "editor text"
        self.apply_edit_called = False

    def apply_edit(self, *args, **kwargs):
        self.apply_edit_called = True
        raise AssertionError("rewrite must not mutate editor state")


def test_rewrite_route_skips_editor_state_mutation() -> None:
    async def run_test() -> None:
        websocket = FakeWebSocket()
        manager = FakeManager()
        generation_manager = GenerationTaskManager(websocket, manager)
        payload = EditorPayload(
            request_id="rewrite-1",
            action="rewrite",
            text="hello",
            prompt="Polish",
        )

        await generation_manager.handle_payload(payload)
        assert generation_manager.current_task is not None
        await generation_manager.current_task

        assert manager.apply_edit_called is False
        assert websocket.payloads[0]["type"] == "token"
        assert websocket.payloads[0]["chunk"] == "Polish: hello"
        assert websocket.payloads[-1]["type"] == "done"

    asyncio.run(run_test())
