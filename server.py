"""
FastAPI WebSocket server for the real-time editor network layer.

The WebSocket listener stays responsive while generation runs in its own
asyncio task. Any new editor payload cancels the active generation before the
state manager applies the incoming edit.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Iterator, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

load_dotenv()

from engine import get_engine
from editor_state_manager import EditorStateManager


class EditorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    action: Literal["edit", "autocomplete", "rewrite"]
    new_text: Optional[str] = None
    edit_char_index: Optional[int] = None
    text: Optional[str] = None
    prompt: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> "EditorPayload":
        if self.action in {"edit", "autocomplete"}:
            if self.new_text is None or self.edit_char_index is None:
                raise ValueError(
                    "edit/autocomplete payloads require new_text and edit_char_index"
                )
        if self.action == "rewrite":
            if self.text is None or self.prompt is None:
                raise ValueError("rewrite payloads require text and prompt")
        return self


class StreamPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    type: Literal["token", "done", "cancelled", "server_error"]
    chunk: str
    latency_ms: int


app = FastAPI(title="Phase 3 Editor WebSocket Server")


def create_state_manager() -> EditorStateManager:
    engine = get_engine()
    manager = EditorStateManager(engine=engine)
    manager.initialize("")
    return manager


# Global for Phase 3. The connection handler still serializes access with a
# lock so later multi-client work can move this to connection scope cleanly.
state_manager = create_state_manager()


def _next_token(token_iterator: Iterator[str]) -> Optional[str]:
    try:
        return next(token_iterator)
    except StopIteration:
        return None


class GenerationTaskManager:
    def __init__(self, websocket: WebSocket, manager: EditorStateManager) -> None:
        self.websocket = websocket
        self.manager = manager
        self.state_lock = asyncio.Lock()
        self.send_lock = asyncio.Lock()
        self.current_task: Optional[asyncio.Task[None]] = None
        self.current_cancel_event: Optional[asyncio.Event] = None

    async def handle_payload(self, payload: EditorPayload) -> None:
        await self.cancel_active_generation()
        started_at = time.perf_counter()

        if payload.action == "rewrite":
            cancel_event = asyncio.Event()
            self.current_cancel_event = cancel_event
            self.current_task = asyncio.create_task(
                self.run_generation(payload, cancel_event)
            )
            return

        try:
            edit_result = await self.apply_edit(payload)
        except Exception as exc:
            await self.safe_send_stream_payload(
                request_id=payload.request_id,
                payload_type="done",
                chunk=f"state_error: {type(exc).__name__}: {exc}",
                started_at=started_at,
            )
            return

        if payload.action == "edit":
            await self.safe_send_stream_payload(
                request_id=payload.request_id,
                payload_type="done",
                chunk=f"state_updated tokens={edit_result.total_cache_len}",
                started_at=started_at,
            )
            return

        cancel_event = asyncio.Event()
        self.current_cancel_event = cancel_event
        self.current_task = asyncio.create_task(
            self.run_generation(payload, cancel_event)
        )

    async def apply_edit(self, payload: EditorPayload) -> Any:
        async with self.state_lock:
            assert payload.new_text is not None
            assert payload.edit_char_index is not None
            edit_char_index = payload.edit_char_index
            if getattr(self.manager.engine, "requires_full_prefill", False):
                edit_char_index = 0
            if edit_char_index > len(self.manager.current_text):
                print(
                    "Clamping stale edit_char_index "
                    f"{edit_char_index} to 0 for current_text_len="
                    f"{len(self.manager.current_text)}",
                    flush=True,
                )
                edit_char_index = 0

            return await asyncio.to_thread(
                self.manager.apply_edit,
                payload.new_text,
                edit_char_index,
            )

    async def cancel_active_generation(self) -> None:
        task = self.current_task
        if task is None or task.done():
            self.current_task = None
            self.current_cancel_event = None
            return

        if self.current_cancel_event is not None:
            self.current_cancel_event.set()
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self.current_task = None
            self.current_cancel_event = None

    async def run_generation(
        self,
        payload: EditorPayload,
        cancel_event: asyncio.Event,
    ) -> None:
        started_at = time.perf_counter()
        cancelled = False
        failed = False

        try:
            token_iterator = await self.create_token_iterator(payload, cancel_event)
            while not cancel_event.is_set():
                token = await asyncio.to_thread(_next_token, token_iterator)
                if token is None:
                    break

                sent = await self.safe_send_stream_payload(
                    request_id=payload.request_id,
                    payload_type="token",
                    chunk=token,
                    started_at=started_at,
                )
                if not sent:
                    cancel_event.set()
                    break

            cancelled = cancel_event.is_set()
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            failed = True
            await self.safe_send_stream_payload(
                request_id=payload.request_id,
                payload_type="server_error",
                chunk=f"generation_error: {type(exc).__name__}: {exc}",
                started_at=started_at,
            )
            return
        finally:
            if cancelled:
                await self.safe_send_stream_payload(
                    request_id=payload.request_id,
                    payload_type="cancelled",
                    chunk="",
                    started_at=started_at,
                )
            elif not failed and not cancel_event.is_set():
                await self.safe_send_stream_payload(
                    request_id=payload.request_id,
                    payload_type="done",
                    chunk="",
                    started_at=started_at,
                )

            if self.current_task is asyncio.current_task():
                self.current_task = None
                self.current_cancel_event = None

    async def create_token_iterator(
        self,
        payload: EditorPayload,
        cancel_event: asyncio.Event,
    ) -> Iterator[str]:
        if payload.action == "autocomplete":
            assert payload.edit_char_index is not None
            if payload.action == "autocomplete" and hasattr(
                self.manager.engine,
                "generate_autocomplete_stream",
            ):
                return await asyncio.to_thread(
                    self.manager.engine.generate_autocomplete_stream,
                    self.manager.current_text,
                    min(
                        max(payload.edit_char_index, 0),
                        len(self.manager.current_text),
                    ),
                    cancel_event,
                    24,
                )
            return await asyncio.to_thread(
                self.manager.engine.generate_stream,
                cancel_event,
                24,
            )

        if payload.action == "rewrite":
            assert payload.text is not None
            assert payload.prompt is not None
            return await asyncio.to_thread(
                self.manager.engine.apply_rewrite,
                payload.text,
                payload.prompt,
                cancel_event,
                512,
            )

        return await asyncio.to_thread(
            self.manager.engine.generate_stream,
            cancel_event,
            80,
        )

    async def send_stream_payload(
        self,
        *,
        request_id: str,
        payload_type: Literal["token", "done", "cancelled", "server_error"],
        chunk: str,
        started_at: float,
    ) -> None:
        payload = StreamPayload(
            request_id=request_id,
            type=payload_type,
            chunk=chunk,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
        async with self.send_lock:
            await self.websocket.send_json(payload.model_dump())

    async def safe_send_stream_payload(self, **kwargs: Any) -> bool:
        """Send payload, returning False if the WebSocket is no longer usable."""
        try:
            await self.send_stream_payload(**kwargs)
            return True
        except (WebSocketDisconnect, RuntimeError) as exc:
            print(f"Failed to send payload: {type(exc).__name__}: {exc}")
            return False
        except Exception as exc:
            print(f"Failed to send payload: {type(exc).__name__}: {exc}")
            return False


@app.websocket("/ws/editor")
async def editor_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    generation_manager = GenerationTaskManager(websocket, state_manager)

    try:
        while True:
            try:
                raw_payload = await websocket.receive_json()
            except (RuntimeError, WebSocketDisconnect):
                # RuntimeError: race condition, socket not ready despite accept() call
                # WebSocketDisconnect: client disconnected before sending data
                break

            try:
                payload = EditorPayload.parse_obj(raw_payload)
            except ValidationError as exc:
                await generation_manager.safe_send_stream_payload(
                    request_id=str(raw_payload.get("request_id", "")),
                    payload_type="done",
                    chunk=f"invalid_payload: {exc.errors()}",
                    started_at=time.perf_counter(),
                )
                continue

            try:
                await generation_manager.handle_payload(payload)
            except Exception as exc:
                await generation_manager.safe_send_stream_payload(
                    request_id=payload.request_id,
                    payload_type="server_error",
                    chunk=f"server_error: {type(exc).__name__}: {exc}",
                    started_at=time.perf_counter(),
                )
                break
    except WebSocketDisconnect:
        await generation_manager.cancel_active_generation()


if __name__ == '__main__':
    import uvicorn
    import os
    host = os.getenv('SERVER_HOST', '127.0.0.1')
    port = int(os.getenv('SERVER_PORT', 8000))
    uvicorn.run(app, host=host, port=port)

