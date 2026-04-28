"""
Client simulator for Phase 3 WebSocket cancellation behavior.

Run the server first:
    uvicorn server:app --reload

Then run:
    python mock_frontend.py
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import time
from typing import Any, Dict

import websockets


SERVER_URL = os.getenv("EDITOR_WS_URL", "ws://127.0.0.1:8000/ws/editor")


async def send_payload(websocket: Any, payload: Dict[str, Any]) -> None:
    print(f">>> {payload}")
    await websocket.send(json.dumps(payload))


async def run_disconnect_probe() -> None:
    async with websockets.connect(SERVER_URL) as websocket:
        await send_payload(
            websocket,
            {
                "request_id": "uuid-disconnect-1",
                "action": "autocomplete",
                "new_text": "The quick brown fox",
                "edit_char_index": 0,
            },
        )
        await asyncio.sleep(0.1)
        await websocket.close()
        print("PASS: disconnected mid-generation without client-side error.")


async def run_cancellation_probe() -> None:
    first_request_id = "uuid-autocomplete-1"
    second_request_id = "uuid-edit-2"

    async with websockets.connect(SERVER_URL) as websocket:
        await send_payload(
            websocket,
            {
                "request_id": first_request_id,
                "action": "autocomplete",
                "new_text": "The quick brown fox",
                "edit_char_index": 0,
            },
        )

        await asyncio.sleep(0.1)

        await send_payload(
            websocket,
            {
                "request_id": second_request_id,
                "action": "edit",
                "new_text": "The quick brown fox!",
                "edit_char_index": len("The quick brown fox"),
            },
        )

        saw_cancelled = False
        saw_edit_done = False
        deadline = time.perf_counter() + 5

        while time.perf_counter() < deadline:
            message = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5))
            print(f"<<< {message}")

            if (
                message["request_id"] == first_request_id
                and message["type"] == "cancelled"
            ):
                saw_cancelled = True
            if (
                message["request_id"] == second_request_id
                and message["type"] == "done"
            ):
                saw_edit_done = True

            if saw_cancelled and saw_edit_done:
                print("PASS: autocomplete was cancelled and edit stayed responsive.")
                return

        raise RuntimeError(
            "Did not observe both cancellation of the first request and "
            "completion of the edit request."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disconnect-midway",
        action="store_true",
        help="Close the WebSocket while generation is in flight.",
    )
    args = parser.parse_args()

    if args.disconnect_midway:
        asyncio.run(run_disconnect_probe())
    else:
        asyncio.run(run_cancellation_probe())
