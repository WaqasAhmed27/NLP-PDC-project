"""
Phase 3 Validation: Concurrent PDC Stress Test

Spawns 50-100 concurrent WebSocket clients to stress-test the hardened
server.py router. Each worker simulates rapid editing with race conditions,
cancellations, and abrupt disconnects.

Metrics tracked:
- Successful connections
- Autocomplete requests sent
- Cancelled responses received
- Server errors encountered
- Edit acknowledgment latency (target: < 50ms)
- Thread pool saturation detection

Usage:
    python stress_test.py [--workers 50] [--loops 10]

Example (aggressive 100 workers, 10 loops each):
    python stress_test.py --workers 100 --loops 10
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import websockets


SERVER_URL = os.getenv("EDITOR_WS_URL", "ws://127.0.0.1:8000/ws/editor")


@dataclass
class StressTestMetrics:
    """Aggregated metrics from all workers."""
    total_connections: int = 0
    total_autocompletes_sent: int = 0
    total_edits_sent: int = 0
    total_cancelled_responses: int = 0
    total_done_responses: int = 0
    total_errors: int = 0
    total_disconnects_caused: int = 0
    edit_latencies_ms: list[float] = field(default_factory=list)
    connection_failures: int = 0
    malformed_responses: int = 0

    def report(self) -> None:
        """Print a comprehensive stress test report."""
        print("\n" + "=" * 80)
        print("PHASE 3 VALIDATION: CONCURRENT PDC STRESS TEST RESULTS")
        print("=" * 80)

        print(f"\n📊 Connection Statistics:")
        print(f"   Total Connections Established: {self.total_connections}")
        print(f"   Connection Failures: {self.connection_failures}")
        success_rate = (
            (self.total_connections / (self.total_connections + self.connection_failures) * 100)
            if (self.total_connections + self.connection_failures) > 0
            else 0
        )
        print(f"   Connection Success Rate: {success_rate:.1f}%")

        print(f"\n📈 Request Statistics:")
        print(f"   Autocomplete Requests Sent: {self.total_autocompletes_sent}")
        print(f"   Edit Requests Sent: {self.total_edits_sent}")
        print(f"   Cancelled Responses Received: {self.total_cancelled_responses}")
        print(f"   Done Responses Received: {self.total_done_responses}")
        cancel_rate = (
            (self.total_cancelled_responses / self.total_autocompletes_sent * 100)
            if self.total_autocompletes_sent > 0
            else 0
        )
        print(f"   Cancellation Success Rate: {cancel_rate:.1f}%")

        print(f"\n🔥 Failure Statistics:")
        print(f"   Total Errors/Exceptions: {self.total_errors}")
        print(f"   Abrupt Disconnects Caused: {self.total_disconnects_caused}")
        print(f"   Malformed Responses: {self.malformed_responses}")

        print(f"\n⏱️ Latency Analysis (Edit Acknowledgment):")
        if self.edit_latencies_ms:
            min_latency = min(self.edit_latencies_ms)
            max_latency = max(self.edit_latencies_ms)
            avg_latency = sum(self.edit_latencies_ms) / len(self.edit_latencies_ms)
            p99_latency = sorted(self.edit_latencies_ms)[
                int(len(self.edit_latencies_ms) * 0.99)
            ] if len(self.edit_latencies_ms) > 1 else min_latency

            print(f"   Min Latency: {min_latency:.2f}ms")
            print(f"   Max Latency: {max_latency:.2f}ms")
            print(f"   Avg Latency: {avg_latency:.2f}ms")
            print(f"   P99 Latency: {p99_latency:.2f}ms")

            # Check if latencies spike (thread pool saturation indicator)
            slow_requests = sum(1 for lat in self.edit_latencies_ms if lat > 50)
            slow_rate = (slow_requests / len(self.edit_latencies_ms)) * 100
            print(f"   Requests > 50ms: {slow_requests}/{len(self.edit_latencies_ms)} ({slow_rate:.1f}%)")

            if max_latency > 500:
                print(f"   ⚠️  WARNING: Max latency exceeded 500ms (possible thread pool saturation)")
            elif max_latency > 200:
                print(f"   ⚠️  CAUTION: Max latency exceeded 200ms")
            else:
                print(f"   ✅ All requests under 200ms (event loop responsive)")
        else:
            print(f"   No latency data collected")

        print(f"\n✅ Summary:")
        total_events = (
            self.total_autocompletes_sent
            + self.total_edits_sent
            + self.total_cancelled_responses
            + self.total_errors
        )
        print(f"   Total Events: {total_events}")
        success_events = self.total_cancelled_responses + self.total_done_responses
        overall_success = (success_events / total_events * 100) if total_events > 0 else 0
        print(f"   Successful Operations: {success_events}/{total_events} ({overall_success:.1f}%)")

        if self.total_errors > 0 or self.malformed_responses > 0:
            print(f"   ⚠️  ISSUES DETECTED: {self.total_errors} errors, {self.malformed_responses} malformed")
            print(f"   Status: 🟡 DEGRADED")
        elif max_latency > 200 if self.edit_latencies_ms else False:
            print(f"   Status: 🟡 ACCEPTABLE (latency concerns)")
        else:
            print(f"   Status: ✅ PASSED")

        print("=" * 80 + "\n")


class StressTestWorker:
    """A single concurrent worker that hammers the server with chaos."""

    def __init__(
        self,
        worker_id: int,
        loops: int = 10,
        metrics: Optional[StressTestMetrics] = None,
    ) -> None:
        self.worker_id = worker_id
        self.loops = loops
        self.metrics = metrics or StressTestMetrics()

    async def run(self) -> None:
        """Execute the worker's stress test loop."""
        for loop_num in range(self.loops):
            try:
                async with websockets.connect(SERVER_URL) as websocket:
                    self.metrics.total_connections += 1

                    # Step 1: Send autocomplete request
                    request_id = f"worker-{self.worker_id}-loop-{loop_num}-{uuid.uuid4().hex[:8]}"
                    autocomplete_payload = {
                        "request_id": request_id,
                        "action": "autocomplete",
                        "new_text": f"Worker {self.worker_id} autocomplete text",
                        "edit_char_index": 0,
                    }
                    await websocket.send(json.dumps(autocomplete_payload))
                    self.metrics.total_autocompletes_sent += 1

                    # Step 2: Random wait (10-100ms) to let generation start
                    await asyncio.sleep(random.uniform(0.01, 0.1))

                    # Step 3: Send edit to interrupt
                    edit_request_id = f"worker-{self.worker_id}-loop-{loop_num}-edit-{uuid.uuid4().hex[:8]}"
                    edit_payload = {
                        "request_id": edit_request_id,
                        "action": "edit",
                        "new_text": f"Worker {self.worker_id} edited text",
                        "edit_char_index": 5,
                    }
                    
                    # Track latency for edit acknowledgment
                    edit_sent_at = time.perf_counter()
                    await websocket.send(json.dumps(edit_payload))
                    self.metrics.total_edits_sent += 1

                    # Step 4: Decide whether to disconnect abruptly (20% chance)
                    should_disconnect = random.random() < 0.2
                    if should_disconnect:
                        await websocket.close()
                        self.metrics.total_disconnects_caused += 1
                        continue

                    # Step 5: Listen for responses
                    try:
                        response_count = 0
                        while response_count < 3:  # Expect up to 3 responses
                            msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                            response = json.loads(msg)

                            # Measure latency from edit send to first response
                            if response_count == 0:
                                latency_ms = (time.perf_counter() - edit_sent_at) * 1000
                                self.metrics.edit_latencies_ms.append(latency_ms)

                            if response["type"] == "cancelled":
                                self.metrics.total_cancelled_responses += 1
                            elif response["type"] == "done":
                                self.metrics.total_done_responses += 1
                            elif response["type"] in ("server_error", "error"):
                                self.metrics.total_errors += 1
                                print(f"Worker {self.worker_id}: {response}")
                            else:
                                self.metrics.malformed_responses += 1

                            response_count += 1
                    except asyncio.TimeoutError:
                        # Timeout waiting for responses is OK in stress test
                        pass
                    except json.JSONDecodeError:
                        self.metrics.malformed_responses += 1

            except ConnectionRefusedError:
                self.metrics.connection_failures += 1
            except websockets.exceptions.WebSocketException as exc:
                self.metrics.total_errors += 1
            except Exception as exc:
                self.metrics.total_errors += 1
                print(f"Worker {self.worker_id}: Unexpected error: {type(exc).__name__}: {exc}")


async def run_stress_test(num_workers: int = 50, loops_per_worker: int = 10) -> StressTestMetrics:
    """Launch all workers concurrently and return aggregated metrics."""
    print(f"\n🚀 Starting Phase 3 PDC Stress Test")
    print(f"   Workers: {num_workers}")
    print(f"   Loops per worker: {loops_per_worker}")
    print(f"   Server URL: {SERVER_URL}")
    print(f"   Total requests: {num_workers * loops_per_worker} loops")
    print(f"\n⏳ Spawning {num_workers} concurrent workers...")

    metrics = StressTestMetrics()

    # Create all workers
    workers = [
        StressTestWorker(worker_id=i, loops=loops_per_worker, metrics=metrics)
        for i in range(num_workers)
    ]

    # Launch all workers simultaneously
    start_time = time.perf_counter()
    try:
        await asyncio.gather(*[worker.run() for worker in workers], return_exceptions=True)
    except KeyboardInterrupt:
        print("\n⚠️  Stress test interrupted by user")

    elapsed = time.perf_counter() - start_time

    # Compute throughput
    print(f"\n✅ All workers completed in {elapsed:.2f} seconds")
    print(f"   Throughput: {metrics.total_connections / elapsed:.1f} connections/sec")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 Concurrent PDC Stress Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stress_test.py                    # Default: 50 workers, 10 loops
    python stress_test.py --workers 100      # Aggressive: 100 workers
    python stress_test.py --workers 25 --loops 5  # Lightweight: 25 workers, 5 loops
        """,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of concurrent workers (default: 50)",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=10,
        help="Iterations per worker (default: 10)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=SERVER_URL,
        help="WebSocket server URL (default: ws://127.0.0.1:8000/ws/editor)",
    )

    args = parser.parse_args()

    # Override global if provided
    globals()["SERVER_URL"] = args.server_url

    # Run stress test
    try:
        metrics = asyncio.run(run_stress_test(num_workers=args.workers, loops_per_worker=args.loops))
        metrics.report()
    except KeyboardInterrupt:
        print("\n\nStress test interrupted.")


if __name__ == "__main__":
    main()
