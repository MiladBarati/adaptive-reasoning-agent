"""
FastAPI Async Architecture Benchmark.

Measures latency, throughput, and event-loop responsiveness
for the RAG Agent API.  Run with:

    python -m benchmarks.api_benchmark --tag pre_async
    python -m benchmarks.api_benchmark --tag post_async
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = os.getenv("BENCHMARK_BASE_URL", "http://127.0.0.1:8000")
RESULTS_DIR = Path(__file__).parent / "results"

# Number of times each single-request latency test is repeated
LATENCY_ROUNDS = 5

# Number of concurrent requests for throughput / responsiveness tests
CONCURRENCY_LEVELS = [1, 5, 10]

# A lightweight query payload (mocked on the server side when benchmarking
# in isolation; otherwise exercises the full pipeline).
QUERY_PAYLOAD = {
    "question": "What is machine learning?",
    "max_iterations": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stats(times: list[float]) -> dict[str, float]:
    """Return p50 / p95 / p99 / mean for a list of durations (seconds)."""
    if not times:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "min": 0, "max": 0}
    s = sorted(times)
    n = len(s)
    return {
        "p50": s[int(n * 0.50)],
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "mean": statistics.mean(s),
        "min": min(s),
        "max": max(s),
    }


async def _timed_get(client: httpx.AsyncClient, url: str) -> float:
    start = time.perf_counter()
    resp = await client.get(url)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    return elapsed


async def _timed_post(client: httpx.AsyncClient, url: str, payload: dict) -> float:
    start = time.perf_counter()
    resp = await client.post(url, json=payload)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    return elapsed


# ---------------------------------------------------------------------------
# Benchmark sections
# ---------------------------------------------------------------------------


async def bench_single_request_latency(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    """Measure latency of individual endpoints sequentially."""
    results: dict[str, Any] = {}

    # /health
    times = [await _timed_get(client, f"{base_url}/health") for _ in range(LATENCY_ROUNDS)]
    results["health"] = _stats(times)

    # /stats
    times = [await _timed_get(client, f"{base_url}/stats") for _ in range(LATENCY_ROUNDS)]
    results["stats"] = _stats(times)

    # /query  (may be slow if it actually calls the LLM)
    times = []
    for _ in range(LATENCY_ROUNDS):
        try:
            t = await _timed_post(client, f"{base_url}/query", QUERY_PAYLOAD)
            times.append(t)
        except httpx.HTTPStatusError:
            # If query fails (no docs, no API key, etc.) record -1
            times.append(-1)
    results["query"] = (
        _stats([t for t in times if t >= 0])
        if any(t >= 0 for t in times)
        else {"error": "all requests failed"}
    )

    return results


async def bench_concurrent_throughput(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    """Fire N concurrent /health requests and measure total throughput."""
    results: dict[str, Any] = {}

    for n in CONCURRENCY_LEVELS:
        start = time.perf_counter()
        tasks = [_timed_get(client, f"{base_url}/health") for _ in range(n)]
        individual_times = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - start
        results[f"health_concurrent_{n}"] = {
            "wall_time_s": round(wall_time, 4),
            "requests_per_second": round(n / wall_time, 2),
            "individual": _stats(list(individual_times)),
        }

    return results


async def bench_event_loop_responsiveness(
    client: httpx.AsyncClient, base_url: str
) -> dict[str, Any]:
    """
    Start a /query request (slow) and simultaneously ping /health (fast).
    If the event loop is blocked, /health latency will spike.
    """
    results: dict[str, Any] = {}

    # Baseline health latency (no load)
    baseline_times = [await _timed_get(client, f"{base_url}/health") for _ in range(3)]
    baseline = _stats(baseline_times)
    results["health_baseline"] = baseline

    # Health latency WHILE a /query is in-flight
    async def _query_background():
        try:
            await client.post(f"{base_url}/query", json=QUERY_PAYLOAD, timeout=120.0)
        except Exception:
            pass  # we only care about health timing

    query_task = asyncio.create_task(_query_background())
    # Give the query a moment to start processing
    await asyncio.sleep(0.05)
    under_load_times = [await _timed_get(client, f"{base_url}/health") for _ in range(3)]
    await query_task  # wait for it to finish

    under_load = _stats(under_load_times)
    results["health_under_query_load"] = under_load
    results["responsiveness_ratio"] = (
        round(under_load["mean"] / baseline["mean"], 2) if baseline["mean"] > 0 else None
    )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_benchmark(tag: str, base_url: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  FastAPI Benchmark  —  tag: {tag}")
    print(f"  Target: {base_url}")
    print(f"{'=' * 60}\n")

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Quick smoke test
        try:
            r = await client.get(f"{base_url}/health")
            r.raise_for_status()
        except Exception as exc:
            print(f"ERROR: Cannot reach {base_url}/health — {exc}")
            print("Is the server running?  Start with:  uvicorn src.api.main:app")
            return

        print("[1/3] Single-request latency …")
        latency = await bench_single_request_latency(client, base_url)
        _print_section("Single-request latency", latency)

        print("[2/3] Concurrent throughput …")
        throughput = await bench_concurrent_throughput(client, base_url)
        _print_section("Concurrent throughput", throughput)

        print("[3/3] Event-loop responsiveness …")
        responsiveness = await bench_event_loop_responsiveness(client, base_url)
        _print_section("Event-loop responsiveness", responsiveness)

    report = {
        "tag": tag,
        "timestamp": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "latency_rounds": LATENCY_ROUNDS,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "results": {
            "single_request_latency": latency,
            "concurrent_throughput": throughput,
            "event_loop_responsiveness": responsiveness,
        },
    }

    out_path = RESULTS_DIR / f"{tag}_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅  Results saved to {out_path}\n")


def _print_section(title: str, data: dict[str, Any]) -> None:
    print(f"\n  --- {title} ---")
    for key, val in data.items():
        if isinstance(val, dict):
            print(f"    {key}:")
            for k2, v2 in val.items():
                print(f"      {k2}: {v2}")
        else:
            print(f"    {key}: {val}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="FastAPI Async Benchmark")
    parser.add_argument(
        "--tag",
        required=True,
        help="Label for this run, e.g. 'pre_async' or 'post_async'",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"Server URL (default: {DEFAULT_BASE_URL})",
    )
    args = parser.parse_args()

    base_url = args.base_url or DEFAULT_BASE_URL
    asyncio.run(run_benchmark(args.tag, base_url))


if __name__ == "__main__":
    main()
