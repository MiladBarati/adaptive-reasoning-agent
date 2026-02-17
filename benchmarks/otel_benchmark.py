"""
OpenTelemetry Overhead Benchmark.

Measures the latency impact of enabling OpenTelemetry tracing
on the RAG Agent API.  Run TWICE — once with OTEL_ENABLED=false
and once with OTEL_ENABLED=true — then compare the results.

Usage:
    1. Ensure OTEL_ENABLED=false in .env, (re)start the server.
       python -m benchmarks.otel_benchmark --tag pre_otel

    2. Set OTEL_ENABLED=true in .env, restart the server.
       python -m benchmarks.otel_benchmark --tag post_otel

Results are saved to benchmarks/results/<tag>_benchmark.json.
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

QUERY_ROUNDS = 5
HEALTH_ROUNDS = 10

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


async def bench_health(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    """Measure /health latency as a control (should be unaffected)."""
    times = [await _timed_get(client, f"{base_url}/health") for _ in range(HEALTH_ROUNDS)]
    return _stats(times)


async def bench_query(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    """Measure end-to-end /query latency."""
    times: list[float] = []
    for i in range(QUERY_ROUNDS):
        try:
            t = await _timed_post(client, f"{base_url}/query", QUERY_PAYLOAD)
            times.append(t)
            print(f"    Round {i + 1}/{QUERY_ROUNDS}: {t:.3f}s")
        except httpx.HTTPStatusError as exc:
            print(f"    Round {i + 1}/{QUERY_ROUNDS}: FAILED ({exc.response.status_code})")
            times.append(-1)

    valid = [t for t in times if t >= 0]
    if not valid:
        return {"error": "all requests failed"}
    return _stats(valid)


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def compare_results(pre_path: Path, post_path: Path) -> None:
    """Print a comparison table between two benchmark runs."""
    if not pre_path.exists() or not post_path.exists():
        print("\n⚠  Cannot compare — need both pre and post result files.")
        return

    with open(pre_path) as f:
        pre = json.load(f)
    with open(post_path) as f:
        post = json.load(f)

    print(f"\n{'=' * 60}")
    print("  OTel Overhead Comparison")
    print(f"{'=' * 60}")

    for section in ["health", "query"]:
        pre_data = pre["results"].get(section, {})
        post_data = post["results"].get(section, {})

        if "error" in pre_data or "error" in post_data:
            print(f"\n  {section}: skipped (errors)")
            continue

        pre_mean = pre_data.get("mean", 0)
        post_mean = post_data.get("mean", 0)

        if pre_mean > 0:
            overhead_pct = ((post_mean - pre_mean) / pre_mean) * 100
        else:
            overhead_pct = 0

        print(f"\n  --- {section.upper()} ---")
        print(
            f"    Pre-OTel  mean: {pre_mean:.4f}s "
            f"(p50={pre_data.get('p50', 0):.4f}s, p95={pre_data.get('p95', 0):.4f}s)"
        )
        print(
            f"    Post-OTel mean: {post_mean:.4f}s "
            f"(p50={post_data.get('p50', 0):.4f}s, p95={post_data.get('p95', 0):.4f}s)"
        )
        print(f"    Overhead:       {overhead_pct:+.2f}%")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_benchmark(tag: str, base_url: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  OTel Overhead Benchmark — tag: {tag}")
    print(f"  Target: {base_url}")
    print(f"{'=' * 60}\n")

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Smoke test
        try:
            r = await client.get(f"{base_url}/health")
            r.raise_for_status()
        except Exception as exc:
            print(f"ERROR: Cannot reach {base_url}/health — {exc}")
            print("Is the server running?")
            return

        print(f"[1/2] Health latency ({HEALTH_ROUNDS} rounds) …")
        health = await bench_health(client, base_url)
        print(f"       mean={health.get('mean', 0):.4f}s\n")

        print(f"[2/2] Query latency ({QUERY_ROUNDS} rounds) …")
        query = await bench_query(client, base_url)
        query_mean = query.get("mean", 0)
        if query_mean:
            print(f"       mean={query_mean:.3f}s\n")

    report = {
        "tag": tag,
        "timestamp": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "query_rounds": QUERY_ROUNDS,
        "health_rounds": HEALTH_ROUNDS,
        "results": {
            "health": health,
            "query": query,
        },
    }

    out_path = RESULTS_DIR / f"{tag}_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✅  Results saved to {out_path}")

    # Auto-compare if both files exist
    if tag == "post_otel":
        pre_path = RESULTS_DIR / "pre_otel_benchmark.json"
        compare_results(pre_path, out_path)
    elif tag == "pre_otel":
        post_path = RESULTS_DIR / "post_otel_benchmark.json"
        if post_path.exists():
            compare_results(out_path, post_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="OTel Overhead Benchmark")
    parser.add_argument(
        "--tag",
        required=True,
        help="Label for this run, e.g. 'pre_otel' or 'post_otel'",
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
