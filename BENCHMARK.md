# API Benchmark Results

Benchmark comparing the FastAPI backend **before** and **after** migrating to a proper async architecture using `asyncio.to_thread()`.

**Date:** 2026-02-11  
**Tool:** `benchmarks/api_benchmark.py`  
**Server:** Uvicorn (single worker) on `127.0.0.1:8000`

---

## Event-Loop Responsiveness ⭐

The most critical metric — measures `/health` latency while a long-running `/query` request is in-flight. If the event loop is blocked by synchronous code, `/health` will spike.

| Metric | Before (sync) | After (async) | Change |
|--------|---------------|---------------|--------|
| `/health` baseline (mean) | 1.44 ms | 1.48 ms | — |
| `/health` under `/query` load (mean) | **3,313 ms** | **2.13 ms** | **1,555× faster** |
| `/health` under load (p95) | 9,936 ms | 2.52 ms | **3,943× faster** |
| Responsiveness ratio | **2,305×** | **1.43×** | ✅ Unblocked |

> **Before:** A single `/query` request blocked the event loop for ~10–30 seconds, making `/health` completely unresponsive (3.3s mean, 9.9s p95).  
> **After:** `/health` stays at ~2ms even while `/query` is processing — the event loop is free.

---

## Single-Request Latency

Sequential measurements — one request at a time, 5 rounds each.

| Endpoint | Before (mean) | After (mean) | Before (p50) | After (p50) |
|----------|---------------|--------------|--------------|-------------|
| `GET /health` | 0.98 ms | 1.36 ms | 0.87 ms | 1.33 ms |
| `GET /stats` | 14.1 ms | 21.4 ms | 1.07 ms | 2.97 ms |
| `POST /query` | 17.9 s | 13.6 s | 11.2 s | 13.1 s |

> `/query` latency is dominated by external LLM API calls (Groq) and varies per run. The `asyncio.to_thread()` overhead is negligible.

---

## Concurrent Throughput

Concurrent `/health` requests measuring wall-clock time and throughput.

| Concurrency | Before (req/s) | After (req/s) | Before (wall time) | After (wall time) |
|-------------|----------------|---------------|---------------------|-------------------|
| 1 | 758 | 505 | 1.3 ms | 2.0 ms |
| 5 | 650 | 674 | 7.7 ms | 7.4 ms |
| 10 | 1,096 | 864 | 9.1 ms | 11.6 ms |

> Lightweight endpoint throughput is comparable. The real benefit is that under heavy load (concurrent `/query` + `/health`), the async version doesn't block.

---

## How to Reproduce

```bash
# Start the server
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Run benchmark
python -m benchmarks.api_benchmark --tag <label>

# Results saved to benchmarks/results/<label>_benchmark.json
```

---

## Raw Data

- [`benchmarks/results/pre_async_benchmark.json`](benchmarks/results/pre_async_benchmark.json)
- [`benchmarks/results/post_async_benchmark.json`](benchmarks/results/post_async_benchmark.json)
