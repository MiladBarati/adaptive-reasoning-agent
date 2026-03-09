# Load Testing Benchmark Results

Benchmark measuring the API performance, stability, and infrastructure resilience under high concurrent load using **Locust**, taking advantage of the `MOCK_LLM=true` flag.

**Date:** 2026-03-09
**Tool:** `load_tests/locustfile.py`
**Server:** Uvicorn (single worker) on `0.0.0.0:8000`

---

## Benchmark Strategy

Running load tests against expensive third-party LLMs and search APIs at high concurrency (50+ users) is cost-prohibitive and will inevitably hit rate limits (429 Too Many Requests), resulting in tests that measure the provider's limits rather than our infrastructure's capabilities.

To solve this, we measure:
1. **Infrastructure Capacity (Mocked):** Using `MOCK_LLM=true` to simulate exactly 1.0s of LLM/Search latency without external network calls. This tests FastAPI, ChromaDB, and Python's `asyncio` event loop under stress.
2. **API Resiliency (Real):** Low concurrency (2-5 users) against real APIs to verify timeout and rate limit handling.

---

## Infrastructure Capacity (Mocked LLM)

### Test Parameters
* **Users:** 50
* **Spawn Rate:** 5 users/second
* **Duration:** 1 minute
* **Environment:** `MOCK_LLM=true`

### Results (Mocked Run: 1 Minute, 50 Users)

| Endpoint | Requests | Failures | Median (ms) | 95%ile (ms) | Req/s |
|----------|----------|----------|-------------|-------------|-------|
| `POST /query` | 571 | 0 (0%) | 52 | 100 | 9.54 |
| `GET /health` | 179 | 0 (0%) | 1 | 4 | 2.99 |
| `GET /stats`  | 202 | 0 (0%) | 1 | 6 | 3.38 |

> **Analysis:** With `MOCK_LLM=true`, the initial uncached `/query` response times are around ~1100ms due to our simulated 1.0s wait in `SlowFakeLLM`. However, Locust randomly selected from 9 predefined questions. This load test **perfectly proved the effectiveness of the semantic cache**: after the first 9 permutations were cached, the latency for `POST /query` dropped from 1100ms down to a median of **52ms**. The `GET /health` endpoint stayed at a steady 1ms median, proving that the underlying asyncio architecture successfully handled concurrency without the event loop being blocked by PyTorch (SentenceTransformers) or heavy LLM usage.

---

## API Resiliency (Real LLM)

### Test Parameters
* **Users:** 5
* **Spawn Rate:** 1 user/second
* **Duration:** 1 minute
* **Environment:** `MOCK_LLM=false`

### Results (Placeholder - Run to populate)

| Endpoint | Requests | Failures (429s/500s) | Median (ms) | 95%ile (ms) |
|----------|----------|----------------------|-------------|-------------|
| `POST /query` | TBD | TBD | TBD | TBD |

> **Analysis:** Here, we are monitoring for successful recovery or graceful degradation when hitting potential API rate limits from Ollama/Groq or Tavily.

---

## How to Run the Benchmarks

To reproduce these benchmarks, make sure you have `locust` installed via `uv`:

```bash
uv add locust
```

**Run Mocked Load Test (Terminal 1 - Server):**
```bash
MOCK_LLM=true uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Run Mocked Load Test (Terminal 2 - Locust):**
```bash
uv run locust -f load_tests/locustfile.py --headless -u 50 -r 5 -t 1m --host http://localhost:8000
```
