# Load Testing the Corrective RAG Agent

This directory contains load tests for the Corrective RAG Agent to ensure performance, stability, and resiliency.

We use **[Locust](https://locust.io/)** as our load testing framework.

## Testing Strategies

Load testing an LLM application requires a different approach than a standard web server due to expensive, rate-limited third-party APIs (like OpenAI, Anthropic, Ollama, Tavily, etc.). We recommend a two-tiered testing approach:

### Level 1: Infrastructure Load Testing (High Concurrency)
The goal here is to test your FastAPI server, connection pooling, Vector database (ChromaDB), and memory management under high load without exhausting rate limits or incurring huge costs.
We achieve this by setting the `MOCK_LLM=true` environment variable, which replaces the real LLM generation and Web Search with a simulated latency sleep and mock string generation.

**How to run (Mocked LLM):**
1. Start the server with the mock flag:
   ```bash
   MOCK_LLM=true uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
2. Run Locust to simulate 50+ concurrent users:
   ```bash
   # Run via web UI
   uv run locust -f load_tests/locustfile.py

   # OR run headlessly for 1 minute:
   uv run locust -f load_tests/locustfile.py --headless -u 50 -r 5 -t 1m --host http://localhost:8000
   ```

### Level 2: API Resiliency Testing (Low Concurrency)
The goal here is to interact with real LLMs and APIs to test your application's rate-limiting handling, timeout configurations, and real-world latency. Keep concurrency very low to avoid massive bills.

**How to run (Real LLM):**
1. Start your real server:
   ```bash
   uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
2. Run Locust with low concurrency (e.g., 2-5 users):
   ```bash
   # Make sure LLM/Tavily are configured properly in environment
   uv run locust -f load_tests/locustfile.py --headless -u 5 -r 1 -t 1m --host http://localhost:8000
   ```

## Endpoint Tests

Currently, the `locustfile.py` covers the following endpoints:
- `POST /query`: Simulates varying user queries to trigger the RAG workflow.
- `GET /health`: Healthcheck polling.
- `GET /stats`: Monitoring endpoints.
