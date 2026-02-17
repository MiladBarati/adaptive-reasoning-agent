# How Our Observability System Works (Simple Explanation)

> **A beginner-friendly guide to understanding monitoring and observability in the Corrective RAG Agent**

---

## Table of Contents

1. [What is Observability?](#what-is-observability)
2. [The Problem It Solves](#the-problem-it-solves)
3. [The Three Pillars of Observability](#the-three-pillars-of-observability)
4. [Our Observability Stack](#our-observability-stack)
5. [How It Works (Step-by-Step)](#how-it-works-step-by-step)
6. [What We Track](#what-we-track)
7. [Real-World Example](#real-world-example)
8. [Setting Up Observability](#setting-up-observability)
9. [Technical Summary](#technical-summary)

---

## What is Observability?

**Observability** is like having **X-ray vision for your application**. It lets you see what's happening inside your system while it's running.

### Simple Analogy

Imagine you're a doctor:
- **Without observability:** Patient says "I don't feel good" (you have no clue what's wrong)
- **With observability:** You have X-rays, blood tests, heart rate monitor (you can diagnose the problem!)

### For Our RAG Agent

- **Without observability:** "The app is slow" (no idea why)
- **With observability:** "Cache hit rate is 20%, retrieval takes 8s, LLM calls use 15K tokens" (specific insights!)

---

## The Problem It Solves

### Problem 1: **"Why is it slow?"**

Your app takes 15 seconds to answer a question. But why?
- Is it the document retrieval?
- The LLM generation?
- The database?
- Network issues?

**Solution:** Track timing for each step and find the bottleneck!

### Problem 2: **"Did it crash? When? Why?"**

Your app stops working at 3 AM. Questions:
- What was the last thing it did?
- Which part failed?
- What error occurred?

**Solution:** Trace every request and log all errors!

### Problem 3: **"How is it performing?"**

You want to know:
- How many requests per minute?
- Cache hit rate?
- Token usage patterns?
- Which queries are slowest?

**Solution:** Collect metrics over time and visualize trends!

---

## The Three Pillars of Observability

Our system uses the **three pillars** of observability:

### 1. **Metrics** 📊 (The Numbers)

**What it is:** Numerical measurements over time

**Examples:**
- Cache hit rate: 75%
- Average response time: 3.2 seconds
- Tokens used per request: 1,247
- Requests per minute: 15

**Analogy:** Like a car dashboard (speed, RPM, fuel level)

**In our system:** Prometheus collects these numbers

### 2. **Traces** 🔍 (The Journey)

**What it is:** The complete journey of a single request through the system

**Example:**
```
Request 1234 took 12.5 seconds:
  ├─ Cache check: 0.1s
  ├─ Query rewrite: 2.3s
  ├─ Document retrieval: 1.8s
  ├─ Relevance grading: 3.2s
  ├─ Answer generation: 4.5s
  └─ Hallucination check: 0.6s
```

**Analogy:** Like GPS tracking showing every turn on your route

**In our system:** OpenTelemetry creates these traces

### 3. **Logs** 📝 (The Details)

**What it is:** Text records of what happened

**Examples:**
```
[INFO] Query rewritten from "What is ML?" to "Define machine learning..."
[WARNING] Hallucination detected, retrying...
[ERROR] Failed to connect to vector store
```

**Analogy:** Like a diary of everything the app does

**In our system:** Python logging module (with structured output)

---

## Our Observability Stack

We use **three open-source tools** working together:

```
┌─────────────────┐
│   RAG Agent     │  ← Your application
│  (Port 8000)    │
└────────┬────────┘
         │ Exposes /metrics
         ↓
┌─────────────────┐
│   Prometheus    │  ← Collects metrics
│  (Port 9090)    │    Every 5 seconds
└────────┬────────┘
         │ Stores time-series data
         ↓
┌─────────────────┐
│    Grafana      │  ← Visualizes data
│  (Port 3000)    │    Pretty dashboards!
└─────────────────┘
```

### Component Breakdown

| Tool | What It Does | URL | Purpose |
|------|-------------|-----|---------|
| **OpenTelemetry** | Instruments code (adds tracking) | N/A | Creates traces & metrics |
| **Prometheus** | Scrapes & stores metrics | http://localhost:9090 | Time-series database |
| **Grafana** | Visualizes metrics | http://localhost:3000 | Beautiful dashboards |

---

## How It Works (Step-by-Step)

### Step 1: **Instrumentation** (Adding Tracking Code)

**What happens:** We add tracking code to important functions.

**Example:**
```python
# Without observability
def retrieve_documents(query):
    docs = vector_store.search(query)
    return docs

# With observability
def retrieve_documents(query):
    with tracer.start_as_current_span("rag.retrieve_documents") as span:
        span.set_attribute("query", query)
        docs = vector_store.search(query)
        span.set_attribute("doc_count", len(docs))
    return docs
```

**What we added:**
- `start_as_current_span()` - Marks the start and end of this operation
- `set_attribute()` - Records important details

**Result:** Every time this function runs, we track:
- How long it took
- What the query was
- How many docs were found

---

### Step 2: **Metrics Collection** (Counting Things)

**What happens:** We count important events.

**Example:**
```python
# Create a counter
cache_hits_counter = meter.create_counter(
    "rag.cache.hits",
    description="Number of cache hits",
    unit="1"
)

# Increment when cache hit occurs
if cache_result:
    cache_hits_counter.add(1)
```

**What this does:**
- Every cache hit increases the counter by 1
- Prometheus scrapes this every 5 seconds
- We can graph cache hits over time

---

### Step 3: **Prometheus Scraping** (Collecting Data)

**What happens:** Prometheus regularly asks our app for metrics.

**The process:**
```
Every 5 seconds:
  Prometheus → GET http://localhost:8000/metrics
  RAG Agent  → Returns current metric values
  Prometheus → Stores in time-series database
```

**Example response:**
```
# TYPE rag_cache_hits_total counter
rag_cache_hits_total 42

# TYPE rag_cache_misses_total counter
rag_cache_misses_total 158

# TYPE rag_llm_tokens_total counter
rag_llm_tokens_total{type="input",model="qwen2.5:14b"} 125478
rag_llm_tokens_total{type="output",model="qwen2.5:14b"} 34562
```

---

### Step 4: **Grafana Visualization** (Making It Pretty)

**What happens:** Grafana queries Prometheus and creates dashboards.

**Example dashboard:**
```
┌─────────────────────────────────────────┐
│   RAG Agent Performance Dashboard       │
├─────────────────────────────────────────┤
│                                         │
│  Cache Hit Rate:        75% ↗          │
│  Avg Response Time:     3.2s ↘         │
│  Requests/min:          15 →           │
│  Token Usage (1h):      247K ↗         │
│                                         │
│  ┌─── Response Time Graph ──────┐     │
│  │     /\                         │     │
│  │    /  \    /\                 │     │
│  │   /    \__/  \___             │     │
│  └────────────────────────────────┘     │
└─────────────────────────────────────────┘
```

**You can:**
- See real-time metrics
- Create custom graphs
- Set up alerts (e.g., "Notify me if response time > 10s")

---

## What We Track

### 1. **Cache Performance**

**What we measure:**
```python
cache_hits_counter = meter.create_counter("rag.cache.hits")
cache_misses_counter = meter.create_counter("rag.cache.misses")
```

**Why it matters:**
- High cache hit rate = Fast responses (reusing answers)
- Low cache hit rate = Slow responses (generating new answers)

**Example insights:**
- "Cache hit rate dropped from 80% to 20%" → Need better caching strategy
- "95% cache hits on weekdays" → Users ask similar questions

---

### 2. **Token Usage**

**What we measure:**
```python
token_usage_counter = meter.create_counter(
    "rag.llm.tokens",
    description="Number of tokens used by LLM"
)

# Track input and output separately
token_usage_counter.add(input_tokens, {"type": "input", "model": "qwen2.5:14b"})
token_usage_counter.add(output_tokens, {"type": "output", "model": "qwen2.5:14b"})
```

**Why it matters:**
- Token usage correlates with cost (if using paid APIs)
- Shows which operations are expensive

**Example insights:**
- "Grading uses 50K tokens/hour" → Optimize prompts
- "Output tokens > input tokens" → LLM generating verbose answers

---

### 3. **Workflow Traces**

**What we track:**
Every step in the RAG pipeline gets a span:

```python
# All tracked operations
with tracer.start_as_current_span("rag.check_cache") as span: ...
with tracer.start_as_current_span("rag.rewrite_query") as span: ...
with tracer.start_as_current_span("rag.retrieve_documents") as span: ...
with tracer.start_as_current_span("rag.grade_documents") as span: ...
with tracer.start_as_current_span("rag.generate_answer") as span: ...
with tracer.start_as_current_span("rag.check_hallucination") as span: ...
with tracer.start_as_current_span("rag.verify_answer") as span: ...
```

**Why it matters:**
- See exactly where time is spent
- Identify bottlenecks
- Debug slow requests

**Example insights:**
- "Grading takes 40% of total time" → Parallelize grading
- "Cache check is instant, but retrieval is slow" → Optimize vector search

---

### 4. **Request Attributes**

**What we track:**
```python
span.set_attribute("cache.query", query[:200])
span.set_attribute("cache.hit", True)
span.set_attribute("cache.similarity_score", 0.94)
span.set_attribute("retriever.k", 4)
span.set_attribute("retriever.doc_count", 3)
span.set_attribute("rag.question", question[:200])
span.set_attribute("rag.iteration", 2)
```

**Why it matters:**
- Understand patterns in queries
- Debug specific requests
- Analyze user behavior

---

## Real-World Example

Let's trace a complete request through the observability system:

### User Query
> "How does machine learning work?"

### What Gets Tracked

#### **Prometheus Metrics** (Before the request)
```
rag_cache_hits_total: 42
rag_cache_misses_total: 158
rag_llm_tokens_total{type="input"}: 125478
rag_llm_tokens_total{type="output"}: 34562
```

---

#### **Trace Timeline** (During the request)

```
Trace ID: abc123def456
Total Duration: 12.5 seconds

├─ [0.00s - 0.08s] rag.check_cache (0.08s)
│  ├─ cache.query: "How does machine learning work?"
│  ├─ cache.hit: false
│  └─ cache.similarity_score: 0.72 (below 0.90 threshold)
│
├─ [0.08s - 2.35s] rag.rewrite_query (2.27s)
│  ├─ rag.question: "How does machine learning work?"
│  └─ tokens_used: 487 input + 64 output
│
├─ [2.35s - 4.12s] rag.retrieve_documents (1.77s)
│  ├─ retriever.query: "Define machine learning..."
│  ├─ retriever.k: 4
│  └─ retriever.doc_count: 4
│
├─ [4.12s - 7.34s] rag.grade_documents (3.22s)
│  ├─ documents_retrieved: 4
│  ├─ documents_relevant: 3
│  └─ tokens_used: 1,234 input + 12 output
│
├─ [7.34s - 11.89s] rag.generate_answer (4.55s)
│  ├─ rag.context_doc_count: 3
│  └─ tokens_used: 2,145 input + 127 output
│
├─ [11.89s - 12.15s] rag.check_hallucination (0.26s)
│  ├─ rag.is_grounded: true
│  └─ tokens_used: 2,272 input + 8 output
│
└─ [12.15s - 12.50s] rag.verify_answer (0.35s)
   ├─ rag.is_answer_good: true
   └─ tokens_used: 2,399 input + 6 output
```

---

#### **Prometheus Metrics** (After the request)
```
rag_cache_hits_total: 42          (no change - cache miss)
rag_cache_misses_total: 159       (+1)
rag_llm_tokens_total{type="input"}: 134015  (+8,537)
rag_llm_tokens_total{type="output"}: 34779  (+217)
```

---

### **Insights from This Trace**

✅ **Bottleneck identified:** Grading (3.22s) and generation (4.55s) are the slowest  
✅ **Cache performance:** 0.72 similarity wasn't enough (threshold: 0.90)  
✅ **Token efficiency:** Used 8,754 tokens total (reasonable for complex query)  
✅ **Quality checks:** Both hallucination and verification passed  

**Optimization ideas:**
- Parallelize document grading → Save ~2.5s
- Adjust cache threshold to 0.70 → More cache hits
- Use smaller model for grading → Reduce token usage

---

## Setting Up Observability

### Option 1: **Disabled** (Default - Zero Overhead)

**How to use:**
```bash
# Don't set OTEL_ENABLED, or set it to "false"
# No tracing, only basic logging
python -m uvicorn src.api.main:app --reload
```

**When to use:**
- Local development
- Don't need monitoring
- Want maximum performance

---

### Option 2: **Enabled with Metrics Only**

**How to use:**
```bash
# In .env file
OTEL_ENABLED=true

# Start the app
python -m uvicorn src.api.main:app --reload

# Prometheus scrapes http://localhost:8000/metrics
# You can view metrics manually at that URL
```

**What you get:**
- Prometheus metrics available at `/metrics`
- No visualization (unless you run Prometheus + Grafana separately)

---

### Option 3: **Full Stack** (Recommended)

**How to use:**
```bash
# In .env file
OTEL_ENABLED=true

# Start observability stack (Prometheus + Grafana)
docker-compose up -d

# Start the app
python -m uvicorn src.api.main:app --reload
```

**What you get:**
- ✅ Full tracing with OpenTelemetry
- ✅ Metrics collected by Prometheus
- ✅ Beautiful dashboards in Grafana
- ✅ Real-time monitoring

**Access points:**
- RAG Agent: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

---

## Technical Summary

### Architecture

```
┌──────────────────────────────────────────┐
│         RAG Agent (Python)               │
│  ┌────────────────────────────────────┐  │
│  │  Application Code                  │  │
│  │  ┌──────────┐  ┌──────────┐       │  │
│  │  │ Tracer   │  │  Meter   │       │  │
│  │  └────┬─────┘  └────┬─────┘       │  │
│  │       │             │              │  │
│  │       ↓             ↓              │  │
│  │  OpenTelemetry SDK                 │  │
│  │       │             │              │  │
│  │       ↓             ↓              │  │
│  │  Console    PrometheusExporter    │  │
│  │   Logs         /metrics            │  │
│  └────────────────────┬───────────────┘  │
└────────────────────────┼──────────────────┘
                         │
                         ↓ HTTP GET /metrics (every 5s)
                ┌────────────────┐
                │   Prometheus   │
                │  (Port 9090)   │
                │                │
                │  Stores data   │
                └────────┬───────┘
                         │
                         ↓ PromQL queries
                ┌────────────────┐
                │    Grafana     │
                │  (Port 3000)   │
                │                │
                │  Dashboards    │
                └────────────────┘
```

---

### Configuration Files

#### **1. OpenTelemetry Setup** (`src/core/telemetry.py`)
```python
def setup_telemetry():
    if os.getenv("OTEL_ENABLED", "false").lower() == "true":
        # Initialize tracer
        resource = Resource.create({"service.name": "rag-agent"})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        
        # Initialize Prometheus metrics
        reader = PrometheusMetricReader()
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[reader]
        )
        metrics.set_meter_provider(meter_provider)
```

#### **2. Prometheus Config** (`config/prometheus.yml`)
```yaml
global:
  scrape_interval: 15s  # Default interval

scrape_configs:
  - job_name: 'rag-agent'
    scrape_interval: 5s  # Check every 5 seconds
    static_configs:
      - targets: ['host.docker.internal:8000']  # Our app
```

#### **3. Docker Compose** (`docker-compose.yml`)
```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
```

---

### Metrics We Export

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|--------|
| `rag.cache.hits` | Counter | Cache hits count | None |
| `rag.cache.misses` | Counter | Cache misses count | None |
| `rag.llm.tokens` | Counter | Token usage | `type` (input/output), `model` |

---

### Spans We Track

| Span Name | What It Tracks | Key Attributes |
|-----------|----------------|----------------|
| `rag.check_cache` | Cache lookup | `cache.query`, `cache.hit`, `cache.similarity_score` |
| `rag.rewrite_query` | Query rewriting | `rag.question` |
| `rag.retrieve_documents` | Document retrieval | `retriever.query`, `retriever.k`, `retriever.doc_count` |
| `rag.grade_documents` | Relevance grading | `documents_retrieved`, `documents_relevant` |
| `rag.web_search` | Web search fallback | `web_search.query`, `web_search.results` |
| `rag.generate_answer` | Answer generation | `rag.question`, `rag.context_doc_count` |
| `rag.check_hallucination` | Hallucination detection | `rag.is_grounded` |
| `rag.verify_answer` | Answer verification | `rag.is_answer_good` |
| `rag.increment_iteration` | Iteration counter | `rag.iteration`, `rag.max_iterations` |
| `cache.check` | Low-level cache check | `cache.query`, `cache.hit` |
| `cache.update` | Cache update | `cache.query` |
| `retriever.retrieve` | Low-level retrieval | `retriever.query`, `retriever.multi_query` |

---

## Summary: Why Observability Matters

### Without Observability:
- ❌ "The app is slow" (no idea why)
- ❌ Debugging is guesswork
- ❌ Can't measure improvements
- ❌ No visibility into production issues

### With Observability:
- ✅ Know exactly where time is spent
- ✅ Track performance over time
- ✅ Identify bottlenecks instantly
- ✅ Measure impact of optimizations
- ✅ Debug production issues with traces
- ✅ Understand user patterns
- ✅ Optimize costs (token usage tracking)

**The observability stack gives you superpowers to understand, debug, and optimize your RAG agent!** 🚀

---

## Further Reading

- **Main documentation:** [HOW_IT_WORKS.md](./HOW_IT_WORKS.md)
- **Architecture:** [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Retriever explained:** [RETRIEVER_EXPLAINED.md](./RETRIEVER_EXPLAINED.md)

---

*Last Updated: February 2026*
