# Corrective & Adaptive RAG Agents

A sophisticated Retrieval-Augmented Generation (RAG) system with adaptive corrective mechanisms for higher factual accuracy. Built with LangChain, LangGraph, and Groq LLM.

## Features

- **Query Rewriting**: Automatically reformulates queries for better retrieval
- **Retrieval Grading**: Evaluates document relevance before generation
- **Self-Reflection**: Checks for hallucinations and factual errors
- **Answer Verification**: Validates answers against sources
- **Iterative Refinement**: Loops back when quality is insufficient
- **Web Search Fallback**: Uses Tavily search when local documents are insufficient

## Architecture

The system uses a LangGraph state machine with multiple corrective loops:

1. Query Rewriting → Retrieval
2. Retrieval → Relevance Grading
3. Relevance Grading → Generation (or Web Search if all irrelevant)
4. Generation → Hallucination Check
5. Hallucination Check → Answer Verification
6. Answer Verification → End or Loop Back (if needed)

## Installation

### Prerequisites

- Python 3.11 or higher (required by `pyproject.toml`)
- pip (Python package manager)
- API Keys (see below)

### Setup Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd "Adaptive Reasoning Agent"
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Alternatively, if using `uv` or modern Python packaging:
```bash
pip install -e .
```

4. Set up environment variables:
Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

Then edit `.env` with your API keys:
```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=your_project_name
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_project_name
CORS_ORIGINS=http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000
```

5. Load sample data (optional but recommended):
```bash
python setup_sample_data.py
```

This will ingest sample documents about machine learning, deep learning, NLP, RAG systems, and vector databases into the vector store.

## Usage

### Web Interface (Gradio)

Run the Gradio web interface:
```bash
python -m src.ui.gradio_app
```

Then open your browser to `http://localhost:7860`

### API Server (FastAPI)

Run the FastAPI server:
```bash
uvicorn src.api.main:app --reload --port 8000
```

API endpoints:
- `GET /` - Root endpoint with API information
- `GET /health` - Health check
- `POST /query` - Submit a query to the RAG agent
- `POST /query/stream` - Submit a query with streaming response (Server-Sent Events)
- `POST /ingest/text` - Ingest text documents into the vector store
- `POST /ingest/files` - Upload and ingest files
- `GET /stats` - Get vector store statistics
- `DELETE /clear` - Clear all documents from the vector store

Visit `http://localhost:8000/docs` for interactive API documentation.

### Programmatic Usage

```python
from src.agents.rag_graph import create_rag_graph

# Create the RAG agent
agent = create_rag_graph()

# Query the agent
result = agent.invoke({
    "question": "What is machine learning?",
    "iterations": 0,
    "max_iterations": 3
})

print(result["generation"])
```

## Project Structure

```
├── src/
│   ├── agents/          # LangGraph workflow and nodes
│   │   ├── state.py     # State management
│   │   ├── nodes.py     # Agent nodes (rewrite, retrieve, grade, etc.)
│   │   └── rag_graph.py # Graph composition and query function
│   ├── core/            # Core RAG components
│   │   ├── embeddings.py      # Embedding models
│   │   ├── vector_store.py    # ChromaDB management
│   │   ├── retriever.py       # Advanced retrieval
│   │   └── logging_config.py  # Logging configuration
│   ├── corrective/      # Corrective mechanisms
│   │   ├── query_rewriter.py       # Query rewriting
│   │   ├── relevance_grader.py     # Document relevance grading
│   │   ├── hallucination_checker.py # Hallucination detection
│   │   └── answer_verifier.py       # Answer verification
│   ├── api/             # FastAPI backend
│   │   └── main.py      # API endpoints
│   └── ui/              # Gradio interface
│       └── gradio_app.py # Web UI
├── tests/               # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   ├── conftest.py      # Shared fixtures
│   └── README.md        # Testing documentation
├── data/
│   ├── documents/       # Sample documents
│   └── examples/        # Example queries
├── chroma_db/           # Vector store persistence (created at runtime)
├── htmlcov/             # Coverage reports (generated)
├── main.py              # Entry point
├── setup_sample_data.py # Sample data ingestion script
├── test_agent.py        # Basic test suite
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Modern Python project configuration
├── pytest.ini          # Pytest configuration
├── README.md            # This file
├── QUICKSTART.md        # Quick start guide
└── PROJECT_SUMMARY.md   # Detailed project summary
```

## Configuration

### Default Settings

- **LLM Model**: `llama-3.3-70b-versatile` (Groq)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Max Iterations**: 3
- **Top-K Retrieval**: 4 documents
- **Vector Store**: ChromaDB (persisted in `./chroma_db`)

### Python Version

This project requires **Python 3.11 or higher** (as specified in `pyproject.toml`).

### Environment Variables

Required API keys (see `.env.example` for full list):
- `GROQ_API_KEY` - Required for LLM inference
- `TAVILY_API_KEY` - Required for web search fallback
- `LANGCHAIN_API_KEY` - Optional, for LangChain tracing
- `LANGSMITH_API_KEY` - Optional, for LangSmith observability
- `CORS_ORIGINS` - Optional, comma-separated list of allowed CORS origins

## Testing

The project includes comprehensive unit and integration tests using pytest with coverage reporting.

### Running Tests

Run all tests:
```bash
pytest
```

Run only unit tests:
```bash
pytest tests/unit/
# or
pytest -m unit
```

Run only integration tests:
```bash
pytest tests/integration/
# or
pytest -m integration
```

Run tests with coverage:
```bash
pytest --cov=src --cov-report=html
```

View coverage report:
```bash
# HTML report opens automatically, or open htmlcov/index.html
```

Run specific test file:
```bash
pytest tests/unit/test_embeddings.py
```

Skip slow tests:
```bash
pytest -m "not slow"
```

Skip tests requiring API keys:
```bash
pytest -m "not requires_api"
```

### Test Structure

- **Unit Tests** (`tests/unit/`): Test individual components in isolation with mocks
  - Fast execution
  - No external dependencies
  - Marked with `@pytest.mark.unit`

- **Integration Tests** (`tests/integration/`): Test component interactions and full workflows
  - May require API keys
  - Slower execution
  - Marked with `@pytest.mark.integration`

### Coverage

The project maintains a minimum coverage threshold of 60% (configured in `pytest.ini`). Coverage reports are generated in multiple formats:
- Terminal output (`--cov-report=term-missing`)
- HTML report (`htmlcov/index.html`)
- XML report (`coverage.xml`)

### Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_api` - Tests requiring API keys

See `tests/README.md` for detailed testing documentation.

## Examples

### Sample Documents

The project includes 5 sample documents in `data/documents/`:
- `machine_learning.txt` - Introduction to machine learning
- `deep_learning.txt` - Deep learning fundamentals
- `nlp_basics.txt` - Natural language processing basics
- `rag_systems.txt` - RAG system architecture
- `vector_databases.txt` - Vector database concepts

### Example Queries

See `data/examples/example_queries.txt` for a comprehensive list of test queries categorized by feature:
- Basic queries
- Query rewriting tests
- Relevance grading tests
- Web search fallback tests
- Multi-step reasoning tests

### Quick Test

After running `setup_sample_data.py`, try:
```python
from src.agents.rag_graph import query_rag_agent

result = query_rag_agent(
    question="What is machine learning?",
    max_iterations=3
)

print(result["generation"])
print("\nWorkflow Steps:")
for step in result["workflow_steps"]:
    print(f"  - {step}")
```

## Development

### Code Quality

The project uses:
- **Ruff** - Fast Python linter and formatter
- **Black** - Code formatter (configured in `pyproject.toml`)
- **pytest** - Testing framework with coverage

Run linting:
```bash
ruff check .
```

Format code:
```bash
black .
```

### Project Configuration

- `pyproject.toml` - Modern Python project configuration with dependencies and tool settings
- `pytest.ini` - Pytest configuration with coverage settings
- `.env.example` - Template for environment variables

## Additional Resources

- **QUICKSTART.md** - Step-by-step setup guide
- **PROJECT_SUMMARY.md** - Detailed project overview and implementation details
- **tests/README.md** - Comprehensive testing documentation

## License

MIT License

