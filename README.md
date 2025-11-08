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

4. Set up environment variables:
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=your_project_name
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=your_project_name
```

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
- `POST /query` - Submit a query to the RAG agent
- `POST /ingest` - Upload and ingest documents
- `GET /health` - Health check
- `GET /stats` - Retrieval statistics

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
│   ├── core/            # Vector store, embeddings, retriever
│   ├── corrective/      # Corrective mechanisms
│   ├── api/             # FastAPI backend
│   └── ui/              # Gradio interface
├── data/
│   └── documents/       # Sample documents
├── requirements.txt
└── README.md
```

## Configuration

Default settings:
- **LLM Model**: `llama-3.3-70b-versatile` (Groq)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Max Iterations**: 3
- **Top-K Retrieval**: 4 documents

## Testing

The project includes comprehensive unit and integration tests using pytest.

### Running Tests

Run all tests:
```bash
pytest
```

Run only unit tests:
```bash
pytest tests/unit/
```

Run only integration tests:
```bash
pytest tests/integration/
```

Run tests with coverage:
```bash
pytest --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/unit/test_embeddings.py
```

Skip slow tests:
```bash
pytest -m "not slow"
```

### Test Structure

- **Unit Tests** (`tests/unit/`): Test individual components in isolation with mocks
- **Integration Tests** (`tests/integration/`): Test component interactions and full workflows

See `tests/README.md` for detailed testing documentation.

## Examples

See the `data/documents/` directory for sample documents and example queries.

## License

MIT License

