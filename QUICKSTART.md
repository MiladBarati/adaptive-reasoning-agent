# Quick Start Guide

Get your Corrective & Adaptive RAG Agent up and running in minutes!

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- API Keys (already in your `.env` file):
  - Groq API Key
  - Tavily API Key (for web search)
  - LangChain API Key (optional, for tracing)
  - OTEL_ENABLED=true (optional, for OpenTelemetry tracing)

## Installation Steps

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all necessary packages including:
- LangChain and LangGraph for agent orchestration
- Groq for LLM inference
- ChromaDB for vector storage
- FastAPI for API backend
- Gradio for web interface
- Sentence Transformers for embeddings

### 3. Load Sample Data

```bash
python setup_sample_data.py
```

This script will:
- Initialize the ChromaDB vector store
- Ingest 5 sample documents (ML, DL, NLP, RAG, Vector DBs)
- Create searchable embeddings
- Display statistics

Expected output:
```
✓ Successfully ingested 5 files into ~100+ chunks
Total document chunks: 100+
```

## Running the Application

### Option 1: Web Interface (Recommended)

Launch the Gradio web interface:

```bash
python -m src.ui.gradio_app
```

Then open your browser to: **http://localhost:7860**

Features:
- 💬 **Query Tab**: Ask questions and see responses
- 📁 **Ingest Documents**: Upload your own documents
- 📊 **Statistics**: Monitor vector store stats
- ⚙️ **Settings**: Configure max iterations and parameters

### Option 2: API Server

Launch the FastAPI backend:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then open: **http://localhost:8000/docs** for API documentation

API Endpoints:
- `POST /query` - Query the RAG agent
- `POST /ingest/text` - Ingest text documents
- `POST /ingest/files` - Upload files
- `GET /stats` - Get vector store statistics
- `GET /health` - Health check

### Option 3: Python Code

Use the agent programmatically:

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

## Example Queries

Try these queries to see the corrective features in action:

### Basic Queries
```
What is machine learning?
Explain deep learning
What are vector databases used for?
```

### Testing Query Rewriting
```
How do computers learn from data?
What's the deal with those neural network things?
```

### Testing Relevance Grading
```
What is GPT-4?
How does BERT work?
What are convolutional neural networks?
```

### Testing Web Search Fallback
```
Who won the Nobel Prize in Physics in 2024?
What is the current price of Bitcoin?
```

### Testing Multi-Step Reasoning
```
Compare supervised and unsupervised learning
What makes transformer models better than RNNs for NLP?
How would vector databases improve a RAG system?
```

See `data/examples/example_queries.txt` for a comprehensive list with explanations.

## Understanding the Workflow

Watch the workflow steps in the UI to see:

1. **Query Rewriting**: Original question → Improved version
2. **Document Retrieval**: Fetching relevant chunks
3. **Relevance Grading**: X relevant, Y irrelevant
4. **Web Search** (if needed): Fallback to Tavily search
5. **Generation**: Creating the answer
6. **Hallucination Check**: Grounded or not grounded
7. **Answer Verification**: Passed or needs improvement
8. **Iteration**: Loops back if quality insufficient

## Adding Your Own Documents

### Via Web Interface

1. Go to **Ingest Documents** tab
2. Choose **Upload Files** or **Paste Text**
3. Upload .txt, .md files or paste text
4. Adjust chunk size and overlap if needed
5. Click **Upload** or **Ingest**

### Via API

```bash
curl -X POST "http://localhost:8000/ingest/text" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Your document text here"],
    "metadatas": [{"source": "my_doc"}],
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

### Via Python

```python
from src.core.vector_store import VectorStoreManager

vsm = VectorStoreManager()
vsm.ingest_files(
    file_paths=["path/to/your/file.txt"],
    chunk_size=1000,
    chunk_overlap=200
)
```

## Configuration

### Adjusting Parameters

In Gradio UI:
- Click ⚙️ **Settings** accordion
- Adjust **Max Iterations** (1-5)
- Higher = more correction loops but slower

In code:
```python
result = query_rag_agent(
    question="Your question",
    max_iterations=3  # Default: 3
)
```

### Changing Models

Edit the model in corrective modules:

```python
# In src/corrective/*.py
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Change model here
    temperature=0
)
```

Available Groq models:
- `llama-3.3-70b-versatile` (default, best quality)
- `llama-3.1-8b-instant` (faster, less accurate)
- `mixtral-8x7b-32768` (good balance)

### Changing Embedding Model

Edit in `src/core/embeddings.py`:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Change here
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

Other options:
- `all-mpnet-base-v2` (better quality, slower)
- `all-MiniLM-L12-v2` (larger, more accurate)

## Troubleshooting

### Issue: "Vector store not initialized"
**Solution**: Run `python setup_sample_data.py` first

### Issue: "API key not found"
**Solution**: Check your `.env` file has GROQ_API_KEY and TAVILY_API_KEY

### Issue: Out of memory
**Solution**: Reduce chunk size or use smaller embedding model

### Issue: Slow responses
**Solution**: 
- Use faster Groq model (llama-3.1-8b-instant)
- Reduce max_iterations
- Reduce retrieval k value

### Issue: Import errors
**Solution**: Ensure virtual environment is activated and run `pip install -r requirements.txt`

## Next Steps

1. **Explore the examples**: Try all queries in `data/examples/example_queries.txt`
2. **Add your data**: Ingest your own documents
3. **Monitor workflow**: Watch how the corrective mechanisms work
4. **Tune parameters**: Adjust iterations, chunk size, retrieval count
5. **Extend functionality**: Add more nodes to the LangGraph workflow
6. **Deploy**: Use FastAPI endpoint in production

## Need Help?

- Check the main `README.md` for architecture details
- Review code documentation in each module
- Examine the plan file: `corrective-adaptive-rag-agents.plan.md`
- Test individual components in Python REPL

## Performance Tips

- **For Speed**: Lower max_iterations, use smaller/faster LLM model
- **For Accuracy**: Higher max_iterations, use llama-3.3-70b-versatile
- **For Scale**: Increase chunk overlap, use metadata filtering
- **For Cost**: Reduce iterations, cache common queries

Enjoy your Corrective & Adaptive RAG Agent! 🚀

