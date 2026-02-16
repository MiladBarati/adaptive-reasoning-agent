"""Shared fixtures and configuration for pytest tests."""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Generator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.core.vector_store import VectorStoreManager
from src.core.embeddings import get_embeddings


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_vector_store_dir(test_data_dir) -> Generator[Path, None, None]:
    """Create a temporary directory for vector store during tests."""
    temp_dir = test_data_dir / "test_chroma_db"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vector_store_manager(temp_vector_store_dir) -> VectorStoreManager:
    """Create a vector store manager for testing."""
    return VectorStoreManager(persist_directory=str(temp_vector_store_dir))


@pytest.fixture
def sample_documents() -> list[Document]:
    """Sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            metadata={"source": "test1", "topic": "machine_learning"},
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers to learn complex patterns.",
            metadata={"source": "test2", "topic": "deep_learning"},
        ),
        Document(
            page_content="Natural language processing helps computers understand human language.",
            metadata={"source": "test3", "topic": "nlp"},
        ),
    ]


@pytest.fixture
def populated_vector_store(vector_store_manager, sample_documents) -> VectorStoreManager:
    """Vector store manager with sample documents already added."""
    vector_store_manager.add_documents(sample_documents)
    return vector_store_manager


@pytest.fixture
def mock_llm_response():
    """Mock LLM response."""

    def _create_mock_response(content: str):
        mock = Mock()
        mock.content = content
        return mock

    return _create_mock_response


@pytest.fixture
def mock_groq_llm(mock_llm_response):
    """Mock Groq LLM."""
    with patch("langchain_groq.ChatGroq") as mock_llm_class:
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=mock_llm_response("Mocked response"))
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        mock_llm_class.return_value = mock_llm
        yield mock_llm


@pytest.fixture
def mock_embeddings():
    """Mock embeddings model."""
    mock_emb = Mock()
    mock_emb.embed_query = Mock(return_value=[0.1] * 384)  # Mock embedding vector
    mock_emb.embed_documents = Mock(return_value=[[0.1] * 384] * 3)
    return mock_emb


@pytest.fixture
def mock_tavily_client():
    """Mock Tavily search client."""
    with patch("tavily.TavilyClient") as mock_client_class:
        mock_client = Mock()
        mock_client.search = Mock(
            return_value={
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "content": "This is test content from web search.",
                    }
                ]
            }
        )
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    monkeypatch.setenv("TAVILY_API_KEY", "test_tavily_key")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "test_langchain_key")


@pytest.fixture
def sample_rag_state():
    """Sample RAG state for testing."""
    return {
        "question": "What is machine learning?",
        "rewritten_question": "",
        "documents": [],
        "generation": "",
        "iterations": 0,
        "max_iterations": 3,
        "web_search_needed": False,
        "web_search_results": [],
        "relevant_docs_count": 0,
        "workflow_steps": [],
        "is_grounded": False,
        "is_answer_good": False,
    }


@pytest.fixture
def mock_structured_output():
    """Mock structured output from LLM."""

    def _create_mock_output(binary_score: str):
        mock = Mock()
        mock.binary_score = binary_score
        return mock

    return _create_mock_output
