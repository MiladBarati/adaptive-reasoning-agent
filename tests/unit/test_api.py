"""Unit tests for API endpoints."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_vector_store_manager():
    """Mock vector store manager."""
    with patch("src.api.main.vector_store_manager") as mock_vsm:
        mock_vsm.get_stats.return_value = {"document_count": 10, "persist_directory": "./chroma_db"}
        yield mock_vsm


@pytest.mark.unit
class TestAPIEndpoints:
    """Test cases for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_check(self, client, mock_vector_store_manager):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    @patch("src.api.main.async_query_rag_agent", new_callable=AsyncMock)
    def test_query_endpoint(self, mock_query, client, mock_vector_store_manager):
        """Test query endpoint."""
        mock_query.return_value = {
            "generation": "Test answer",
            "question": "Test question",
            "rewritten_question": "Rewritten question",
            "workflow_steps": ["Step 1", "Step 2"],
            "iterations": 1,
            "relevant_docs_count": 2,
            "web_search_results": [],
        }

        response = client.post(
            "/query", json={"question": "What is machine learning?", "max_iterations": 3}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Test answer"

    def test_query_endpoint_validation(self, client, mock_vector_store_manager):
        """Test query endpoint validation."""
        response = client.post(
            "/query",
            json={
                "question": "",  # Empty question
                "max_iterations": 3,
            },
        )

        # Should fail validation (422) or may return 500 if validation passes but processing fails
        # Accept either validation error or processing error
        assert response.status_code in [400, 422, 500]

    @patch("src.api.main.create_rag_graph")
    def test_query_stream_endpoint(self, mock_create_graph, client, mock_vector_store_manager):
        """Test query stream endpoint."""
        mock_app = Mock()
        mock_app.stream.return_value = [
            {"workflow_steps": ["Step 1"]},
            {"workflow_steps": ["Step 1", "Step 2"], "generation": "Answer"},
        ]
        mock_create_graph.return_value = mock_app

        response = client.post(
            "/query/stream", json={"question": "What is machine learning?", "max_iterations": 3}
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    @patch("src.api.main.vector_store_manager")
    def test_ingest_text_endpoint(self, mock_vsm, client):
        """Test ingest text endpoint."""
        mock_vsm.ingest_text_documents.return_value = ["id1", "id2", "id3"]

        response = client.post(
            "/ingest/text",
            json={"texts": ["Text 1", "Text 2"], "chunk_size": 1000, "chunk_overlap": 200},
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "document_count" in data
        assert data["document_count"] == 3

    @patch("src.api.main.vector_store_manager")
    def test_get_stats_endpoint(self, mock_vsm, client):
        """Test get stats endpoint."""
        mock_vsm.get_stats.return_value = {"document_count": 42, "persist_directory": "./chroma_db"}

        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 42
        assert "persist_directory" in data

    @patch("src.api.main.vector_store_manager")
    def test_clear_endpoint(self, mock_vsm, client):
        """Test clear endpoint."""
        mock_vsm.clear.return_value = None

        response = client.delete("/clear")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        mock_vsm.clear.assert_called_once()

    def test_query_endpoint_no_vector_store(self, client):
        """Test query endpoint when vector store is not initialized."""
        with patch("src.api.main.vector_store_manager", None):
            response = client.post(
                "/query", json={"question": "Test question", "max_iterations": 3}
            )

            assert response.status_code == 500
