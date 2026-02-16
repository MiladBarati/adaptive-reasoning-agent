"""Integration tests for API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.vector_store import VectorStoreManager


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def test_vector_store_manager(temp_vector_store_dir):
    """Create a test vector store manager."""
    return VectorStoreManager(persist_directory=str(temp_vector_store_dir))


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.mark.slow
    @patch("src.api.main.vector_store_manager")
    @patch("src.api.main.async_query_rag_agent", new_callable=AsyncMock)
    def test_query_endpoint_integration(self, mock_query, mock_vsm, client):
        """Test query endpoint with full integration."""
        mock_query.return_value = {
            "generation": "Machine learning is a subset of AI.",
            "question": "What is machine learning?",
            "rewritten_question": "What is machine learning and how does it work?",
            "workflow_steps": ["Query rewritten", "Documents retrieved", "Answer generated"],
            "iterations": 1,
            "relevant_docs_count": 2,
            "web_search_results": [],
        }
        mock_vsm.get_stats.return_value = {"document_count": 10, "persist_directory": "./chroma_db"}

        response = client.post(
            "/query", json={"question": "What is machine learning?", "max_iterations": 3}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Machine learning is a subset of AI."
        assert len(data["workflow_steps"]) > 0

    @pytest.mark.slow
    @patch("src.api.main.vector_store_manager")
    def test_ingest_and_query_flow(self, mock_vsm, client, test_vector_store_manager):
        """Test full flow of ingestion and querying."""
        # Setup mock
        mock_vsm.ingest_text_documents.return_value = ["id1", "id2"]
        mock_vsm.get_stats.return_value = {"document_count": 2, "persist_directory": "./chroma_db"}

        # Ingest documents
        ingest_response = client.post(
            "/ingest/text",
            json={
                "texts": ["Machine learning is AI.", "Deep learning uses neural networks."],
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        )

        assert ingest_response.status_code == 200

        # Get stats
        stats_response = client.get("/stats")
        assert stats_response.status_code == 200

    @pytest.mark.slow
    def test_api_error_handling(self, client):
        """Test API error handling."""
        # Test with invalid request
        response = client.post(
            "/query",
            json={
                "question": "Test",
                "max_iterations": 100,  # Invalid: exceeds max
            },
        )

        # Should fail validation
        assert response.status_code in [400, 422]

    @pytest.mark.slow
    @patch("src.api.main.vector_store_manager")
    def test_streaming_endpoint(self, mock_vsm, client):
        """Test streaming query endpoint."""
        from unittest.mock import Mock

        mock_app = Mock()
        mock_app.stream.return_value = [
            {"workflow_steps": ["Step 1"], "iterations": 0},
            {"workflow_steps": ["Step 1", "Step 2"], "iterations": 0, "generation": "Answer"},
        ]

        with patch("src.api.main.create_rag_graph", return_value=mock_app):
            response = client.post(
                "/query/stream", json={"question": "What is machine learning?", "max_iterations": 2}
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
