"""Integration tests for the full RAG workflow."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.agents.rag_graph import create_rag_graph, query_rag_agent
from src.core.vector_store import VectorStoreManager


@pytest.mark.integration
class TestRAGWorkflow:
    """Integration tests for the complete RAG workflow."""

    @pytest.mark.slow
    def test_create_rag_graph(self, vector_store_manager):
        """Test creating the RAG graph."""
        graph = create_rag_graph(vector_store_manager=vector_store_manager)

        assert graph is not None
        # Graph should be compiled
        assert hasattr(graph, "invoke")

    @pytest.mark.slow
    @patch("src.agents.nodes.ChatGroq")
    @patch("src.corrective.query_rewriter.ChatGroq")
    @patch("src.corrective.relevance_grader.ChatGroq")
    @patch("src.corrective.hallucination_checker.ChatGroq")
    @patch("src.corrective.answer_verifier.ChatGroq")
    @patch("src.core.retriever.ChatGroq")
    @patch("langchain_core.output_parsers.StrOutputParser")
    def test_query_rag_agent_full_workflow(
        self,
        mock_parser,
        mock_retriever_groq,
        mock_verifier_groq,
        mock_checker_groq,
        mock_grader_groq,
        mock_rewriter_groq,
        mock_nodes_groq,
        populated_vector_store,
    ):
        """Test the full RAG agent workflow with mocks."""
        # Setup mocks
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Machine learning is a subset of AI."
        mock_llm.invoke.return_value = mock_response

        # Mock structured output
        mock_structured = Mock()
        mock_structured.binary_score = "yes"
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_structured

        # Mock StrOutputParser
        mock_parser_instance = Mock()
        mock_parser_instance.invoke.return_value = "Machine learning is a subset of AI."
        mock_parser.return_value = mock_parser_instance

        for mock_groq in [
            mock_verifier_groq,
            mock_checker_groq,
            mock_grader_groq,
            mock_rewriter_groq,
            mock_nodes_groq,
            mock_retriever_groq,
        ]:
            mock_groq.return_value = mock_llm

        result = query_rag_agent(
            question="What is machine learning?",
            max_iterations=2,
            vector_store_manager=populated_vector_store,
        )

        assert result is not None
        assert "generation" in result
        assert "workflow_steps" in result
        assert len(result["workflow_steps"]) > 0
        assert result["question"] == "What is machine learning?"

    @pytest.mark.slow
    def test_rag_workflow_with_sample_data(self, populated_vector_store):
        """Test RAG workflow with actual sample data."""
        # This test requires actual API keys and may be slow
        # Skip if API keys are not available
        import os

        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not available")

        result = query_rag_agent(
            question="What is machine learning?",
            max_iterations=1,
            vector_store_manager=populated_vector_store,
        )

        assert result is not None
        assert "generation" in result
        assert len(result["workflow_steps"]) > 0

    @pytest.mark.slow
    def test_rag_workflow_iteration_limit(self, populated_vector_store):
        """Test that RAG workflow respects iteration limits."""
        import os

        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not available")

        result = query_rag_agent(
            question="What is deep learning?",
            max_iterations=1,
            vector_store_manager=populated_vector_store,
        )

        assert result["iterations"] <= 1

    @pytest.mark.slow
    @patch("tavily.TavilyClient")
    def test_rag_workflow_web_search_fallback(self, mock_tavily_client, vector_store_manager):
        """Test RAG workflow with web search fallback."""
        import os

        if not os.getenv("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not available")

        # Mock Tavily client
        mock_client = Mock()
        mock_client.search.return_value = {
            "results": [
                {
                    "content": "Web search result content",
                    "url": "https://example.com",
                    "title": "Example Result",
                }
            ]
        }
        mock_tavily_client.return_value = mock_client

        # Use empty vector store to trigger web search
        result = query_rag_agent(
            question="What is quantum computing?",
            max_iterations=1,
            vector_store_manager=vector_store_manager,
        )

        assert result is not None
        # Web search should be triggered when no relevant docs found
