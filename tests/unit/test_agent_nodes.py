"""Unit tests for agent nodes module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.agents import nodes
from src.agents.state import RAGState


@pytest.mark.unit
class TestAgentNodes:
    """Test cases for agent nodes."""

    @pytest.fixture(autouse=True)
    def mock_tracer(self):
        """Mock the tracer to avoid AttributeErrors."""
        with patch("src.agents.nodes.tracer") as mock:
            # Mock the context manager returned by start_as_current_span
            mock_span = MagicMock()
            mock.start_as_current_span.return_value.__enter__.return_value = mock_span
            yield mock

    def test_initialize_components(self, vector_store_manager):
        """Test component initialization."""
        nodes.initialize_components(vector_store_manager)

        assert nodes.vector_store_manager == vector_store_manager
        assert nodes.retriever is not None
        assert nodes.query_rewriter is not None
        assert nodes.relevance_grader is not None
        assert nodes.hallucination_checker is not None
        assert nodes.answer_verifier is not None
        assert nodes.llm is not None

    @patch("src.agents.nodes.query_rewriter")
    def test_rewrite_query(self, mock_rewriter, vector_store_manager, sample_rag_state):
        """Test rewrite_query node."""
        nodes.initialize_components(vector_store_manager)
        mock_rewriter.rewrite.return_value = "What is machine learning and how does it work?"

        result = nodes.rewrite_query(sample_rag_state)

        assert result["rewritten_question"] != ""
        assert "rewritten" in result["workflow_steps"][-1].lower()

    @patch("src.agents.nodes.retriever")
    def test_retrieve_documents(self, mock_retriever, vector_store_manager, sample_rag_state):
        """Test retrieve_documents node."""
        nodes.initialize_components(vector_store_manager)

        mock_docs = [Document(page_content="Test content", metadata={})]
        mock_retriever.retrieve.return_value = mock_docs

        result = nodes.retrieve_documents(sample_rag_state)

        assert len(result["documents"]) > 0
        assert "retrieved" in result["workflow_steps"][-1].lower()

    def test_grade_documents(self, vector_store_manager, sample_rag_state):
        """Test grade_documents node."""
        nodes.initialize_components(vector_store_manager)

        mock_docs = [Document(page_content="Test", metadata={})]
        sample_rag_state["documents"] = mock_docs

        # Mock the relevance_grader's grade_documents method
        nodes.relevance_grader.grade_documents = Mock(return_value=(mock_docs, []))

        result = nodes.grade_documents(sample_rag_state)

        assert result["relevant_docs_count"] > 0
        assert result["web_search_needed"] is False
        assert "graded" in result["workflow_steps"][-1].lower()

    @patch("tavily.TavilyClient")
    def test_web_search(self, mock_tavily_client, vector_store_manager, sample_rag_state):
        """Test web_search node."""
        nodes.initialize_components(vector_store_manager)

        mock_client = Mock()
        mock_client.search.return_value = {
            "results": [
                {
                    "content": "Web search result",
                    "url": "https://example.com",
                    "title": "Test Result",
                }
            ]
        }
        mock_tavily_client.return_value = mock_client

        result = nodes.web_search(sample_rag_state)

        assert len(result["web_search_results"]) > 0
        assert "web search" in result["workflow_steps"][-1].lower()

    @patch("src.agents.nodes.llm")
    def test_generate_answer(self, mock_llm, vector_store_manager, sample_rag_state):
        """Test generate_answer node."""
        nodes.initialize_components(vector_store_manager)

        mock_docs = [Document(page_content="Test content", metadata={})]
        sample_rag_state["documents"] = mock_docs

        mock_response = MagicMock()
        mock_response.content = "This is a generated answer."
        mock_llm.invoke.return_value = mock_response

        result = nodes.generate_answer(sample_rag_state)

        assert result["generation"] != ""
        assert "generated" in result["workflow_steps"][-1].lower()

    def test_check_hallucination(self, vector_store_manager, sample_rag_state):
        """Test check_hallucination node."""
        nodes.initialize_components(vector_store_manager)

        sample_rag_state["generation"] = "Test answer"
        sample_rag_state["documents"] = [Document(page_content="Test", metadata={})]

        # Mock the hallucination_checker's check method
        nodes.hallucination_checker.check = Mock(return_value=True)

        result = nodes.check_hallucination(sample_rag_state)

        assert result["is_grounded"] is True
        assert "hallucination" in result["workflow_steps"][-1].lower()

    def test_verify_answer(self, vector_store_manager, sample_rag_state):
        """Test verify_answer node."""
        nodes.initialize_components(vector_store_manager)

        sample_rag_state["generation"] = "Test answer"

        # Mock the answer_verifier's verify method
        nodes.answer_verifier.verify = Mock(return_value=True)

        result = nodes.verify_answer(sample_rag_state)

        assert result["is_answer_good"] is True
        assert "verification" in result["workflow_steps"][-1].lower()

    def test_increment_iteration(self, vector_store_manager, sample_rag_state):
        """Test increment_iteration node."""
        nodes.initialize_components(vector_store_manager)

        initial_iterations = sample_rag_state["iterations"]
        result = nodes.increment_iteration(sample_rag_state)

        assert result["iterations"] == initial_iterations + 1
        assert "iteration" in result["workflow_steps"][-1].lower()

    def test_decide_to_generate_with_docs(self, sample_rag_state):
        """Test decide_to_generate when documents are available."""
        sample_rag_state["web_search_needed"] = False

        decision = nodes.decide_to_generate(sample_rag_state)

        assert decision == "generate"

    def test_decide_to_generate_without_docs(self, sample_rag_state):
        """Test decide_to_generate when no documents are available."""
        sample_rag_state["web_search_needed"] = True

        decision = nodes.decide_to_generate(sample_rag_state)

        assert decision == "web_search"

    def test_decide_after_hallucination_check_grounded(self, sample_rag_state):
        """Test decide_after_hallucination_check when grounded."""
        sample_rag_state["is_grounded"] = True

        decision = nodes.decide_after_hallucination_check(sample_rag_state)

        assert decision == "verify"

    def test_decide_after_hallucination_check_not_grounded(self, sample_rag_state):
        """Test decide_after_hallucination_check when not grounded."""
        sample_rag_state["is_grounded"] = False
        sample_rag_state["iterations"] = 1
        sample_rag_state["max_iterations"] = 3

        decision = nodes.decide_after_hallucination_check(sample_rag_state)

        assert decision == "retry"

    def test_decide_after_verification_good(self, sample_rag_state):
        """Test decide_after_verification when answer is good."""
        sample_rag_state["is_answer_good"] = True

        decision = nodes.decide_after_verification(sample_rag_state)

        assert decision == "end"

    def test_decide_after_verification_bad(self, sample_rag_state):
        """Test decide_after_verification when answer needs improvement."""
        sample_rag_state["is_answer_good"] = False
        sample_rag_state["iterations"] = 1
        sample_rag_state["max_iterations"] = 3

        decision = nodes.decide_after_verification(sample_rag_state)

        assert decision == "retry"
