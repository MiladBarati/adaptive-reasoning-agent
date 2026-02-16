"""Unit tests for agent state module."""

import pytest

from src.agents.state import RAGState


@pytest.mark.unit
class TestRAGState:
    """Test cases for RAGState TypedDict."""

    def test_state_structure(self):
        """Test that RAGState has correct structure."""
        state: RAGState = {
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

        assert state["question"] == "What is machine learning?"
        assert state["iterations"] == 0
        assert state["max_iterations"] == 3
        assert isinstance(state["workflow_steps"], list)

    def test_state_initialization(self, sample_rag_state):
        """Test state initialization with fixture."""
        assert sample_rag_state["question"] == "What is machine learning?"
        assert sample_rag_state["iterations"] == 0
        assert sample_rag_state["max_iterations"] == 3
