"""Unit tests for relevance grader module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.corrective.relevance_grader import RelevanceGrader, GradeDocument


@pytest.mark.unit
class TestRelevanceGrader:
    """Test cases for RelevanceGrader."""
    
    @patch('src.corrective.relevance_grader.ChatGroq')
    def test_initialization(self, mock_groq):
        """Test relevance grader initialization."""
        mock_llm = MagicMock()
        mock_groq.return_value = mock_llm
        
        grader = RelevanceGrader()
        
        assert grader.llm is not None
        assert grader.prompt is not None
        assert grader.chain is not None
    
    @patch('src.corrective.relevance_grader.ChatGroq')
    def test_grade_relevant(self, mock_groq):
        """Test grading a relevant document."""
        mock_llm = MagicMock()
        mock_structured_output = Mock()
        mock_structured_output.binary_score = "yes"
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_structured_output
        mock_groq.return_value = mock_llm
        
        grader = RelevanceGrader()
        grader.chain = Mock()
        grader.chain.invoke.return_value = mock_structured_output
        
        doc = Document(page_content="Machine learning is a subset of AI.", metadata={})
        is_relevant = grader.grade(doc, "What is machine learning?")
        
        assert is_relevant is True
    
    @patch('src.corrective.relevance_grader.ChatGroq')
    def test_grade_irrelevant(self, mock_groq):
        """Test grading an irrelevant document."""
        mock_llm = MagicMock()
        mock_structured_output = Mock()
        mock_structured_output.binary_score = "no"
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_structured_output
        mock_groq.return_value = mock_llm
        
        grader = RelevanceGrader()
        grader.chain = Mock()
        grader.chain.invoke.return_value = mock_structured_output
        
        doc = Document(page_content="The weather is sunny today.", metadata={})
        is_relevant = grader.grade(doc, "What is machine learning?")
        
        assert is_relevant is False
    
    @patch('src.corrective.relevance_grader.ChatGroq')
    def test_grade_error_handling(self, mock_groq):
        """Test error handling in grading."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("API Error")
        mock_groq.return_value = mock_llm
        
        grader = RelevanceGrader()
        grader.chain = Mock()
        grader.chain.invoke.side_effect = Exception("API Error")
        
        doc = Document(page_content="Test content", metadata={})
        is_relevant = grader.grade(doc, "test question")
        
        # Should default to relevant on error
        assert is_relevant is True
    
    @patch('src.corrective.relevance_grader.ChatGroq')
    def test_grade_documents(self, mock_groq):
        """Test grading multiple documents."""
        mock_llm = MagicMock()
        mock_groq.return_value = mock_llm
        
        grader = RelevanceGrader()
        
        # Mock grade method to return alternating results
        grader.grade = Mock(side_effect=[True, False, True])
        
        docs = [
            Document(page_content="ML content", metadata={}),
            Document(page_content="Weather content", metadata={}),
            Document(page_content="AI content", metadata={})
        ]
        
        relevant, irrelevant = grader.grade_documents(docs, "What is machine learning?")
        
        assert len(relevant) == 2
        assert len(irrelevant) == 1
        assert all(isinstance(doc, Document) for doc in relevant)
        assert all(isinstance(doc, Document) for doc in irrelevant)
    
    @patch('src.corrective.relevance_grader.ChatGroq')
    def test_filter_relevant(self, mock_groq):
        """Test filtering to only relevant documents."""
        mock_llm = MagicMock()
        mock_groq.return_value = mock_llm
        
        grader = RelevanceGrader()
        grader.grade_documents = Mock(return_value=(
            [Document(page_content="Relevant", metadata={})],
            [Document(page_content="Irrelevant", metadata={})]
        ))
        
        docs = [Document(page_content="Test", metadata={})]
        relevant = grader.filter_relevant(docs, "test question")
        
        assert len(relevant) == 1
        assert all(isinstance(doc, Document) for doc in relevant)

