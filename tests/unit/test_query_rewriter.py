"""Unit tests for query rewriter module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.corrective.query_rewriter import QueryRewriter


@pytest.mark.unit
class TestQueryRewriter:
    """Test cases for QueryRewriter."""
    
    @patch('src.corrective.query_rewriter.ChatGroq')
    def test_initialization(self, mock_groq):
        """Test query rewriter initialization."""
        mock_llm = MagicMock()
        mock_groq.return_value = mock_llm
        
        rewriter = QueryRewriter()
        
        assert rewriter.llm is not None
        assert rewriter.prompt is not None
        assert rewriter.chain is not None
    
    @patch('src.corrective.query_rewriter.ChatGroq')
    def test_rewrite(self, mock_groq):
        """Test query rewriting."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "What is machine learning and how does it work?"
        mock_llm.invoke.return_value = mock_response
        mock_groq.return_value = mock_llm
        
        rewriter = QueryRewriter()
        rewriter.llm = mock_llm
        
        original = "What is ML?"
        rewritten = rewriter.rewrite(original)
        
        assert rewritten is not None
        assert isinstance(rewritten, str)
        assert len(rewritten) > 0
    
    @patch('src.corrective.query_rewriter.ChatGroq')
    def test_rewrite_error_handling(self, mock_groq):
        """Test error handling in query rewriting."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API Error")
        mock_groq.return_value = mock_llm
        
        rewriter = QueryRewriter()
        rewriter.llm = mock_llm
        
        original = "test query"
        rewritten = rewriter.rewrite(original)
        
        # Should return original query on error
        assert rewritten == original
    
    @patch('src.corrective.query_rewriter.ChatGroq')
    def test_rewrite_multiple(self, mock_groq):
        """Test generating multiple query variations."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "1. What is machine learning?\n2. How does machine learning work?\n3. Explain machine learning"
        mock_llm.invoke.return_value = mock_response
        mock_groq.return_value = mock_llm
        
        rewriter = QueryRewriter()
        rewriter.llm = mock_llm
        
        original = "What is ML?"
        variations = rewriter.rewrite_multiple(original, num_variations=3)
        
        assert isinstance(variations, list)
        assert len(variations) > 0
        assert all(isinstance(v, str) for v in variations)
    
    @patch('src.corrective.query_rewriter.ChatGroq')
    def test_rewrite_multiple_error_handling(self, mock_groq):
        """Test error handling in multiple query rewriting."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API Error")
        mock_groq.return_value = mock_llm
        
        rewriter = QueryRewriter()
        rewriter.llm = mock_llm
        
        original = "test query"
        variations = rewriter.rewrite_multiple(original, num_variations=3)
        
        # Should return original query in list on error
        assert variations == [original]

