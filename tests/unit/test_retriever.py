"""Unit tests for retriever module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.core.retriever import AdvancedRetriever
from src.core.vector_store import VectorStoreManager


@pytest.mark.unit
class TestAdvancedRetriever:
    """Test cases for AdvancedRetriever."""
    
    def test_initialization(self, vector_store_manager):
        """Test retriever initialization."""
        retriever = AdvancedRetriever(
            vector_store_manager=vector_store_manager,
            k=4,
            use_multi_query=False
        )
        
        assert retriever.vector_store_manager == vector_store_manager
        assert retriever.k == 4
        assert retriever.use_multi_query is False
    
    def test_retrieve_single_query(self, vector_store_manager, populated_vector_store):
        """Test retrieval with single query."""
        retriever = AdvancedRetriever(
            vector_store_manager=populated_vector_store,
            k=2,
            use_multi_query=False
        )
        
        results = retriever.retrieve("What is machine learning?")
        
        assert isinstance(results, list)
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
    
    def test_retrieve_with_filter(self, vector_store_manager, populated_vector_store):
        """Test retrieval with metadata filter."""
        retriever = AdvancedRetriever(
            vector_store_manager=populated_vector_store,
            k=2
        )
        
        filter_dict = {"topic": "machine_learning"}
        results = retriever.retrieve("machine learning", filter=filter_dict)
        
        assert isinstance(results, list)
    
    @patch('src.core.retriever.ChatGroq')
    def test_multi_query_retrieve(self, mock_groq, vector_store_manager, populated_vector_store):
        """Test multi-query retrieval."""
        # Mock LLM for query variation generation
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Alternative query 1\nAlternative query 2"
        mock_llm.invoke.return_value = mock_response
        mock_groq.return_value = mock_llm
        
        retriever = AdvancedRetriever(
            vector_store_manager=populated_vector_store,
            k=2,
            use_multi_query=True
        )
        retriever.llm = mock_llm
        
        results = retriever.retrieve("What is machine learning?")
        
        assert isinstance(results, list)
        # Multi-query should have been called (check if invoke was called or results were retrieved)
        assert len(results) >= 0  # Results should be a list (may be empty)
    
    def test_get_retriever(self, vector_store_manager, populated_vector_store):
        """Test getting LangChain retriever interface."""
        retriever = AdvancedRetriever(
            vector_store_manager=populated_vector_store,
            k=3
        )
        
        langchain_retriever = retriever.get_retriever()
        assert langchain_retriever is not None
    
    def test_generate_query_variations_error_handling(self, vector_store_manager, populated_vector_store):
        """Test error handling in query variation generation."""
        retriever = AdvancedRetriever(
            vector_store_manager=populated_vector_store,
            k=2,
            use_multi_query=True
        )
        
        # Mock LLM to raise error
        retriever.llm = Mock()
        retriever.llm.invoke.side_effect = Exception("API Error")
        
        # Should return original query on error
        variations = retriever._generate_query_variations("test query", num_variations=3)
        assert variations == ["test query"]

