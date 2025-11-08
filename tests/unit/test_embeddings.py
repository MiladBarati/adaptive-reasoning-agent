"""Unit tests for embeddings module."""

import pytest
from src.core.embeddings import get_embeddings


@pytest.mark.unit
class TestEmbeddings:
    """Test cases for embeddings functionality."""
    
    def test_get_embeddings_returns_embeddings(self):
        """Test that get_embeddings returns an embeddings model."""
        embeddings = get_embeddings()
        assert embeddings is not None
    
    def test_embeddings_generate_vectors(self):
        """Test that embeddings can generate vectors."""
        embeddings = get_embeddings()
        test_text = "This is a test sentence."
        vector = embeddings.embed_query(test_text)
        
        assert vector is not None
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(x, (int, float)) for x in vector)
    
    def test_embeddings_consistency(self):
        """Test that same text produces same embedding."""
        embeddings = get_embeddings()
        test_text = "Consistency test"
        
        vector1 = embeddings.embed_query(test_text)
        vector2 = embeddings.embed_query(test_text)
        
        assert vector1 == vector2
    
    def test_embeddings_different_texts(self):
        """Test that different texts produce different embeddings."""
        embeddings = get_embeddings()
        
        vector1 = embeddings.embed_query("First text")
        vector2 = embeddings.embed_query("Second text")
        
        assert vector1 != vector2
    
    def test_embeddings_batch(self):
        """Test batch embedding generation."""
        embeddings = get_embeddings()
        texts = ["First text", "Second text", "Third text"]
        
        vectors = embeddings.embed_documents(texts)
        
        assert len(vectors) == len(texts)
        assert all(len(v) > 0 for v in vectors)

