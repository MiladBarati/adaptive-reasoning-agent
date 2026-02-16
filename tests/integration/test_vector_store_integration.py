"""Integration tests for vector store operations."""

import pytest
from langchain_core.documents import Document

from src.core.vector_store import VectorStoreManager
from src.core.retriever import AdvancedRetriever


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Integration tests for vector store operations."""

    def test_document_ingestion_and_retrieval(self, vector_store_manager):
        """Test full cycle of document ingestion and retrieval."""
        # Ingest documents
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand language.",
        ]
        metadatas = [
            {"source": "ml_doc", "topic": "machine_learning"},
            {"source": "dl_doc", "topic": "deep_learning"},
            {"source": "nlp_doc", "topic": "nlp"},
        ]

        ids = vector_store_manager.ingest_text_documents(
            texts=texts, metadatas=metadatas, chunk_size=100, chunk_overlap=20
        )

        assert len(ids) > 0

        # Retrieve documents
        results = vector_store_manager.similarity_search(query="What is machine learning?", k=2)

        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc in results)

    def test_retriever_integration(self, populated_vector_store):
        """Test AdvancedRetriever with vector store."""
        retriever = AdvancedRetriever(
            vector_store_manager=populated_vector_store, k=2, use_multi_query=False
        )

        results = retriever.retrieve("What is machine learning?")

        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_vector_store_persistence(self, temp_vector_store_dir):
        """Test that vector store persists data."""
        # Create first manager and add documents
        vsm1 = VectorStoreManager(persist_directory=str(temp_vector_store_dir))
        texts = ["Test document for persistence."]
        vsm1.ingest_text_documents(texts=texts)

        # Create second manager with same directory
        vsm2 = VectorStoreManager(persist_directory=str(temp_vector_store_dir))

        # Should be able to retrieve documents
        results = vsm2.similarity_search("test", k=1)
        assert len(results) > 0

    def test_vector_store_stats(self, populated_vector_store):
        """Test vector store statistics."""
        stats = populated_vector_store.get_stats()

        assert "document_count" in stats
        assert stats["document_count"] > 0
        assert "persist_directory" in stats

    def test_vector_store_clear_and_reuse(self, vector_store_manager, sample_documents):
        """Test clearing vector store and reusing it."""
        # Add documents
        ids = vector_store_manager.add_documents(sample_documents)
        assert len(ids) > 0

        # Clear
        vector_store_manager.clear()

        # Add again
        ids2 = vector_store_manager.add_documents(sample_documents)
        assert len(ids2) > 0
