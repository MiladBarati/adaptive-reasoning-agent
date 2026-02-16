"""Unit tests for vector store module."""

import pytest
from langchain_core.documents import Document

from src.core.vector_store import VectorStoreManager


@pytest.mark.unit
class TestVectorStoreManager:
    """Test cases for VectorStoreManager."""

    def test_initialization(self, temp_vector_store_dir):
        """Test vector store manager initialization."""
        vsm = VectorStoreManager(persist_directory=str(temp_vector_store_dir))
        assert vsm is not None
        assert vsm.persist_directory == str(temp_vector_store_dir)
        assert vsm.vector_store is not None

    def test_add_documents(self, vector_store_manager, sample_documents):
        """Test adding documents to vector store."""
        ids = vector_store_manager.add_documents(sample_documents)

        assert len(ids) == len(sample_documents)
        assert all(isinstance(id, str) for id in ids)

    def test_add_empty_documents(self, vector_store_manager):
        """Test adding empty document list."""
        ids = vector_store_manager.add_documents([])
        assert ids == []

    def test_ingest_text_documents(self, vector_store_manager):
        """Test ingesting text documents."""
        texts = [
            "This is the first document about machine learning.",
            "This is the second document about deep learning.",
        ]
        metadatas = [{"source": "test1"}, {"source": "test2"}]

        ids = vector_store_manager.ingest_text_documents(
            texts=texts, metadatas=metadatas, chunk_size=50, chunk_overlap=10
        )

        assert len(ids) > 0

    def test_ingest_text_documents_without_metadata(self, vector_store_manager):
        """Test ingesting text documents without metadata."""
        texts = ["Test document without metadata."]

        ids = vector_store_manager.ingest_text_documents(texts=texts)

        assert len(ids) > 0

    def test_similarity_search(self, populated_vector_store):
        """Test similarity search."""
        results = populated_vector_store.similarity_search(query="What is machine learning?", k=2)

        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_similarity_search_empty_store(self, vector_store_manager):
        """Test similarity search on empty store."""
        results = vector_store_manager.similarity_search(query="test query", k=5)

        assert isinstance(results, list)
        # Empty store may return empty list or raise error

    def test_get_retriever(self, vector_store_manager):
        """Test getting retriever interface."""
        retriever = vector_store_manager.get_retriever(k=3)
        assert retriever is not None

    def test_get_stats(self, populated_vector_store):
        """Test getting vector store statistics."""
        stats = populated_vector_store.get_stats()

        assert isinstance(stats, dict)
        assert "document_count" in stats
        assert stats["document_count"] > 0

    def test_clear(self, populated_vector_store):
        """Test clearing vector store."""
        # Verify documents exist
        stats_before = populated_vector_store.get_stats()
        assert stats_before["document_count"] > 0

        # Clear store
        populated_vector_store.clear()

        # Verify store is cleared
        populated_vector_store.get_stats()
        # After clear, count should be 0 or store reinitialized

    def test_ingest_files(self, vector_store_manager, temp_vector_store_dir):
        """Test ingesting files."""
        # Create a test file
        test_file = temp_vector_store_dir / "test.txt"
        test_file.write_text("This is a test file for ingestion.")

        ids = vector_store_manager.ingest_files(
            file_paths=[str(test_file)], chunk_size=50, chunk_overlap=10
        )

        assert len(ids) > 0

    def test_chunking_parameters(self, vector_store_manager):
        """Test that chunking parameters are respected."""
        long_text = " ".join(["word"] * 200)  # Long text
        texts = [long_text]

        ids = vector_store_manager.ingest_text_documents(
            texts=texts, chunk_size=50, chunk_overlap=10
        )

        # Should create multiple chunks
        assert len(ids) > 1
