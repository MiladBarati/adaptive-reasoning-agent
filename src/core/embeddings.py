"""Embedding models setup for the RAG system."""

from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings() -> Any:
    """
    Initialize and return HuggingFace embeddings model.

    Uses sentence-transformers/all-MiniLM-L6-v2 for efficient and accurate embeddings.
    If MOCK_LLM=true, returns FakeEmbeddings to bypass torch concurrency issues.

    Returns:
        Embeddings model
    """
    import os

    mock_llm = os.getenv("MOCK_LLM", "false").lower() == "true"
    if mock_llm:
        from langchain_core.embeddings.fake import DeterministicFakeEmbedding

        from src.core.logging_config import get_logger

        get_logger(__name__).warning("Using MOCK Embeddings for vector operations!")
        return DeterministicFakeEmbedding(size=384)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings
