"""Embedding models setup for the RAG system."""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and return HuggingFace embeddings model.
    
    Uses sentence-transformers/all-MiniLM-L6-v2 for efficient and accurate embeddings.
    
    Returns:
        HuggingFaceEmbeddings: Configured embeddings model
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

