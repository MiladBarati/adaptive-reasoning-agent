"""
Script to ingest sample documents into the vector store.

Run this script to populate the vector store with sample documents
about machine learning, deep learning, NLP, RAG systems, and vector databases.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from src.core.vector_store import VectorStoreManager
from src.core.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)


def ingest_sample_documents() -> None:
    """Ingest all sample documents from data/documents directory."""

    logger.info("=" * 80)
    logger.info("Corrective & Adaptive RAG Agent - Sample Data Setup")
    logger.info("=" * 80)

    # Initialize vector store manager
    logger.info("1. Initializing vector store...")
    vector_store_manager = VectorStoreManager(persist_directory="./chroma_db")

    # Get all sample documents
    documents_dir = Path("data/documents")
    if not documents_dir.exists():
        logger.error(f"Documents directory not found: {documents_dir}")
        return

    # Find all .txt files
    doc_files = list(documents_dir.glob("*.txt"))

    if not doc_files:
        logger.warning(f"No .txt files found in {documents_dir}")
        return

    logger.info(f"2. Found {len(doc_files)} sample documents:")
    for doc_file in doc_files:
        logger.info(f"   - {doc_file.name}")

    # Ingest documents
    logger.info("3. Ingesting documents into vector store...")
    file_paths = [str(doc_file) for doc_file in doc_files]

    try:
        ids = vector_store_manager.ingest_files(
            file_paths=file_paths, chunk_size=1000, chunk_overlap=200
        )

        logger.info(f"Successfully ingested {len(doc_files)} files into {len(ids)} chunks")

        # Get and display stats
        stats = vector_store_manager.get_stats()
        logger.info("4. Vector Store Statistics:")
        logger.info(f"   - Total document chunks: {stats.get('document_count', 0)}")
        logger.info(f"   - Storage location: {stats.get('persist_directory', 'N/A')}")

        logger.info("=" * 80)
        logger.info("Setup Complete!")
        logger.info("=" * 80)
        logger.info("You can now:")
        logger.info("1. Run the Gradio web interface: python -m src.ui.gradio_app")
        logger.info("2. Run the FastAPI server: uvicorn src.api.main:app --reload")
        logger.info("3. Use the agent programmatically from src.agents.rag_graph")
        logger.info("Example queries to try:")
        logger.info("  - What is machine learning?")
        logger.info("  - Explain deep learning architectures")
        logger.info("  - How do vector databases work?")
        logger.info("  - What is RAG and how does it work?")
        logger.info("  - Compare supervised and unsupervised learning")
        logger.info("See data/examples/example_queries.txt for more test queries!")

    except Exception as e:
        logger.error(f"Error ingesting documents: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    ingest_sample_documents()
