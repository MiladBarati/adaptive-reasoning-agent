"""Vector store management using ChromaDB."""

import os
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.core.embeddings import get_embeddings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages the ChromaDB vector store for document storage and retrieval."""

    def __init__(self, persist_directory: str = "./chroma_db") -> None:
        """
        Initialize the vector store manager.

        Args:
            persist_directory: Directory path for persisting the vector store
        """
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        self.vector_store: Optional[Chroma] = None
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """Initialize or load existing ChromaDB vector store."""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            # Load existing vector store
            self.vector_store = Chroma(
                persist_directory=self.persist_directory, embedding_function=self.embeddings
            )
            logger.info(f"Loaded existing vector store from {self.persist_directory}")
        else:
            # Create new vector store
            self.vector_store = Chroma(
                persist_directory=self.persist_directory, embedding_function=self.embeddings
            )
            logger.info(f"Created new vector store at {self.persist_directory}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        ids = self.vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to vector store")
        return ids

    def ingest_text_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[str]:
        """
        Ingest text documents with chunking.

        Args:
            texts: List of text strings to ingest
            metadatas: Optional metadata for each text
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Returns:
            List of document IDs
        """
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Split documents
        documents = []
        for i, text in enumerate(texts):
            chunks = text_splitter.split_text(text)
            for j, chunk in enumerate(chunks):
                metadata = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
                metadata.update({"chunk_id": j, "source_id": i})
                documents.append(Document(page_content=chunk, metadata=metadata))

        # Add to vector store
        return self.add_documents(documents)

    def ingest_files(
        self, file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        """
        Ingest documents from files.

        Args:
            file_paths: List of file paths to ingest
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Returns:
            List of document IDs
        """
        texts = []
        metadatas = []

        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    texts.append(text)
                    metadatas.append({"source": file_path, "filename": os.path.basename(file_path)})
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
                continue

        return self.ingest_text_documents(texts, metadatas, chunk_size, chunk_overlap)

    def get_vector_store(self) -> Chroma:
        """
        Get the ChromaDB vector store instance.

        Returns:
            Chroma vector store
        """
        return self.vector_store

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of relevant documents
        """
        return self.vector_store.similarity_search(query, k=k, filter=filter)

    def get_retriever(self, k: int = 4) -> BaseRetriever:
        """
        Get a retriever interface for the vector store.

        Args:
            k: Number of documents to retrieve

        Returns:
            VectorStoreRetriever
        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        if self.vector_store:
            # Delete the collection and reinitialize
            try:
                self.vector_store.delete_collection()
                logger.info("Cleared vector store")
                self._initialize_vector_store()
            except Exception as e:
                logger.error(f"Error clearing vector store: {e}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with vector store statistics
        """
        try:
            collection = self.vector_store._collection
            count = collection.count()
            return {"document_count": count, "persist_directory": self.persist_directory}
        except Exception as e:
            return {"error": str(e), "persist_directory": self.persist_directory}
