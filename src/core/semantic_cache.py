"""Semantic cache implementation using ChromaDB."""

import os
import json
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from langchain_chroma import Chroma
from langchain_core.documents import Document
from opentelemetry import trace

from src.core.embeddings import get_embeddings
from src.core.logging_config import get_logger
from src.core.telemetry import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class SemanticCache:
    """
    Semantic cache for RAG queries using ChromaDB.

    Stores successful query-answer pairs and retrieves them based on
    semantic similarity of new queries.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "semantic_cache",
        similarity_threshold: float = 0.90,
    ) -> None:
        """
        Initialize the semantic cache.

        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the cache collection
            similarity_threshold: Cosine similarity threshold (0-1) for cache hits
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.embeddings = get_embeddings()
        self.vector_store: Optional[Chroma] = None

        self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize the ChromaDB collection for caching."""
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
            logger.info(f"Initialized semantic cache in collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize semantic cache: {e}", exc_info=True)

    def check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check the cache for a semantically similar query.

        Args:
            query: The user's question

        Returns:
            Dictionary containing the cached answer and metadata, or None if no hit
        """
        with tracer.start_as_current_span("cache.check") as span:
            span.set_attribute("cache.query", query[:200])

            if not self.vector_store:
                span.set_attribute("cache.hit", False)
                return None

            try:
                results = self.vector_store.similarity_search_with_relevance_scores(query, k=1)

                if not results:
                    span.set_attribute("cache.hit", False)
                    return None

                document, score = results[0]
                span.set_attribute("cache.similarity_score", score)

                if score >= self.similarity_threshold:
                    logger.info(f"Cache HIT for query='{query}' (score={score:.4f})")
                    span.set_attribute("cache.hit", True)

                    try:
                        metadata = document.metadata
                        return {
                            "answer": metadata.get("answer", ""),
                            "original_query": document.page_content,
                            "rewritten_query": metadata.get("rewritten_query", ""),
                            "timestamp": metadata.get("timestamp"),
                            "similarity_score": score,
                        }
                    except Exception as e:
                        logger.error(f"Error parsing cache metadata: {e}")
                        return None
                else:
                    logger.debug(
                        f"Cache MISS for query='{query}' (score={score:.4f} < {self.similarity_threshold})"
                    )
                    span.set_attribute("cache.hit", False)
                    return None

            except Exception as e:
                logger.error(f"Error checking cache: {e}", exc_info=True)
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.record_exception(e)
                return None

    def update_cache(self, query: str, answer: str, rewritten_query: str = "") -> None:
        """
        Add a new query-answer pair to the cache.

        Args:
            query: Original user question
            answer: Generated answer
            rewritten_query: The rewritten version of the query (optional)
        """
        with tracer.start_as_current_span("cache.update") as span:
            span.set_attribute("cache.query", query[:200])

            if not self.vector_store:
                return

            try:
                timestamp = datetime.now().isoformat()

                document = Document(
                    page_content=query,
                    metadata={
                        "answer": answer,
                        "rewritten_query": rewritten_query,
                        "timestamp": timestamp,
                        "type": "cache_entry",
                    },
                )

                self.vector_store.add_documents([document])
                logger.info(f"Updated cache with new entry for query='{query}'")

            except Exception as e:
                logger.error(f"Failed to update cache: {e}", exc_info=True)
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.record_exception(e)

    def clear_cache(self) -> None:
        """Clear the semantic cache."""
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
                self._initialize_cache()
                logger.info("Semantic cache cleared")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
