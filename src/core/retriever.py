"""Advanced retriever with multi-query and filtering capabilities."""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from dotenv import load_dotenv

from src.core.vector_store import VectorStoreManager
from src.core.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)


class AdvancedRetriever:
    """Advanced retriever with multi-query and semantic search capabilities."""
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        k: int = 4,
        use_multi_query: bool = False
    ) -> None:
        """
        Initialize the advanced retriever.
        
        Args:
            vector_store_manager: Vector store manager instance
            k: Number of documents to retrieve
            use_multi_query: Whether to use multi-query retrieval
        """
        self.vector_store_manager = vector_store_manager
        self.k = k
        self.use_multi_query = use_multi_query
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        
    def retrieve(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            filter: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        if self.use_multi_query:
            return self._multi_query_retrieve(query, filter)
        else:
            return self.vector_store_manager.similarity_search(
                query, k=self.k, filter=filter
            )
    
    def _multi_query_retrieve(
        self,
        query: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve documents using multiple query variations.
        
        Args:
            query: Original search query
            filter: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        # Generate query variations
        query_variations = self._generate_query_variations(query)
        
        # Retrieve documents for each variation
        all_docs = []
        seen_content = set()
        
        for q in query_variations:
            docs = self.vector_store_manager.similarity_search(
                q, k=self.k, filter=filter
            )
            for doc in docs:
                # Deduplicate based on content
                if doc.page_content not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        # Return top-k unique documents
        return all_docs[:self.k]
    
    def _generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate variations of the query for multi-query retrieval.
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations including the original
        """
        prompt = ChatPromptTemplate.from_template(
            """You are an AI assistant helping to improve document retrieval.
            Given a user question, generate {num} alternative versions of the question 
            to retrieve relevant documents from a vector database. 
            
            Provide these alternative questions separated by newlines.
            
            Original question: {question}
            
            Alternative questions:"""
        )
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "question": query,
                "num": num_variations - 1  # -1 because we'll include original
            })
            
            # Parse response
            variations = [query]  # Start with original
            alt_questions = response.content.strip().split('\n')
            for q in alt_questions:
                q = q.strip()
                # Remove numbering if present
                if q and len(q) > 0:
                    if q[0].isdigit() and '.' in q[:3]:
                        q = q.split('.', 1)[1].strip()
                    if q:
                        variations.append(q)
            
            return variations[:num_variations]
        except Exception as e:
            logger.error(f"Error generating query variations: {e}", exc_info=True)
            return [query]
    
    def get_retriever(self) -> BaseRetriever:
        """
        Get a retriever interface compatible with LangChain.
        
        Returns:
            Retriever instance
        """
        return self.vector_store_manager.get_retriever(k=self.k)

