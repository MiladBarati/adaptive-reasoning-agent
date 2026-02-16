"""State management for the RAG agent graph."""

from typing import List, TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict):
    """State for the RAG agent workflow."""
    
    question: str  # Original user question
    rewritten_question: str  # Rewritten question for better retrieval
    documents: List[Document]  # Retrieved documents
    generation: str  # Generated answer
    iterations: int  # Current iteration count
    max_iterations: int  # Maximum allowed iterations
    web_search_needed: bool  # Whether web search is needed
    web_search_results: List[Document]  # Results from web search
    relevant_docs_count: int  # Count of relevant documents
    workflow_steps: List[str]  # Log of workflow steps taken
    is_grounded: bool  # Whether the answer is grounded in documents
    is_answer_good: bool  # Whether the answer is good
    cache_hit: bool  # Whether the answer came from cache

