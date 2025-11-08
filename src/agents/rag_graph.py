"""LangGraph workflow for the Corrective & Adaptive RAG Agent."""

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from typing import Optional, Dict, Any

from src.agents.state import RAGState
from src.agents import nodes
from src.core.vector_store import VectorStoreManager
from src.core.logging_config import get_logger

logger = get_logger(__name__)


def create_rag_graph(
    vector_store_manager: Optional[VectorStoreManager] = None,
    persist_directory: str = "./chroma_db"
) -> CompiledGraph:
    """
    Create the RAG agent graph with corrective mechanisms.
    
    Args:
        vector_store_manager: Optional pre-initialized vector store manager
        persist_directory: Directory for vector store persistence
        
    Returns:
        Compiled LangGraph workflow
    """
    # Initialize vector store manager if not provided
    if vector_store_manager is None:
        vector_store_manager = VectorStoreManager(persist_directory=persist_directory)
    
    # Initialize all node components
    nodes.initialize_components(vector_store_manager)
    
    # Create the graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("rewrite", nodes.rewrite_query)
    workflow.add_node("retrieve", nodes.retrieve_documents)
    workflow.add_node("grade", nodes.grade_documents)
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("generate", nodes.generate_answer)
    workflow.add_node("check_hallucination", nodes.check_hallucination)
    workflow.add_node("verify", nodes.verify_answer)
    workflow.add_node("increment", nodes.increment_iteration)
    
    # Set entry point
    workflow.set_entry_point("rewrite")
    
    # Add edges
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade")
    
    # Conditional edge after grading
    workflow.add_conditional_edges(
        "grade",
        nodes.decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "check_hallucination")
    
    # Conditional edge after hallucination check
    workflow.add_conditional_edges(
        "check_hallucination",
        nodes.decide_after_hallucination_check,
        {
            "verify": "verify",
            "retry": "increment"
        }
    )
    
    # Conditional edge after verification
    workflow.add_conditional_edges(
        "verify",
        nodes.decide_after_verification,
        {
            "end": END,
            "retry": "increment"
        }
    )
    
    # After incrementing, go back to rewrite
    workflow.add_edge("increment", "rewrite")
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def query_rag_agent(
    question: str,
    max_iterations: int = 3,
    vector_store_manager: Optional[VectorStoreManager] = None,
    persist_directory: str = "./chroma_db"
) -> Dict[str, Any]:
    """
    Query the RAG agent with a question.
    
    Args:
        question: User's question
        max_iterations: Maximum correction iterations
        vector_store_manager: Optional pre-initialized vector store manager
        persist_directory: Directory for vector store persistence
        
    Returns:
        Dictionary with final state including answer and workflow steps
    """
    # Create the graph
    app = create_rag_graph(vector_store_manager, persist_directory)
    
    # Initialize state
    initial_state: Dict[str, Any] = {
        "question": question,
        "rewritten_question": "",
        "documents": [],
        "generation": "",
        "iterations": 0,
        "max_iterations": max_iterations,
        "web_search_needed": False,
        "web_search_results": [],
        "relevant_docs_count": 0,
        "workflow_steps": [],
        "is_grounded": False,
        "is_answer_good": False
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    return final_state


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Create a test query
    result = query_rag_agent(
        question="What is machine learning?",
        max_iterations=3
    )
    
    logger.info("="*80)
    logger.info("FINAL ANSWER:")
    logger.info("="*80)
    logger.info(result["generation"])
    logger.info("="*80)
    logger.info("WORKFLOW STEPS:")
    logger.info("="*80)
    for step in result["workflow_steps"]:
        logger.info(f"- {step}")

