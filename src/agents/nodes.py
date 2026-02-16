"""Node implementations for the RAG agent graph."""


from typing import List, Optional, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
from opentelemetry import trace

from src.agents.state import RAGState
from src.core.vector_store import VectorStoreManager
from src.core.semantic_cache import SemanticCache
from src.core.retriever import AdvancedRetriever
from src.corrective.query_rewriter import QueryRewriter
from src.corrective.relevance_grader import RelevanceGrader
from src.corrective.hallucination_checker import HallucinationChecker
from src.corrective.answer_verifier import AnswerVerifier
from src.core.logging_config import get_logger
from src.core.logging_config import get_logger
from src.core.telemetry import get_tracer, get_meter

load_dotenv()

logger = get_logger(__name__)
logger = get_logger(__name__)
tracer = get_tracer(__name__)
meter = get_meter(__name__)

# Metrics
cache_hits_counter = meter.create_counter(
    "rag.cache.hits",
    description="Number of semantic cache hits",
    unit="1",
)
cache_misses_counter = meter.create_counter(
    "rag.cache.misses",
    description="Number of semantic cache misses",
    unit="1",
)
token_usage_counter = meter.create_counter(
    "rag.llm.tokens",
    description="Number of tokens used by LLM",
    unit="1",
)

# Initialize components (these will be set by the graph)
vector_store_manager: Optional[VectorStoreManager] = None
semantic_cache: Optional[SemanticCache] = None
retriever: Optional[AdvancedRetriever] = None
query_rewriter: Optional[QueryRewriter] = None
relevance_grader: Optional[RelevanceGrader] = None
hallucination_checker: Optional[HallucinationChecker] = None
answer_verifier: Optional[AnswerVerifier] = None
llm: Optional[ChatOllama] = None


def initialize_components(vsm: VectorStoreManager) -> None:
    """Initialize all components with the vector store manager."""
    global vector_store_manager, semantic_cache, retriever, query_rewriter, relevance_grader
    global hallucination_checker, answer_verifier, llm
    
    vector_store_manager = vsm
    semantic_cache = SemanticCache(persist_directory=vsm.persist_directory)
    retriever = AdvancedRetriever(vector_store_manager, k=4)
    query_rewriter = QueryRewriter()
    relevance_grader = RelevanceGrader()
    hallucination_checker = HallucinationChecker()
    answer_verifier = AnswerVerifier()
    llm = ChatOllama(model="qwen2.5:14b", temperature=0.7)


def check_cache(state: RAGState) -> RAGState:
    """
    Node to check for cached answers.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with cache result
    """
    with tracer.start_as_current_span("rag.check_cache") as span:
        logger.info("---CHECKING CACHE---")
        question = state["question"]
        span.set_attribute("rag.question", question[:200])
        
        cache_result = semantic_cache.check_cache(question)
        # cache_result = None # FORCE MISS FOR BENCHMARK
        
        if cache_result:
            logger.info("Cache hit found")
            state["generation"] = cache_result["answer"]
            state["cache_hit"] = True
            state["workflow_steps"].append(f"Cache HIT (score={cache_result['similarity_score']:.4f})")
            cache_hits_counter.add(1)
        else:
            logger.info("Cache miss")
            state["cache_hit"] = False
            state["workflow_steps"].append("Cache MISS")
            cache_misses_counter.add(1)
        
        span.set_attribute("rag.cache_hit", state["cache_hit"])
        
    return state


def rewrite_query(state: RAGState) -> RAGState:
    """
    Node to rewrite the query for better retrieval.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with rewritten question
    """
    with tracer.start_as_current_span("rag.rewrite_query") as span:
        logger.info("---REWRITING QUERY---")
        question = state["question"]
        span.set_attribute("rag.question", question[:200])
        
        rewritten_question = query_rewriter.rewrite(question)
        
        # Record token usage (QueryRewriter uses an internal chain/LLM invoke)
        # Note: QueryRewriter class needs to return or expose response metadata to capture tokens.
        # Assuming for now we might miss this or need to update QueryRewriter.
        # Let's inspect QueryRewriter in a separate step or assume we can't get it easily without refactoring QueryRewriter.
        # Actually, let's keep it simple for now and only instrument the direct LLM calls or if QueryRewriter exposes it.
        # Checking QueryRewriter code is needed. For now, I'll allow it to be missed in this pass if it's encapsulated.
        
        logger.debug(f"Original query: {question}")
        logger.info(f"Rewritten query: {rewritten_question}")
        
        state["rewritten_question"] = rewritten_question
        state["workflow_steps"].append(f"Query rewritten: {rewritten_question}")
    
    return state


def retrieve_documents(state: RAGState) -> RAGState:
    """
    Node to retrieve documents from the vector store.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with retrieved documents
    """
    with tracer.start_as_current_span("rag.retrieve_documents") as span:
        logger.info("---RETRIEVING DOCUMENTS---")
        question = state.get("rewritten_question", state["question"])
        span.set_attribute("rag.question", question[:200])
        
        documents = retriever.retrieve(question)
        logger.info(f"Retrieved {len(documents)} documents")
        
        state["documents"] = documents
        state["workflow_steps"].append(f"Retrieved {len(documents)} documents")
        span.set_attribute("rag.document_count", len(documents))
    
    return state


def grade_documents(state: RAGState) -> RAGState:
    """
    Node to grade document relevance.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with filtered documents
    """
    with tracer.start_as_current_span("rag.grade_documents") as span:
        logger.info("---GRADING DOCUMENTS---")
        question = state["question"]
        documents = state["documents"]
        span.set_attribute("rag.question", question[:200])
        span.set_attribute("rag.input_document_count", len(documents))
        
        relevant_docs, irrelevant_docs = relevance_grader.grade_documents(documents, question)
        
        state["documents"] = relevant_docs
        state["relevant_docs_count"] = len(relevant_docs)
        state["web_search_needed"] = len(relevant_docs) == 0
        state["workflow_steps"].append(
            f"Graded: {len(relevant_docs)} relevant, {len(irrelevant_docs)} irrelevant"
        )
        span.set_attribute("rag.relevant_count", len(relevant_docs))
        span.set_attribute("rag.irrelevant_count", len(irrelevant_docs))
    
    return state


def web_search(state: RAGState) -> RAGState:
    """
    Node to perform web search using Tavily.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with web search results
    """
    with tracer.start_as_current_span("rag.web_search") as span:
        logger.info("---WEB SEARCH---")
        question = state.get("rewritten_question", state["question"])
        span.set_attribute("rag.question", question[:200])
        
        try:
            from tavily import TavilyClient
            import os
            
            tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            response = tavily_client.search(query=question, max_results=3)
            
            # Convert Tavily results to Documents
            web_docs = []
            for result in response.get("results", []):
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("url", ""),
                        "title": result.get("title", ""),
                        "type": "web_search"
                    }
                )
                web_docs.append(doc)
            
            logger.info(f"Web search returned {len(web_docs)} results")
            state["web_search_results"] = web_docs
            state["documents"] = web_docs  # Use web results as documents
            state["workflow_steps"].append(f"Web search: {len(web_docs)} results")
            span.set_attribute("rag.web_result_count", len(web_docs))
            
        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            state["web_search_results"] = []
            state["workflow_steps"].append(f"Web search failed: {str(e)}")
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.record_exception(e)
    
    return state


def generate_answer(state: RAGState) -> RAGState:
    """
    Node to generate an answer based on retrieved documents.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with generated answer
    """
    with tracer.start_as_current_span("rag.generate_answer") as span:
        logger.info("---GENERATING ANSWER---")
        question = state["question"]
        documents = state["documents"]
        span.set_attribute("rag.question", question[:200])
        span.set_attribute("rag.context_doc_count", len(documents))
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, say that you don't know.
            Use three sentences maximum and keep the answer concise.
            
            Question: {question}
            
            Context: {context}
            
            Answer:"""
        )
        
        chain = prompt | llm | StrOutputParser()
        

        # We need to access the raw response to get metadata.
        # StrOutputParser strips it. We should invoke the LLM directly or use a chain that preserves it if we want tokens.
        # Ideally, we bind the tool or use introspection.
        # However, ChatOllama returns a AIMessage which has usage_metadata.
        # But `chain.invoke` with StrOutputParser returns a string.
        # To get tokens, we need to bypass StrOutputParser or access the raw message.
        
        # Refactoring to get token usage:
        msg = prompt.invoke({"question": question, "context": context})
        response = llm.invoke(msg)
        generation = response.content
        
        # Record tokens
        if response.usage_metadata:
             input_tokens = response.usage_metadata.get("input_tokens", 0)
             output_tokens = response.usage_metadata.get("output_tokens", 0)
             token_usage_counter.add(input_tokens, {"type": "input", "model": llm.model})
             token_usage_counter.add(output_tokens, {"type": "output", "model": llm.model})
        
        logger.info(f"Generated answer: {generation[:100]}...")
        logger.debug(f"Full answer: {generation}")
        
        state["generation"] = generation
        state["workflow_steps"].append("Answer generated")
    
    return state


def check_hallucination(state: RAGState) -> RAGState:
    """
    Node to check if the generated answer is grounded in documents.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state (hallucination check recorded in workflow_steps)
    """
    with tracer.start_as_current_span("rag.check_hallucination") as span:
        logger.info("---CHECKING HALLUCINATION---")
        documents = state["documents"]
        generation = state["generation"]
        
        is_grounded = hallucination_checker.check(documents, generation)
        
        state["workflow_steps"].append(
            f"Hallucination check: {'grounded' if is_grounded else 'not grounded'}"
        )
        
        # Store result in state for routing
        state["is_grounded"] = is_grounded
        span.set_attribute("rag.is_grounded", is_grounded)
    
    return state


def verify_answer(state: RAGState) -> RAGState:
    """
    Node to verify if the answer addresses the question.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state (verification recorded in workflow_steps)
    """
    with tracer.start_as_current_span("rag.verify_answer") as span:
        logger.info("---VERIFYING ANSWER---")
        question = state["question"]
        generation = state["generation"]
        
        is_good = answer_verifier.verify(question, generation)
        
        state["workflow_steps"].append(
            f"Answer verification: {'passed' if is_good else 'needs improvement'}"
        )
        
        # Store result in state for routing
        state["is_answer_good"] = is_good
        span.set_attribute("rag.is_answer_good", is_good)
        
        # Update cache if answer is verified and good
        if is_good:
            try:
                semantic_cache.update_cache(
                    query=question,
                    answer=generation,
                    rewritten_query=state.get("rewritten_question", "")
                )
                state["workflow_steps"].append("Cache updated")
            except Exception as e:
                logger.error(f"Failed to update cache: {e}")
    
    return state


def increment_iteration(state: RAGState) -> RAGState:
    """
    Node to increment the iteration counter.
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with incremented iteration
    """
    with tracer.start_as_current_span("rag.increment_iteration") as span:
        state["iterations"] = state.get("iterations", 0) + 1
        logger.info(f"---ITERATION {state['iterations']}/{state['max_iterations']}---")
        state["workflow_steps"].append(f"Iteration {state['iterations']}")
        span.set_attribute("rag.iteration", state["iterations"])
        span.set_attribute("rag.max_iterations", state["max_iterations"])
    
    return state


# Conditional edge functions

def decide_to_generate(state: RAGState) -> str:
    """
    Determine whether to generate an answer or perform web search.
    
    Args:
        state: Current RAG state
        
    Returns:
        Next node name
    """
    if state.get("web_search_needed", False):
        logger.info("Decision: No relevant documents found, performing web search")
        return "web_search"
    else:
        logger.info("Decision: Relevant documents found, proceeding to generation")
        return "generate"


def decide_after_hallucination_check(state: RAGState) -> str:
    """
    Determine next step after hallucination check.
    
    Args:
        state: Current RAG state
        
    Returns:
        Next node name
    """
    is_grounded = state.get("is_grounded", False)
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if is_grounded:
        logger.info("Decision: Answer is grounded, proceeding to verification")
        return "verify"
    elif iterations < max_iterations:
        logger.info("Decision: Answer not grounded, retrying")
        return "retry"
    else:
        logger.warning("Decision: Max iterations reached, proceeding to verification anyway")
        return "verify"


def decide_after_verification(state: RAGState) -> str:
    """
    Determine whether to end or retry after answer verification.
    
    Args:
        state: Current RAG state
        
    Returns:
        Next node name
    """
    is_good = state.get("is_answer_good", True)
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)
    
    if is_good:
        logger.info("Decision: Answer is good, ending workflow")
        return "end"
    elif iterations < max_iterations:
        logger.info("Decision: Answer needs improvement, retrying")
        return "retry"
    else:
        logger.warning("Decision: Max iterations reached, ending workflow")
        return "end"
