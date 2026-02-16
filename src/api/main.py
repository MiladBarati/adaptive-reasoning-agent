"""FastAPI backend for the Corrective & Adaptive RAG Agent."""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agents.rag_graph import (
    async_query_rag_agent,
    create_rag_graph,
)
from src.core.logging_config import get_logger
from src.core.telemetry import setup_telemetry
from src.core.vector_store import VectorStoreManager

load_dotenv()

logger = get_logger(__name__)

# Global vector store manager
vector_store_manager: VectorStoreManager | None = None


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event("startup"))
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize components on startup, clean up on shutdown."""
    global vector_store_manager
    # Initialize OpenTelemetry before anything else
    setup_telemetry()
    vector_store_manager = await asyncio.to_thread(
        VectorStoreManager, persist_directory="./chroma_db"
    )
    logger.info("Vector store manager initialized")
    yield
    # Shutdown logic (if any) goes here
    logger.info("Application shutting down")


# ---------------------------------------------------------------------------
# App & middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Corrective & Adaptive RAG Agent API",
    description="API for querying a RAG agent with corrective mechanisms",
    version="1.0.0",
    lifespan=lifespan,
)

# Get CORS origins from environment variable
cors_origins_str = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000",
)
cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]
logger.info(f"CORS allowed origins: {cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Auto-instrument FastAPI with OpenTelemetry (adds HTTP spans to all endpoints)
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPI OpenTelemetry instrumentation enabled")

    # Expose Prometheus metrics endpoint
    from prometheus_client import make_asgi_app

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
except ImportError:
    logger.debug(
        "opentelemetry-instrumentation-fastapi or prometheus_client not installed, skipping"
    )


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str = Field(..., description="User's question")
    max_iterations: int = Field(3, description="Maximum correction iterations", ge=1, le=10)
    stream: bool = Field(False, description="Enable streaming response")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str = Field(..., description="Generated answer")
    question: str = Field(..., description="Original question")
    rewritten_question: str = Field("", description="Rewritten question")
    workflow_steps: list[str] = Field([], description="List of workflow steps taken")
    iterations: int = Field(0, description="Number of iterations performed")
    relevant_docs_count: int = Field(0, description="Number of relevant documents found")
    web_search_used: bool = Field(False, description="Whether web search was used")


class IngestRequest(BaseModel):
    """Request model for text ingestion."""

    texts: list[str] = Field(..., description="List of texts to ingest")
    metadatas: list[dict[str, Any]] | None = Field(
        None, description="Optional metadata for each text"
    )
    chunk_size: int = Field(1000, description="Size of text chunks", ge=100, le=5000)
    chunk_overlap: int = Field(200, description="Overlap between chunks", ge=0, le=1000)


class IngestResponse(BaseModel):
    """Response model for ingestion."""

    message: str = Field(..., description="Status message")
    document_count: int = Field(..., description="Number of documents ingested")
    ids: list[str] = Field([], description="Document IDs")


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""

    document_count: int = Field(..., description="Total number of documents in vector store")
    persist_directory: str = Field(..., description="Vector store directory")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Corrective & Adaptive RAG Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store": ("initialized" if vector_store_manager else "not initialized"),
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG agent with a question.

    Args:
        request: Query request with question and parameters

    Returns:
        Query response with answer and metadata
    """
    if not vector_store_manager:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        # Run the RAG agent asynchronously (offloaded to a thread)
        result = await async_query_rag_agent(
            question=request.question,
            max_iterations=request.max_iterations,
            vector_store_manager=vector_store_manager,
        )

        return QueryResponse(
            answer=result.get("generation", "No answer generated"),
            question=result.get("question", request.question),
            rewritten_question=result.get("rewritten_question", ""),
            workflow_steps=result.get("workflow_steps", []),
            iterations=result.get("iterations", 0),
            relevant_docs_count=result.get("relevant_docs_count", 0),
            web_search_used=len(result.get("web_search_results", [])) > 0,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/query/stream", tags=["RAG"])
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """
    Query the RAG agent with streaming response.

    Args:
        request: Query request with question and parameters

    Returns:
        Server-sent events stream
    """
    if not vector_store_manager:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate server-sent events for streaming."""
        try:
            # Create the graph (fast, CPU-only)
            app_graph = create_rag_graph(vector_store_manager=vector_store_manager)

            # Initialize state
            initial_state: dict[str, Any] = {
                "question": request.question,
                "rewritten_question": "",
                "documents": [],
                "generation": "",
                "iterations": 0,
                "max_iterations": request.max_iterations,
                "web_search_needed": False,
                "web_search_results": [],
                "relevant_docs_count": 0,
                "workflow_steps": [],
                "is_grounded": False,
                "is_answer_good": False,
            }

            # Offload the synchronous stream() to a thread and
            # collect all state updates.
            def _run_stream() -> list[dict[str, Any]]:
                return list(app_graph.stream(initial_state))

            state_updates = await asyncio.to_thread(_run_stream)

            # Now yield SSE events from the collected updates
            final_state: dict[str, Any] = initial_state
            for state_update in state_updates:
                if state_update:
                    node_name = list(state_update.keys())[-1]
                    state = state_update[node_name]
                    final_state = state

                    event_data: dict[str, Any] = {
                        "type": "state_update",
                        "data": {
                            "workflow_steps": state.get("workflow_steps", []),
                            "iterations": state.get("iterations", 0),
                        },
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    await asyncio.sleep(0.01)

            # Send final answer
            final_event: dict[str, Any] = {
                "type": "final_answer",
                "data": {
                    "answer": final_state.get("generation", ""),
                    "question": final_state.get("question", ""),
                    "rewritten_question": final_state.get("rewritten_question", ""),
                    "workflow_steps": final_state.get("workflow_steps", []),
                    "iterations": final_state.get("iterations", 0),
                    "relevant_docs_count": final_state.get("relevant_docs_count", 0),
                },
            }
            yield f"data: {json.dumps(final_event)}\n\n"

        except Exception as e:
            error_event: dict[str, Any] = {
                "type": "error",
                "data": {"message": str(e)},
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_text(request: IngestRequest) -> IngestResponse:
    """
    Ingest text documents into the vector store.

    Args:
        request: Ingestion request with texts and parameters

    Returns:
        Ingestion response with status
    """
    if not vector_store_manager:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        ids = await asyncio.to_thread(
            vector_store_manager.ingest_text_documents,
            texts=request.texts,
            metadatas=request.metadatas,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        return IngestResponse(
            message=f"Successfully ingested {len(ids)} document chunks",
            document_count=len(ids),
            ids=ids,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting documents: {str(e)}",
        )


@app.post("/ingest/files", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_files(
    files: list[UploadFile] = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> IngestResponse:
    """
    Ingest files into the vector store.

    Args:
        files: List of files to upload
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Ingestion response with status
    """
    if not vector_store_manager:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        texts = []
        metadatas = []

        for file in files:
            content = await file.read()
            text = content.decode("utf-8")
            texts.append(text)
            metadatas.append({"source": file.filename, "filename": file.filename})

        ids = await asyncio.to_thread(
            vector_store_manager.ingest_text_documents,
            texts=texts,
            metadatas=metadatas,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return IngestResponse(
            message=(f"Successfully ingested {len(files)} files into {len(ids)} chunks"),
            document_count=len(ids),
            ids=ids,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting files: {str(e)}",
        )


@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats() -> StatsResponse:
    """
    Get vector store statistics.

    Returns:
        Statistics about the vector store
    """
    if not vector_store_manager:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        stats = await asyncio.to_thread(vector_store_manager.get_stats)

        return StatsResponse(
            document_count=stats.get("document_count", 0),
            persist_directory=stats.get("persist_directory", ""),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}",
        )


@app.delete("/clear", tags=["Admin"])
async def clear_vector_store() -> dict[str, str]:
    """
    Clear all documents from the vector store.

    Returns:
        Status message
    """
    if not vector_store_manager:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        await asyncio.to_thread(vector_store_manager.clear)
        return {"message": "Vector store cleared successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing vector store: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
