"""Gradio web interface for the Corrective & Adaptive RAG Agent."""

import gradio as gr
from typing import List, Tuple, Optional, Any
import os
from dotenv import load_dotenv

from src.agents.rag_graph import create_rag_graph
from src.core.vector_store import VectorStoreManager
from src.core.telemetry import setup_telemetry
from typing import List, Tuple, Optional, Any, Dict
import json

load_dotenv()

# Initialize Telemetry
setup_telemetry()
if os.getenv("OTEL_ENABLED", "false").lower() == "true":
    try:
        from prometheus_client import start_http_server

        # Start Prometheus metrics server on port 8000 to match docker-compose config
        start_http_server(8000)
        print("Prometheus metrics server started on port 8000")
    except Exception as e:
        print(f"Failed to start metrics server: {e}")

# Global vector store manager
vector_store_manager = VectorStoreManager(persist_directory="./chroma_db")


def format_workflow_steps(steps: List[str]) -> str:
    """Format workflow steps for display."""
    if not steps:
        return "No workflow steps recorded."

    formatted = "### Workflow Steps:\n\n"
    for i, step in enumerate(steps, 1):
        formatted += f"{i}. {step}\n"

    return formatted


def format_documents(documents: List[Any]) -> str:
    """Format retrieved documents for display."""
    if not documents:
        return "No documents retrieved."

    formatted = "### Retrieved Documents:\n\n"
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        formatted += f"**Document {i}** (Source: {source})\n"
        formatted += f"{doc.page_content[:300]}...\n\n"
        formatted += "---\n\n"

    return formatted


def query_agent(question: str, max_iterations: int, history: List[Tuple[str, str]]) -> Any:
    """
    Query the RAG agent with streaming updates.

    Args:
        question: User's question
        max_iterations: Maximum correction iterations
        history: Chat history

    Yields:
        Tuple of (updated_question, workflow_info, documents_info, updated_history)
    """
    if not question.strip():
        yield "", "Please enter a question.", "", history
        return

    try:
        # Create the graph
        app_graph = create_rag_graph(vector_store_manager=vector_store_manager)

        # Initialize state
        state: Dict[str, Any] = {
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
            "is_answer_good": False,
        }

        # Update history with a placeholder for the answer
        history.append((question, "⏳ Thinking..."))
        yield "", "Initializing workflow...", "Retrieving documents...", history

        # Run the workflow and yield updates
        for output in app_graph.stream(state):
            if not output:
                continue

            # Each output is a dict mapping node names to state updates
            node_name = list(output.keys())[0]
            state = output[node_name]

            # Extract results
            answer = state.get("generation", "⏳ Processing...")
            workflow_steps = state.get("workflow_steps", [])
            documents = state.get("documents", [])
            iterations = state.get("iterations", 0)
            relevant_docs = state.get("relevant_docs_count", 0)

            # Format workflow information
            workflow_info = f"### Query Analysis (Step: {node_name.replace('_', ' ').title()})\n\n"
            workflow_info += f"- **Original Question**: {question}\n"
            workflow_info += (
                f"- **Rewritten Question**: {state.get('rewritten_question', 'Pending...')}\n"
            )
            workflow_info += f"- **Iterations**: {iterations}/{max_iterations}\n"
            workflow_info += f"- **Relevant Documents**: {relevant_docs}\n"
            workflow_info += (
                f"- **Web Search Used**: {'Yes' if state.get('web_search_results') else 'No'}\n\n"
            )
            workflow_info += format_workflow_steps(workflow_steps)

            # Format documents
            documents_info = format_documents(documents)

            # Update chat history (replace the placeholder or append/update last)
            if history and history[-1][0] == question:
                history[-1] = (question, answer if answer else "⏳ Generating answer...")

            yield "", workflow_info, documents_info, history

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if history and history[-1][1] == "⏳ Thinking...":
            history[-1] = (question, error_msg)
        else:
            history.append((question, error_msg))
        yield "", error_msg, "", history


def upload_documents(files: Optional[List[Any]], chunk_size: int, chunk_overlap: int) -> str:
    """
    Upload and ingest documents into the vector store.

    Args:
        files: List of uploaded files
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Status message
    """
    if not files:
        return "No files uploaded."

    try:
        file_paths = [file.name for file in files]
        ids = vector_store_manager.ingest_files(
            file_paths=file_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        return f"✓ Successfully ingested {len(files)} files into {len(ids)} chunks"

    except Exception as e:
        return f"✗ Error ingesting files: {str(e)}"


def upload_text(text: str, chunk_size: int, chunk_overlap: int) -> str:
    """
    Upload and ingest raw text into the vector store.

    Args:
        text: Text to ingest
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Status message
    """
    if not text.strip():
        return "No text provided."

    try:
        ids = vector_store_manager.ingest_text_documents(
            texts=[text],
            metadatas=[{"source": "manual_input"}],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return f"✓ Successfully ingested text into {len(ids)} chunks"

    except Exception as e:
        return f"✗ Error ingesting text: {str(e)}"


def get_vector_store_stats() -> str:
    """Get vector store statistics."""
    try:
        stats = vector_store_manager.get_stats()
        return f"""### Vector Store Statistics

- **Document Count**: {stats.get("document_count", 0)} chunks
- **Storage Location**: {stats.get("persist_directory", "N/A")}
"""
    except Exception as e:
        return f"Error getting stats: {str(e)}"


def clear_vector_store() -> str:
    """Clear all documents from the vector store."""
    try:
        vector_store_manager.clear()
        return "✓ Vector store cleared successfully"
    except Exception as e:
        return f"✗ Error clearing vector store: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Corrective & Adaptive RAG Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Corrective & Adaptive RAG Agent

    An advanced Retrieval-Augmented Generation system with adaptive corrective mechanisms:
    - **Query Rewriting**: Reformulates queries for better retrieval
    - **Relevance Grading**: Filters irrelevant documents
    - **Hallucination Checking**: Verifies answer accuracy
    - **Answer Verification**: Ensures quality responses
    - **Iterative Refinement**: Loops back when needed
    """)

    with gr.Tabs():
        # Query Tab
        with gr.Tab("💬 Query"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Conversation", height=400, show_label=True)

                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Ask a Question",
                            placeholder="Enter your question here...",
                            lines=2,
                        )

                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary")
                        clear_btn = gr.Button("Clear Chat")

                    with gr.Accordion("⚙️ Settings", open=False):
                        max_iterations = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Max Iterations",
                            info="Maximum number of correction loops",
                        )

                with gr.Column(scale=1):
                    workflow_output = gr.Markdown(
                        label="Workflow Information", value="Workflow steps will appear here..."
                    )

            with gr.Row():
                documents_output = gr.Markdown(
                    label="Retrieved Documents", value="Retrieved documents will appear here..."
                )

            # Chat state
            chat_history = gr.State([])

            # Event handlers
            submit_btn.click(
                fn=query_agent,
                inputs=[question_input, max_iterations, chat_history],
                outputs=[question_input, workflow_output, documents_output, chatbot],
            )

            question_input.submit(
                fn=query_agent,
                inputs=[question_input, max_iterations, chat_history],
                outputs=[question_input, workflow_output, documents_output, chatbot],
            )

            clear_btn.click(
                fn=lambda: ([], "", ""), outputs=[chatbot, workflow_output, documents_output]
            )

        # Ingestion Tab
        with gr.Tab("📁 Ingest Documents"):
            gr.Markdown("### Upload Documents to Vector Store")

            with gr.Tabs():
                with gr.Tab("Upload Files"):
                    file_upload = gr.File(
                        label="Upload Files",
                        file_count="multiple",
                        file_types=[".txt", ".md", ".pdf"],
                    )

                    with gr.Row():
                        file_chunk_size = gr.Slider(
                            minimum=100, maximum=2000, value=1000, step=100, label="Chunk Size"
                        )
                        file_chunk_overlap = gr.Slider(
                            minimum=0, maximum=500, value=200, step=50, label="Chunk Overlap"
                        )

                    file_upload_btn = gr.Button("Upload Files", variant="primary")
                    file_status = gr.Textbox(label="Upload Status", lines=2)

                    file_upload_btn.click(
                        fn=upload_documents,
                        inputs=[file_upload, file_chunk_size, file_chunk_overlap],
                        outputs=file_status,
                    )

                with gr.Tab("Paste Text"):
                    text_input = gr.Textbox(
                        label="Paste Text", placeholder="Paste your text here...", lines=10
                    )

                    with gr.Row():
                        text_chunk_size = gr.Slider(
                            minimum=100, maximum=2000, value=1000, step=100, label="Chunk Size"
                        )
                        text_chunk_overlap = gr.Slider(
                            minimum=0, maximum=500, value=200, step=50, label="Chunk Overlap"
                        )

                    text_upload_btn = gr.Button("Ingest Text", variant="primary")
                    text_status = gr.Textbox(label="Ingestion Status", lines=2)

                    text_upload_btn.click(
                        fn=upload_text,
                        inputs=[text_input, text_chunk_size, text_chunk_overlap],
                        outputs=text_status,
                    )

        # Stats & Admin Tab
        with gr.Tab("📊 Statistics & Admin"):
            gr.Markdown("### Vector Store Management")

            with gr.Row():
                stats_btn = gr.Button("Refresh Stats", variant="secondary")
                clear_store_btn = gr.Button("Clear Vector Store", variant="stop")

            stats_output = gr.Markdown(label="Statistics")
            clear_status = gr.Textbox(label="Clear Status", lines=2)

            stats_btn.click(fn=get_vector_store_stats, outputs=stats_output)

            clear_store_btn.click(fn=clear_vector_store, outputs=clear_status)

            # Load stats on tab load
            demo.load(fn=get_vector_store_stats, outputs=stats_output)

    gr.Markdown("""
    ---
    ### 📖 How to Use

    1. **Ingest Documents**: Upload files or paste text in the "Ingest Documents" tab
    2. **Ask Questions**: Go to the "Query" tab and ask questions about your documents
    3. **View Workflow**: See the agent's reasoning process in the workflow panel
    4. **Adjust Settings**: Configure max iterations and other parameters
    5. **Monitor Stats**: Check vector store statistics in the "Statistics & Admin" tab

    The agent will automatically:
    - Rewrite queries for better retrieval
    - Grade document relevance
    - Check for hallucinations
    - Verify answer quality
    - Iterate until a good answer is found (up to max iterations)
    """)


def launch_gradio_app(
    server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False
) -> None:
    """
    Launch the Gradio web interface.

    Args:
        server_name: Server host
        server_port: Server port
        share: Whether to create a public link
    """
    demo.launch(server_name=server_name, server_port=server_port, share=share)


if __name__ == "__main__":
    launch_gradio_app()
