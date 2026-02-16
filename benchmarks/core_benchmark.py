"""
Core RAG Benchmark Suite.

Compares:
1. No RAG (Direct LLM generation)
2. Regular RAG (Retrieve -> Generate)
3. Corrective RAG (Full agentic pipeline)

Metrics:
- Latency (seconds)
- Answer Relevance (LLM-graded 0-1)
- Groundedness (LLM-graded 0-1)
"""

import asyncio
import json
import os
import statistics
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.agents.rag_graph import query_rag_agent
from src.core.retriever import AdvancedRetriever

# Import system components
from src.core.vector_store import VectorStoreManager

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

BENCHMARK_QUESTIONS = [
    {
        "id": 1,
        "question": "What is machine learning?",
        "expected_topics": ["algorithms", "data", "training", "predictions"],
        "difficulty": "easy",
    },
    {
        "id": 2,
        "question": "How does a Vector Database work?",
        "expected_topics": ["embeddings", "similarity search", "high-dimensional vectors"],
        "difficulty": "medium",
    },
    {
        "id": 3,
        "question": "Explain the difference between supervised and unsupervised learning.",
        "expected_topics": ["labeled data", "unlabeled data", "clustering", "classification"],
        "difficulty": "medium",
    },
    {
        "id": 4,
        "question": "What is the capital of France?",
        "expected_topics": ["Paris"],
        "difficulty": "easy_external_knowledge",
    },
    {
        "id": 5,
        "question": "Who won the 2024 Super Bowl?",
        "expected_topics": ["Kansas City Chiefs"],
        "difficulty": "hallucination_check",  # Not in docs
    },
]

# ---------------------------------------------------------------------------
# EVALUATION MODELS
# ---------------------------------------------------------------------------


class GradeResult(BaseModel):
    score: int = Field(description="Score from 0 to 1 (0=bad, 1=good)")
    reasoning: str = Field(description="Explanation for the score")


class BenchmarkResult(BaseModel):
    system: str
    question: str
    answer: str
    latency: float
    relevance_score: int
    groundedness_score: int


# ---------------------------------------------------------------------------
# BENCHMARK RUNNER
# ---------------------------------------------------------------------------


class Benchmarker:
    def __init__(self):
        # Use smaller model to avoid CUDA OOM errors during benchmark
        # Explicitly set num_ctx to 2048 to save VRAM
        self.llm = ChatOllama(model="qwen2.5:3b", temperature=0, num_ctx=2048)
        self.eval_llm = ChatOllama(model="qwen2.5:3b", temperature=0, num_ctx=2048)

        # Initialize RAG components
        self.vsm = VectorStoreManager(persist_directory="./chroma_db")
        self.retriever = AdvancedRetriever(self.vsm, k=4)

        # Evaluation Chains
        self.relevance_grader = (
            ChatPromptTemplate.from_template(
                """You are a grader assessing relevance of a retrieved answer to a user question.

                Question: {question}
                Answer: {answer}

                Does the answer address the question?
                Give a score of 1 if it answers the question, 0 if it does not or says "I don't know" when it should know.
                Provide reasoning.

                {format_instructions}"""
            )
            | self.eval_llm
            | JsonOutputParser(pydantic_object=GradeResult)
        )

        self.groundedness_grader = (
            ChatPromptTemplate.from_template(
                """You are a grader assessing if an answer is grounded in / supported by a set of facts.

                Facts: {context}
                Answer: {answer}

                Is the answer grounded in the facts?
                Give a score of 1 if yes, 0 if no (hallucinated).
                Provide reasoning.

                {format_instructions}"""
            )
            | self.eval_llm
            | JsonOutputParser(pydantic_object=GradeResult)
        )

    # --- SYSTEMS ---

    async def run_no_rag(self, question: str) -> dict[str, Any]:
        """Baseline: Direct LLM generation."""
        start = time.perf_counter()

        prompt = ChatPromptTemplate.from_template("Answer the question: {question}")
        chain = prompt | self.llm
        answer = await chain.ainvoke({"question": question})
        if hasattr(answer, "content"):
            answer = answer.content

        latency = time.perf_counter() - start
        return {"answer": answer, "latency": latency, "context": ""}

    async def run_regular_rag(self, question: str) -> dict[str, Any]:
        """Baseline: Retrieve -> Generate (No corrections)."""
        start = time.perf_counter()

        # 1. Retrieve
        docs = self.retriever.retrieve(question)
        context = "\n\n".join([d.page_content for d in docs])

        # 2. Generate
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based ONLY on the context provided.

            Context: {context}
            Question: {question}
            """
        )
        chain = prompt | self.llm
        answer = await chain.ainvoke({"question": question, "context": context})
        if hasattr(answer, "content"):
            answer = answer.content

        latency = time.perf_counter() - start
        return {"answer": answer, "latency": latency, "context": context}

    async def run_corrective_rag(self, question: str) -> dict[str, Any]:
        """Target: Full Corrective RAG Agent."""
        start = time.perf_counter()

        # We use the sync wrapper but in a thread to keep it fair/async if needed,
        # though for benchmarking script generic asyncio is fine.
        result = await asyncio.to_thread(
            query_rag_agent, question=question, max_iterations=3, vector_store_manager=self.vsm
        )

        answer = result["generation"]
        # Extract context from final documents for evaluation
        docs = result.get("documents", [])
        if docs and isinstance(docs[0], Document):
            context = "\n\n".join([d.page_content for d in docs])
        else:
            context = ""  # Fallback if no docs retrieved (e.g. web search or pure generation)

        latency = time.perf_counter() - start
        return {"answer": answer, "latency": latency, "context": context}

    # --- EVALUATION ---

    async def evaluate_answer(self, question: str, answer: str, context: str) -> dict[str, Any]:
        """Grade the answer for relevance and groundedness."""

        # Grade Relevance
        try:
            rel_grade = await self.relevance_grader.ainvoke(
                {
                    "question": question,
                    "answer": answer,
                    "format_instructions": "Return JSON with 'score' (int) and 'reasoning' (str).",
                }
            )
            rel_score = rel_grade["score"]
        except Exception as e:
            print(f"Error grading relevance: {e}")
            rel_score = 0

        # Grade Groundedness (Only if context exists, otherwise assumed 1 for No RAG if relevant?)
        # Actually for No RAG, "groundedness" checks if it matches the 'context' (which is empty).
        # So No RAG groundedness relative to *retrieved docs* is 0.
        # But here we want to know if it's hallucinated relative to *world knowledge*?
        # Standard RAG eval usually checks groundedness against *retrieved context*.

        if context:
            try:
                ground_grade = await self.groundedness_grader.ainvoke(
                    {
                        "context": context,
                        "answer": answer,
                        "format_instructions": "Return JSON with 'score' (int) and 'reasoning' (str).",
                    }
                )
                ground_score = ground_grade["score"]
            except Exception as e:
                print(f"Error grading groundedness: {e}")
                ground_score = 0
        else:
            # If no context (No RAG), groundedness is N/A or we can say 1 if we trust the model.
            # Let's mark as -1 (N/A) for No RAG context.
            ground_score = -1

        return {"relevance": rel_score, "groundedness": ground_score}

    # --- MAIN LOOP ---

    async def run(self):
        print(f"{'=' * 80}")
        print(f"CORE RAG BENCHMARK - {len(BENCHMARK_QUESTIONS)} Questions")
        print(f"{'=' * 80}")

        results = []

        for i, item in enumerate(BENCHMARK_QUESTIONS):
            q = item["question"]
            print(f"\n[Q{i + 1}] {q}")

            # 1. No RAG
            print("  Running No RAG...", end="", flush=True)
            try:
                res_no = await self.run_no_rag(q)
                # Small sleep to let VRAM cleanup
                await asyncio.sleep(2)

                eval_no = await self.evaluate_answer(q, res_no["answer"], res_no["context"])
                print(f" Done ({res_no['latency']:.2f}s)")
                results.append({**item, "system": "No RAG", **res_no, **eval_no})
            except Exception as e:
                print(f" Failed: {e}")

            await asyncio.sleep(2)

            # 2. Regular RAG
            print("  Running Regular RAG...", end="", flush=True)
            try:
                res_reg = await self.run_regular_rag(q)
                await asyncio.sleep(2)

                eval_reg = await self.evaluate_answer(q, res_reg["answer"], res_reg["context"])
                print(f" Done ({res_reg['latency']:.2f}s)")
                results.append({**item, "system": "Regular RAG", **res_reg, **eval_reg})
            except Exception as e:
                print(f" Failed: {e}")

            await asyncio.sleep(2)

            # 3. Corrective RAG
            print("  Running Corrective RAG...", end="", flush=True)
            try:
                res_corr = await self.run_corrective_rag(q)
                await asyncio.sleep(2)

                eval_corr = await self.evaluate_answer(q, res_corr["answer"], res_corr["context"])
                print(f" Done ({res_corr['latency']:.2f}s)")
                results.append({**item, "system": "Corrective RAG", **res_corr, **eval_corr})
            except Exception as e:
                print(f" Failed: {e}")

            # End of question sleep
            await asyncio.sleep(5)

        # Report
        self.print_report(results)
        self.save_results(results)

    def print_report(self, results):
        print(f"\n{'=' * 80}")
        print("BENCHMARK REPORT")
        print(f"{'=' * 80}")

        systems = ["No RAG", "Regular RAG", "Corrective RAG"]

        print(f"{'System':<20} | {'Latency (s)':<12} | {'Relevance':<10} | {'Groundedness':<12}")
        print("-" * 65)

        for sys in systems:
            sys_res = [r for r in results if r["system"] == sys]
            if not sys_res:
                continue

            avg_lat = statistics.mean([r["latency"] for r in sys_res])
            avg_rel = statistics.mean([r["relevance"] for r in sys_res])

            ground_scores = [r["groundedness"] for r in sys_res if r["groundedness"] != -1]
            avg_ground = statistics.mean(ground_scores) if ground_scores else 0.0

            print(f"{sys:<20} | {avg_lat:<12.2f} | {avg_rel:<10.2f} | {avg_ground:<12.2f}")

    def save_results(self, results):
        os.makedirs("benchmarks/results", exist_ok=True)
        path = f"benchmarks/results/core_benchmark_{int(time.time())}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {path}")


if __name__ == "__main__":
    b = Benchmarker()
    asyncio.run(b.run())
