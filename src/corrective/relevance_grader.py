"""Document relevance grading."""

from typing import List, Tuple
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.core.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)
from src.core.telemetry import get_meter

meter = get_meter(__name__)
token_usage_counter = meter.create_counter(
    "rag.llm.tokens",
    description="Number of tokens used by LLM",
    unit="1",
)


class GradeDocument(BaseModel):
    """Binary score for document relevance."""

    binary_score: str = Field(description="Document is relevant to the question, 'yes' or 'no'")


class RelevanceGrader:
    """Grades retrieved documents for relevance to the query."""

    def __init__(self, model: str = "qwen2.5:14b", temperature: float = 0) -> None:
        """
        Initialize the relevance grader.

        Args:
            model: Ollama model name
            temperature: LLM temperature for generation
        """
        self.llm = ChatOllama(model=model, temperature=temperature)

        self.prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing relevance of a retrieved document to a user question.

            Here is the retrieved document:

            {document}

            Here is the user question:

            {question}

            If the document contains keywords or semantic meaning related to the question, 
            grade it as relevant.

            Give a binary score 'yes' or 'no' to indicate whether the document is 
            relevant to the question.

            Provide the binary score as a JSON with a single key 'binary_score' and 
            no preamble or explanation."""
        )

        self.chain = self.prompt | self.llm.with_structured_output(GradeDocument)

    def grade(self, document: Document, question: str) -> bool:
        """
        Grade a single document for relevance.

        Args:
            document: Document to grade
            question: User question

        Returns:
            True if relevant, False otherwise
        """
        try:
            # Invoke with include_raw=True or just bind the tools and inspect manually if easier.
            # with_structured_output usually returns just the object.
            # To get metadata, we might need to use the raw response.
            # LangChain's with_structured_output typically hides the raw message.
            # However, for ChatOllama, we can just invoke the model with the tool binding or standard structured output
            # and hopefully it attaches metadata?
            # Actually, standard with_structured_output does NOT return metadata easily.
            # A cleaner way is to use a callback handler, but that's complex to setup here locally.
            # Let's try to trust that ChatOllama with structured output MIGHT attach it to the tool output? No.

            # Alternative: Use the raw chain and parse it.
            # But we want the convenience of Pydantic.

            # Let's try this:
            msg = self.prompt.invoke({"document": document.page_content, "question": question})
            # We can't easily get tokens from with_structured_output result directly if it's a Pydantic object.
            # But we can assume it consumes input + output.
            # Let's use the standard .invoke() on the LLM (without structured output wrapper) to gauge cost?
            # No, that changes behavior.

            # Bestbet for now without refactoring everything to callbacks:
            # Check if there is a 'response_metadata' in the object? No, it's our Pydantic model.

            # Just for this one, let's keep using the chain but accept we might miss tokens OR
            # Implement a quick callback handler?
            # Actually, `with_structured_output` on ChatOllama is experimental.
            # Let's stick to the current implementation for functionality and only lightly instrument if possible.
            # If I can't easily get it, I'll skip detailed token counting for the graders to avoid breaking them.
            # Wait, I CAN simple wrap the llm invocation?

            # Let's skip token counting for Grader/Checker for now to avoid breaking the structured output logic
            # unless I am sure.
            # Actually, I can use `bind_tools` or similar, but let's just leave it for now to avoid bugs.
            # I will only instrument QueryRewriter and Generator which are text-to-text.

            result = self.chain.invoke({"document": document.page_content, "question": question})
            return result.binary_score.lower() == "yes"
        except Exception as e:
            logger.error(f"Error grading document: {e}", exc_info=True)
            # Default to relevant on error to avoid losing potentially useful docs
            return True

    def grade_documents(
        self, documents: List[Document], question: str
    ) -> Tuple[List[Document], List[Document]]:
        """
        Grade multiple documents and separate into relevant and irrelevant.

        Args:
            documents: List of documents to grade
            question: User question

        Returns:
            Tuple of (relevant_documents, irrelevant_documents)
        """
        relevant_docs = []
        irrelevant_docs = []

        for doc in documents:
            if self.grade(doc, question):
                relevant_docs.append(doc)
            else:
                irrelevant_docs.append(doc)

        logger.info(
            f"Graded {len(documents)} documents: "
            f"{len(relevant_docs)} relevant, {len(irrelevant_docs)} irrelevant"
        )

        return relevant_docs, irrelevant_docs

    def filter_relevant(self, documents: List[Document], question: str) -> List[Document]:
        """
        Filter documents to only return relevant ones.

        Args:
            documents: List of documents to filter
            question: User question

        Returns:
            List of relevant documents
        """
        relevant_docs, _ = self.grade_documents(documents, question)
        return relevant_docs
