"""Answer quality verification."""


from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.core.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)


class GradeAnswer(BaseModel):
    """Binary score for answer quality."""

    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


class AnswerVerifier:
    """Verifies that generated answers properly address the user's question."""

    def __init__(self, model: str = "qwen2.5:14b", temperature: float = 0) -> None:
        """
        Initialize the answer verifier.

        Args:
            model: Ollama model name
            temperature: LLM temperature for generation
        """
        self.llm = ChatOllama(model=model, temperature=temperature)

        self.prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing whether an answer addresses / resolves a question.

            Here is the user question:

            {question}

            Here is the generated answer:

            {generation}

            Give a binary score 'yes' or 'no' to indicate whether the answer addresses
            and resolves the question.

            Score 'yes' if the answer:
            - Directly addresses the question
            - Provides relevant information
            - Is reasonably complete

            Score 'no' if the answer:
            - Misses the point of the question
            - Is off-topic or irrelevant
            - Is too vague or incomplete to be useful

            Provide the binary score as a JSON with a single key 'binary_score' and
            no preamble or explanation."""
        )

        self.chain = self.prompt | self.llm.with_structured_output(GradeAnswer)

    def verify(self, question: str, generation: str) -> bool:
        """
        Verify if an answer properly addresses the question.

        Args:
            question: User's original question
            generation: Generated answer to verify

        Returns:
            True if answer is good, False if needs improvement
        """
        try:
            result = self.chain.invoke({"question": question, "generation": generation})

            is_good = result.binary_score.lower() == "yes"

            if is_good:
                logger.info("Answer properly addresses the question")
            else:
                logger.warning("Answer needs improvement")

            return is_good
        except Exception as e:
            logger.error(f"Error verifying answer: {e}", exc_info=True)
            # Default to accepting on error
            return True

    def verify_with_feedback(self, question: str, generation: str) -> tuple[bool, str]:
        """
        Verify answer with detailed feedback.

        Args:
            question: User's original question
            generation: Generated answer to verify

        Returns:
            Tuple of (is_good, feedback)
        """
        feedback_prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing answer quality.

            User question:
            {question}

            Generated answer:
            {generation}

            Evaluate whether the answer properly addresses the question. Provide:
            1. Strengths (if any)
            2. Weaknesses (if any)
            3. Suggestions for improvement (if needed)
            4. Final verdict: 'GOOD' or 'NEEDS_IMPROVEMENT'

            Format:
            Strengths: [list]
            Weaknesses: [list]
            Suggestions: [list]
            Verdict: [GOOD or NEEDS_IMPROVEMENT]"""
        )

        try:
            chain = feedback_prompt | self.llm
            result = chain.invoke({"question": question, "generation": generation})

            response = result.content

            # Extract verdict
            is_good = (
                "GOOD" in response.split("Verdict:")[-1].upper()
                and "NEEDS_IMPROVEMENT" not in response.split("Verdict:")[-1].upper()
            )

            return is_good, response
        except Exception as e:
            logger.error(f"Error verifying answer with feedback: {e}", exc_info=True)
            return True, f"Error: {str(e)}"

    def suggest_improvements(self, question: str, generation: str) -> str:
        """
        Get specific suggestions for improving an answer.

        Args:
            question: User's original question
            generation: Generated answer to improve

        Returns:
            Improvement suggestions
        """
        improvement_prompt = ChatPromptTemplate.from_template(
            """You are an expert at improving answers to questions.

            User question:
            {question}

            Current answer:
            {generation}

            Provide 3-5 specific, actionable suggestions for improving this answer.
            Focus on:
            - Addressing gaps in the response
            - Adding relevant details
            - Improving clarity
            - Ensuring completeness

            Suggestions:"""
        )

        try:
            chain = improvement_prompt | self.llm
            result = chain.invoke({"question": question, "generation": generation})

            return result.content.strip()
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}", exc_info=True)
            return "Unable to generate suggestions."
