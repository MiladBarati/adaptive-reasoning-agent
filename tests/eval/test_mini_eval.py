"""Integration mini-eval suite for Pull Requests."""

import pytest
from src.corrective.answer_verifier import AnswerVerifier

@pytest.fixture(scope="module")
def verifier():
    """Provides a fresh AnswerVerifier instance."""
    # We allow the Action to inject a smaller model via LLM_MODEL env var if needed.
    return AnswerVerifier()


@pytest.mark.integration
class TestMiniEval:
    """Mini-evaluation suite to ensure the agent's logic hasn't regressed."""

    def test_verifier_can_detect_good_answer(self, verifier):
        """Test that the AnswerVerifier correctly identifies a high-quality answer."""
        question = "How do I start a local backend server in Python?"
        generation = (
            "You can start a simple local backend server using Python's built-in http.server "
            "module by running `python -m http.server 8000` in your terminal. This will serve "
            "files from your current directory on port 8000."
        )

        is_good = verifier.verify(question, generation)
        assert is_good is True, "Verifier falsely rejected a perfectly good answer."

    def test_verifier_can_detect_bad_answer(self, verifier):
        """Test that the AnswerVerifier correctly identifies a poor or irrelevant answer."""
        question = "What is the capital of France?"
        generation = "I like eating cheese."

        is_good = verifier.verify(question, generation)
        assert is_good is False, "Verifier falsely accepted a completely irrelevant answer."

    def test_verifier_can_detect_vague_answer(self, verifier):
        """Test that a vague/incomplete answer is rejected."""
        question = "How do you explain the architectural concept of microservices?"
        generation = "It is when you make things small."

        is_good = verifier.verify(question, generation)
        assert is_good is False, "Verifier falsely accepted a vague and incomplete answer."
