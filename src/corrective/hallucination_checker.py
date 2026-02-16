"""Hallucination detection for generated answers."""

from typing import List, Tuple
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.core.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)


class GradeHallucination(BaseModel):
    """Binary score for hallucination check."""
    
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class HallucinationChecker:
    """Checks if generated answers are grounded in retrieved documents."""
    
    def __init__(self, model: str = "qwen2.5:14b", temperature: float = 0) -> None:
        """
        Initialize the hallucination checker.
        
        Args:
            model: Ollama model name
            temperature: LLM temperature for generation
        """
        self.llm = ChatOllama(model=model, temperature=temperature)
        
        self.prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing whether an answer is grounded in / supported by 
            a set of retrieved facts.
            
            Here are the retrieved facts:
            
            {documents}
            
            Here is the answer:
            
            {generation}
            
            Give a binary score 'yes' or 'no' to indicate whether the answer is grounded 
            in the retrieved facts.
            
            Score 'yes' if the answer is supported by the facts, even if not all details 
            are covered.
            Score 'no' if the answer contains information that contradicts or is not 
            supported by the facts.
            
            Provide the binary score as a JSON with a single key 'binary_score' and 
            no preamble or explanation."""
        )
        
        self.chain = self.prompt | self.llm.with_structured_output(GradeHallucination)
    
    def check(self, documents: List[Document], generation: str) -> bool:
        """
        Check if a generation is grounded in the documents.
        
        Args:
            documents: List of source documents
            generation: Generated answer to check
            
        Returns:
            True if grounded, False if hallucinated
        """
        try:
            # Concatenate document contents
            docs_content = "\n\n".join([doc.page_content for doc in documents])
            
            result = self.chain.invoke({
                "documents": docs_content,
                "generation": generation
            })
            
            is_grounded = result.binary_score.lower() == "yes"
            
            if is_grounded:
                logger.info("Answer is grounded in documents")
            else:
                logger.warning("Answer contains hallucinations")
            
            return is_grounded
        except Exception as e:
            logger.error(f"Error checking hallucination: {e}", exc_info=True)
            # Default to not grounded on error to be safe
            return False
    
    def check_with_reasoning(
        self,
        documents: List[Document],
        generation: str
    ) -> Tuple[bool, str]:
        """
        Check for hallucination with reasoning explanation.
        
        Args:
            documents: List of source documents
            generation: Generated answer to check
            
        Returns:
            Tuple of (is_grounded, reasoning)
        """
        reasoning_prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing whether an answer is grounded in retrieved facts.
            
            Retrieved facts:
            
            {documents}
            
            Answer to check:
            
            {generation}
            
            Analyze whether the answer is supported by the facts. Explain your reasoning 
            in 2-3 sentences, then provide a final verdict: 'GROUNDED' or 'HALLUCINATED'.
            
            Format your response as:
            Reasoning: [your analysis]
            Verdict: [GROUNDED or HALLUCINATED]"""
        )
        
        try:
            # Concatenate document contents
            docs_content = "\n\n".join([doc.page_content for doc in documents])
            
            chain = reasoning_prompt | self.llm
            result = chain.invoke({
                "documents": docs_content,
                "generation": generation
            })
            
            response = result.content
            
            # Extract verdict
            is_grounded = "GROUNDED" in response.upper() and "HALLUCINATED" not in response.split("Verdict:")[-1].upper()
            
            return is_grounded, response
        except Exception as e:
            logger.error(f"Error checking hallucination with reasoning: {e}", exc_info=True)
            return False, f"Error: {str(e)}"

