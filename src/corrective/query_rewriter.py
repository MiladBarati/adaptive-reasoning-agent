"""Query rewriting for improved retrieval."""

from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from src.core.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)


class QueryRewriter:
    """Rewrites user queries to improve retrieval quality."""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0) -> None:
        """
        Initialize the query rewriter.
        
        Args:
            model: Groq model name
            temperature: LLM temperature for generation
        """
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_template(
            """You are an expert at reformulating search queries to improve document retrieval.
            
            Analyze the user's question and rewrite it to be more effective for semantic search.
            Make the query more specific, clear, and optimized for retrieving relevant information.
            
            Guidelines:
            - Keep the core intent of the question
            - Add relevant context or domain-specific terms
            - Make implicit information explicit
            - Break down complex questions if needed
            - Use clear, unambiguous language
            
            Original question: {question}
            
            Rewritten question:"""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def rewrite(self, question: str) -> str:
        """
        Rewrite a query for better retrieval.
        
        Args:
            question: Original user question
            
        Returns:
            Rewritten question
        """
        try:
            rewritten = self.chain.invoke({"question": question})
            return rewritten.strip()
        except Exception as e:
            logger.error(f"Error rewriting query: {e}", exc_info=True)
            return question  # Return original on error
    
    def rewrite_multiple(self, question: str, num_variations: int = 3) -> List[str]:
        """
        Generate multiple rewritten variations of a query.
        
        Args:
            question: Original user question
            num_variations: Number of variations to generate
            
        Returns:
            List of rewritten questions
        """
        multi_prompt = ChatPromptTemplate.from_template(
            """You are an expert at reformulating search queries to improve document retrieval.
            
            Generate {num} different variations of the user's question, each optimized for 
            semantic search from a different angle. Each variation should:
            - Maintain the core intent
            - Approach the question from a different perspective
            - Be clear and specific
            
            Original question: {question}
            
            Provide {num} variations, one per line:"""
        )
        
        try:
            chain = multi_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": question, "num": num_variations})
            
            # Parse variations
            variations = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line:
                    # Remove numbering if present
                    if line[0].isdigit() and '.' in line[:3]:
                        line = line.split('.', 1)[1].strip()
                    if line:
                        variations.append(line)
            
            return variations[:num_variations]
        except Exception as e:
            logger.error(f"Error generating query variations: {e}", exc_info=True)
            return [question]

