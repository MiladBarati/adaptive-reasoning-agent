"""Query rewriting for improved retrieval."""

from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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


class QueryRewriter:
    """Rewrites user queries to improve retrieval quality."""
    
    def __init__(self, model: str = "qwen2.5:14b", temperature: float = 0) -> None:
        """
        Initialize the query rewriter.
        
        Args:
            model: Ollama model name
            temperature: LLM temperature for generation
        """
        self.llm = ChatOllama(model=model, temperature=temperature)
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
        # self.chain = self.prompt | self.llm | StrOutputParser() # Decomposing for metrics
    
    def rewrite(self, question: str) -> str:
        """
        Rewrite a query for better retrieval.
        
        Args:
            question: Original user question
            
        Returns:
            Rewritten question
        """
        try:
            # We need to access the raw response to get metadata.
            # Convert chain to not use StrOutputParser implicitly or break it down
            msg = self.prompt.invoke({"question": question})
            response = self.llm.invoke(msg)
            rewritten = response.content

            if response.usage_metadata:
                input_tokens = response.usage_metadata.get("input_tokens", 0)
                output_tokens = response.usage_metadata.get("output_tokens", 0)
                token_usage_counter.add(input_tokens, {"type": "input", "model": self.llm.model})
                token_usage_counter.add(output_tokens, {"type": "output", "model": self.llm.model})

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

