"""Document relevance grading."""

from typing import List, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.core.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)


class GradeDocument(BaseModel):
    """Binary score for document relevance."""
    
    binary_score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )


class RelevanceGrader:
    """Grades retrieved documents for relevance to the query."""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0) -> None:
        """
        Initialize the relevance grader.
        
        Args:
            model: Groq model name
            temperature: LLM temperature for generation
        """
        self.llm = ChatGroq(model=model, temperature=temperature)
        
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
            result = self.chain.invoke({
                "document": document.page_content,
                "question": question
            })
            return result.binary_score.lower() == "yes"
        except Exception as e:
            logger.error(f"Error grading document: {e}", exc_info=True)
            # Default to relevant on error to avoid losing potentially useful docs
            return True
    
    def grade_documents(
        self,
        documents: List[Document],
        question: str
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
        
        logger.info(f"Graded {len(documents)} documents: "
                   f"{len(relevant_docs)} relevant, {len(irrelevant_docs)} irrelevant")
        
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

