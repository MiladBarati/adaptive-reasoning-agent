"""
Test script to verify the Corrective & Adaptive RAG Agent is working correctly.

This script tests basic functionality without requiring the full setup.
"""

import os
from dotenv import load_dotenv
from src.core.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)


def test_imports():
    """Test that all imports work correctly."""
    print("\n" + "="*80)
    print("TEST 1: Checking Imports")
    print("="*80)
    
    try:
        print("✓ Importing core components...")
        from src.core.embeddings import get_embeddings
        from src.core.vector_store import VectorStoreManager
        from src.core.retriever import AdvancedRetriever
        
        print("✓ Importing corrective modules...")
        from src.corrective.query_rewriter import QueryRewriter
        from src.corrective.relevance_grader import RelevanceGrader
        from src.corrective.hallucination_checker import HallucinationChecker
        from src.corrective.answer_verifier import AnswerVerifier
        
        print("✓ Importing agent components...")
        from src.agents.state import RAGState
        from src.agents.nodes import initialize_components
        from src.agents.rag_graph import create_rag_graph, query_rag_agent
        
        print("✓ Importing API and UI...")
        from src.api.main import app
        from src.ui.gradio_app import demo
        
        print("\n✓ All imports successful!")
        return True
        
    except Exception as e:
        logger.error(f"Import error: {e}", exc_info=True)
        print(f"\n✗ Import error: {e}")
        return False


def test_api_keys():
    """Test that API keys are configured."""
    print("\n" + "="*80)
    print("TEST 2: Checking API Keys")
    print("="*80)
    
    keys_ok = True
    
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"✓ GROQ_API_KEY: {groq_key[:10]}...{groq_key[-4:]}")
    else:
        print("✗ GROQ_API_KEY not found")
        keys_ok = False
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        print(f"✓ TAVILY_API_KEY: {tavily_key[:10]}...{tavily_key[-4:]}")
    else:
        print("✗ TAVILY_API_KEY not found")
        keys_ok = False
    
    langchain_key = os.getenv("LANGCHAIN_API_KEY")
    if langchain_key:
        print(f"✓ LANGCHAIN_API_KEY: {langchain_key[:10]}...{langchain_key[-4:]}")
    else:
        print("⚠ LANGCHAIN_API_KEY not found (optional)")
    
    return keys_ok


def test_embeddings():
    """Test embeddings generation."""
    print("\n" + "="*80)
    print("TEST 3: Testing Embeddings")
    print("="*80)
    
    try:
        from src.core.embeddings import get_embeddings
        
        print("Initializing embedding model...")
        embeddings = get_embeddings()
        
        print("Generating test embedding...")
        test_text = "This is a test sentence for embeddings."
        embedding = embeddings.embed_query(test_text)
        
        print(f"✓ Generated embedding with {len(embedding)} dimensions")
        print(f"✓ First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Embeddings test failed: {e}", exc_info=True)
        print(f"✗ Embeddings test failed: {e}")
        return False


def test_query_rewriter():
    """Test query rewriting."""
    print("\n" + "="*80)
    print("TEST 4: Testing Query Rewriter")
    print("="*80)
    
    try:
        from src.corrective.query_rewriter import QueryRewriter
        
        print("Initializing query rewriter...")
        rewriter = QueryRewriter()
        
        test_query = "How do computers learn stuff?"
        print(f"Original query: {test_query}")
        
        print("Rewriting query...")
        rewritten = rewriter.rewrite(test_query)
        
        print(f"Rewritten query: {rewritten}")
        print("✓ Query rewriting successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"Query rewriter test failed: {e}", exc_info=True)
        print(f"✗ Query rewriter test failed: {e}")
        return False


def test_vector_store():
    """Test vector store operations."""
    print("\n" + "="*80)
    print("TEST 5: Testing Vector Store")
    print("="*80)
    
    try:
        from src.core.vector_store import VectorStoreManager
        
        print("Initializing vector store...")
        vsm = VectorStoreManager(persist_directory="./test_chroma_db")
        
        print("Adding test document...")
        test_docs = ["Machine learning is a subset of artificial intelligence."]
        ids = vsm.ingest_text_documents(
            texts=test_docs,
            metadatas=[{"source": "test"}],
            chunk_size=1000,
            chunk_overlap=200
        )
        
        print(f"✓ Added {len(ids)} document chunks")
        
        print("Testing similarity search...")
        results = vsm.similarity_search("What is machine learning?", k=1)
        
        print(f"✓ Retrieved {len(results)} documents")
        if results:
            print(f"✓ Top result: {results[0].page_content[:100]}...")
        
        stats = vsm.get_stats()
        print(f"✓ Vector store stats: {stats}")
        
        # Cleanup
        print("\nCleaning up test database...")
        vsm.clear()
        
        import shutil
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")
        
        print("✓ Vector store test successful!")
        return True
        
    except Exception as e:
        logger.error(f"Vector store test failed: {e}", exc_info=True)
        print(f"✗ Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("CORRECTIVE & ADAPTIVE RAG AGENT - TEST SUITE")
    print("="*80)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("API Keys", test_api_keys()))
    results.append(("Embeddings", test_embeddings()))
    results.append(("Query Rewriter", test_query_rewriter()))
    results.append(("Vector Store", test_vector_store()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print("\n" + "-"*80)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    print("="*80)
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: python setup_sample_data.py")
        print("2. Run: python -m src.ui.gradio_app")
        print("3. Open: http://localhost:7860")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("Make sure you have:")
        print("1. Installed all dependencies: pip install -r requirements.txt")
        print("2. Set up your .env file with API keys")
        print("3. Activated your virtual environment")
    
    return total_passed == total_tests


if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.warning("Test interrupted by user")
        print("\n\nTest interrupted by user.")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

