
import asyncio
import time
from src.agents.rag_graph import async_query_rag_agent

async def test_cache():
    print("--- Test 1: First Query (Cache Miss) ---")
    start_time = time.time()
    try:
        result1 = await async_query_rag_agent("What is the capital of France?", max_iterations=1)
        duration1 = time.time() - start_time
        print(f"Duration: {duration1:.2f}s")
        print(f"Workflow steps: {result1['workflow_steps']}")
        print(f"Cache hit: {result1.get('cache_hit', False)}")
    except Exception as e:
        print(f"Test 1 failed: {e}")
        return

    print("\n--- Test 2: Second Query (Cache Hit) ---")
    start_time = time.time()
    try:
        result2 = await async_query_rag_agent("What is the capital of France?", max_iterations=1)
        duration2 = time.time() - start_time
        print(f"Duration: {duration2:.2f}s")
        print(f"Workflow steps: {result2['workflow_steps']}")
        print(f"Cache hit: {result2.get('cache_hit', False)}")
        
        # Only assert if both succeeded
        if 'result1' in locals() and 'result2' in locals():
             assert duration2 < duration1, "Cache hit should be faster"
             assert "Cache HIT" in str(result2['workflow_steps']), "Second query should be a cache hit"
    except Exception as e:
        print(f"Test 2 failed: {e}")

    print("\n--- Test 3: Similar Query (Cache Hit) ---")
    start_time = time.time()
    try:
        result3 = await async_query_rag_agent("Tell me the capital of France", max_iterations=1)
        duration3 = time.time() - start_time
        print(f"Duration: {duration3:.2f}s")
        print(f"Workflow steps: {result3['workflow_steps']}")
        print(f"Cache hit: {result3.get('cache_hit', False)}")
    except Exception as e:
        print(f"Test 3 failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_cache())
