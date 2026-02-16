import shutil
import tempfile
import unittest

from src.core.semantic_cache import SemanticCache


class TestSemanticCache(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the vector store
        self.test_dir = tempfile.mkdtemp()
        self.cache = SemanticCache(
            persist_directory=self.test_dir,
            collection_name="test_semantic_cache",
            similarity_threshold=0.90,
        )

    def tearDown(self):
        # Force garbage collection to release file handles
        self.cache = None
        import gc

        gc.collect()

        # specific for Windows: wait a bit or ignore errors
        import time

        for _ in range(3):
            try:
                shutil.rmtree(self.test_dir)
                break
            except PermissionError:
                time.sleep(0.1)
            except Exception as e:
                print(f"Failed to clean up test dir: {e}")
                break

    def test_cache_miss_initially(self):
        query = "What is the capital of France?"
        result = self.cache.check_cache(query)
        self.assertIsNone(result, "Cache should be empty initially")

    def test_cache_hit_after_update(self):
        query = "What is the capital of France?"
        answer = "The capital of France is Paris."
        rewritten_query = "What is the capital city of France?"

        # Update cache
        self.cache.update_cache(query, answer, rewritten_query)

        # Check cache with exact same query
        result = self.cache.check_cache(query)
        self.assertIsNotNone(result, "Should be a cache hit")
        self.assertEqual(result["answer"], answer)
        self.assertEqual(result["original_query"], query)

        # Check cache with similar query
        similar_query = "Tell me the capital of France"
        result_similar = self.cache.check_cache(similar_query)
        self.assertIsNotNone(result_similar, "Should be a cache hit for similar query")
        self.assertEqual(result_similar["answer"], answer)

    def test_cache_miss_for_dissimilar_query(self):
        query = "What is the capital of France?"
        answer = "The capital of France is Paris."
        self.cache.update_cache(query, answer)

        dissimilar_query = "What is the population of Germany?"
        result = self.cache.check_cache(dissimilar_query)
        self.assertIsNone(result, "Should be a cache miss for dissimilar query")

    def test_clear_cache(self):
        query = "Test Query"
        answer = "Test Answer"
        self.cache.update_cache(query, answer)

        self.cache.clear_cache()
        result = self.cache.check_cache(query)
        self.assertIsNone(result, "Cache should be empty after clearing")


if __name__ == "__main__":
    unittest.main()
