import random

from locust import HttpUser, between, task

# Sample questions for load testing
QUESTIONS = [
    "What is machine learning?",
    "How does the retriever work?",
    "Explain the architecture of the Adaptive RAG Agent.",
    "What is semantic caching?",
    "How do we handle hallucinations?",
    "What vector store is used?",
    "How does query rewriting improve results?",
    "Give an example of a relevant document.",
    "What is OpenTelemetry used for here?",
]


class RAGUser(HttpUser):
    # Wait between 1 and 5 seconds between tasks
    wait_time = between(1, 5)

    @task(3)
    def query_endpoint(self):
        """Test the main /query endpoint."""
        question = random.choice(QUESTIONS)
        payload = {
            "question": question,
            "max_iterations": 2,  # Keep iterations low for load tests
            "stream": False,
        }

        # Test the endpoint. We group by endpoint name rather than the specific query
        # to get aggregated stats in locust.
        with self.client.post(
            "/query", json=payload, catch_response=True, name="/query"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "answer" in data:
                    response.success()
                else:
                    response.failure(f"Response missing 'answer' key: {data}")
            else:
                response.failure(f"Failed with status {response.status_code}: {response.text}")

    @task(1)
    def health_check(self):
        """Test the health endpoint."""
        self.client.get("/health", name="/health")

    @task(1)
    def stats_endpoint(self):
        """Test the stats endpoint."""
        self.client.get("/stats", name="/stats")
