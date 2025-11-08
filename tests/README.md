# Test Suite for Adaptive Reasoning Agent

This directory contains comprehensive unit and integration tests for the Adaptive Reasoning Agent project.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                     # Unit tests
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_query_rewriter.py
│   ├── test_relevance_grader.py
│   ├── test_hallucination_checker.py
│   ├── test_answer_verifier.py
│   ├── test_agent_state.py
│   ├── test_agent_nodes.py
│   └── test_api.py
└── integration/              # Integration tests
    ├── test_rag_workflow.py
    ├── test_vector_store_integration.py
    └── test_api_integration.py
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Only Unit Tests
```bash
pytest tests/unit/
```

### Run Only Integration Tests
```bash
pytest tests/integration/
```

### Run Specific Test File
```bash
pytest tests/unit/test_embeddings.py
```

### Run Tests with Coverage
```bash
pytest --cov=src --cov-report=html
```

### Run Tests by Marker
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Run Tests Verbosely
```bash
pytest -v
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual components in isolation
- Use mocks for external dependencies
- Fast execution
- Located in `tests/unit/`

### Integration Tests (`@pytest.mark.integration`)
- Test component interactions
- May use real dependencies (with API keys)
- Slower execution
- Located in `tests/integration/`

### Slow Tests (`@pytest.mark.slow`)
- Tests that take longer to run
- May require API keys
- Can be skipped with `pytest -m "not slow"`

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `vector_store_manager`: Vector store manager instance
- `populated_vector_store`: Vector store with sample documents
- `sample_documents`: Sample Document objects
- `mock_groq_llm`: Mock Groq LLM
- `mock_embeddings`: Mock embeddings model
- `mock_tavily_client`: Mock Tavily search client
- `sample_rag_state`: Sample RAG state dictionary

## Requirements

Tests require the following dependencies (already in `requirements.txt`):
- `pytest>=8.0.0`
- `pytest-asyncio>=0.23.0`
- `pytest-cov>=4.1.0`
- `pytest-mock>=3.12.0`
- `httpx>=0.27.0`

## Environment Variables

Some integration tests may require API keys:
- `GROQ_API_KEY`: For LLM operations
- `TAVILY_API_KEY`: For web search tests
- `LANGCHAIN_API_KEY`: Optional, for LangSmith tracing

Tests will skip if required API keys are not available.

## Coverage

To generate coverage reports:
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

Coverage reports will be generated in:
- Terminal output: `--cov-report=term-missing`
- HTML report: `htmlcov/index.html`
- XML report: `coverage.xml`

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- Unit tests run quickly without external dependencies
- Integration tests can be conditionally run
- Coverage reports are generated for code quality metrics

## Writing New Tests

When adding new tests:

1. **Unit Tests**: Place in `tests/unit/` and use mocks for external dependencies
2. **Integration Tests**: Place in `tests/integration/` and test component interactions
3. **Use Fixtures**: Leverage existing fixtures from `conftest.py`
4. **Add Markers**: Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`)
5. **Follow Naming**: Test files should start with `test_` and test functions should start with `test_`

## Example Test

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
class TestMyComponent:
    """Test cases for MyComponent."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        component = MyComponent()
        
        # Act
        result = component.do_something()
        
        # Assert
        assert result is not None
    
    @patch('module.external_dependency')
    def test_with_mock(self, mock_dependency):
        """Test with mocked dependency."""
        mock_dependency.return_value = "mocked"
        # ... test code
```

