import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

class TestSchema(BaseModel):
    value: str = Field(description="A test value")

def verify_ollama():
    print("Initializing ChatOllama with qwen2.5:14b...")
    try:
        llm = ChatOllama(model="qwen2.5:14b", temperature=0)
        
        print("\nTesting simple generation...")
        response = llm.invoke("Hello, are you working? Reply with 'Yes, I am working'.")
        print(f"Response: {response.content}")
        
        print("\nTesting structured output (used in graders)...")
        try:
            structured_llm = llm.with_structured_output(TestSchema)
            structured_response = structured_llm.invoke("Say 'test'")
            print(f"Structured Response: {structured_response}")
        except Exception as e:
            print(f"Structured output failed: {e}")
            print("Note: This might be expected if the model or library version has issues with structured output, but basic generation should work.")

        return True
    except Exception as e:
        print(f"Failed to initialize or run ChatOllama: {e}")
        return False

if __name__ == "__main__":
    verify_ollama()
