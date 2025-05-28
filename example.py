#!/usr/bin/env python3
"""
Example usage of the LLM Wrapper package

This script demonstrates how to use the various wrapper classes
for different LLM providers.
"""

import os
from llm_wrapper import OpenAIWrapper, ClaudeWrapper, GeminiWrapper, TogetherAIWrapper

def main():
    """Main example function"""
    
    # Single request example
    print("=== Single Request Example ===")
    
    try:
        # Initialize wrapper (will auto-load from .env file)
        gpt = OpenAIWrapper(model_name="gpt-4o-mini")
        
        # Or initialize with direct API key
        # gpt = OpenAIWrapper(model_name="gpt-4o-mini", api_key="your-api-key")
        
        response = gpt("What is the capital of France?")
        print(f"Response: {response}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your API keys in a .env file or pass them directly")
    
    # Batch processing example
    print("\n=== Batch Processing Example ===")
    
    try:
        gpt = OpenAIWrapper(model_name="gpt-4o-mini")
        
        # Prepare multiple queries
        queries = {
            0: {"user_prompt": "What is artificial intelligence?"},
            1: {"user_prompt": "Explain machine learning in simple terms"},
            2: {"user_prompt": "What is the future of AI?"}
        }
        
        print(f"Processing {len(queries)} queries...")
        
        # Run batch processing
        f_out = gpt.run_batch(queries)
        print(f"Results saved to: {f_out}")
        
        # Get results
        results = gpt.process_batch()
        
        for key, response in results.items():
            print(f"\nQuery {key}: {response[:100]}...")
            
    except ValueError as e:
        print(f"Error: {e}")
    
    # Multi-provider comparison
    print("\n=== Multi-Provider Comparison ===")
    
    prompt = "Explain quantum computing in one sentence."
    
    providers = {
        "OpenAI": lambda: OpenAIWrapper(model_name="gpt-4o-mini"),
        "Claude": lambda: ClaudeWrapper(),
        "Gemini": lambda: GeminiWrapper(),
        "Together": lambda: TogetherAIWrapper()
    }
    
    for name, wrapper_fn in providers.items():
        try:
            wrapper = wrapper_fn()
            response = wrapper(prompt)
            print(f"{name}: {response}")
        except ValueError as e:
            print(f"{name}: Error - {e}")

if __name__ == "__main__":
    main() 