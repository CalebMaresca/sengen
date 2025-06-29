#!/usr/bin/env python3
"""Utility script to display information about supported model providers and setup instructions."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sengen import get_supported_models, ModelProvider

def main():
    """Display information about supported model providers."""
    print("SenGen Model Provider Information")
    print("=" * 50)
    
    models_info = get_supported_models()
    
    for provider, info in models_info.items():
        print(f"\n🔧 {provider.upper()}")
        print(f"Description: {info['description']}")
        print(f"Requirements: {', '.join(info['requirements'])}")
        print(f"Setup: {info['setup']}")
        print(f"Example models: {', '.join(info['models'][:3])}...")
        
        # Show example configuration
        print("\nExample configuration:")
        if provider == "openai":
            config_example = """
scenario:
  provider: "openai"
  model_name: "gpt-4o-mini"
  temperature: 0.7
  api_key: "your-api-key-here"  # or set OPENAI_API_KEY env var

agent:
  provider: "openai"
  model_name: "gpt-4o-mini"
  temperature: 0.2"""
        elif provider == "ollama":
            config_example = """
scenario:
  provider: "ollama"
  model_name: "llama3"
  temperature: 0.7
  base_url: "http://localhost:11434"

agent:
  provider: "ollama"
  model_name: "llama3"
  temperature: 0.2"""
        else:  # huggingface
            config_example = """
# Note: HuggingFace only supports agents, not scenario generation
# For scenarios, use OpenAI or Ollama providers

agent:
  provider: "huggingface"
  model_name: "mistralai/Mistral-7B-Instruct-v0.1" 
  temperature: 0.2
  device: "auto"
  torch_dtype: "float16"
  load_in_4bit: true"""
        
        print(config_example)
        print("-" * 40)

    print("\n💡 Tips:")
    print("• Scenario generation: Use OpenAI or Ollama (reliable structured output)")
    print("• Agent decisions: All providers supported (OpenAI, Ollama, HuggingFace)")
    print("• Mixed approach recommended: Ollama for scenarios, OpenAI for agents")
    print("• HuggingFace: Agent-only support due to structured output limitations")
    print("• Check example config files: config_ollama.yaml, config_mixed.yaml")

if __name__ == "__main__":
    main() 