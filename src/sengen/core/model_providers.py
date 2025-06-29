"""Model provider abstractions for different LLM backends."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, Union
from enum import Enum
from dataclasses import dataclass
import os

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    raise ImportError("langchain_openai is required for OpenAI provider")

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import BitsAndBytesConfig
except ImportError:
    ChatHuggingFace = None
    HuggingFacePipeline = None
    BitsAndBytesConfig = None


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


@dataclass
class ModelConfig:
    """Configuration for different model providers."""
    provider: Union[ModelProvider, str]
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    # OpenAI specific
    api_key: Optional[str] = None
    # Ollama specific  
    base_url: Optional[str] = None
    # HuggingFace specific
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    # Additional parameters
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if isinstance(self.provider, str):
            self.provider = ModelProvider(self.provider)
        if self.extra_params is None:
            self.extra_params = {}


class LLMProviderFactory:
    """Factory for creating LLM instances from different providers."""
    
    @staticmethod
    def create_llm(config: ModelConfig) -> Any:
        """Create an LLM instance based on the provider configuration."""
        
        if config.provider == ModelProvider.OPENAI:
            return LLMProviderFactory._create_openai_llm(config)
        elif config.provider == ModelProvider.OLLAMA:
            return LLMProviderFactory._create_ollama_llm(config)
        elif config.provider == ModelProvider.HUGGINGFACE:
            return LLMProviderFactory._create_huggingface_llm(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
    
    @staticmethod
    def _create_openai_llm(config: ModelConfig) -> ChatOpenAI:
        """Create OpenAI LLM instance."""
        params = {
            "model_name": config.model_name,
            "temperature": config.temperature,
        }
        
        if config.max_tokens:
            params["max_tokens"] = config.max_tokens
            
        if config.api_key:
            params["openai_api_key"] = config.api_key
        
        params.update(config.extra_params)
        return ChatOpenAI(**params)
    
    @staticmethod
    def _create_ollama_llm(config: ModelConfig) -> Any:
        """Create Ollama LLM instance."""
        if ChatOllama is None:
            raise ImportError("langchain_ollama is required for Ollama provider. Install with: pip install langchain-ollama")
        
        params = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        
        if config.base_url:
            params["base_url"] = config.base_url
        else:
            # Default Ollama URL
            params["base_url"] = "http://localhost:11434"
            
        if config.max_tokens:
            params["num_predict"] = config.max_tokens
        
        params.update(config.extra_params)
        return ChatOllama(**params)
    
    @staticmethod
    def _create_huggingface_llm(config: ModelConfig) -> Any:
        """Create HuggingFace LLM instance."""
        if HuggingFacePipeline is None or ChatHuggingFace is None:
            raise ImportError("langchain_huggingface is required for HuggingFace provider. Install with: pip install langchain-huggingface")
        
        # Set up pipeline parameters
        pipeline_kwargs = {
            "max_new_tokens": config.max_tokens or 512,
            "do_sample": True,
            "temperature": config.temperature,
            "return_full_text": False,
        }
        pipeline_kwargs.update(config.extra_params)
        
        # Set up model loading parameters
        model_kwargs = {}
        
        # Handle quantization configuration
        if config.load_in_4bit or config.load_in_8bit:
            if BitsAndBytesConfig is None:
                raise ImportError("bitsandbytes is required for quantization. Install with: pip install bitsandbytes")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.load_in_4bit,
                load_in_8bit=config.load_in_8bit,
                bnb_4bit_quant_type="nf4" if config.load_in_4bit else None,
                bnb_4bit_compute_dtype=config.torch_dtype or "float16" if config.load_in_4bit else None,
                bnb_4bit_use_double_quant=True if config.load_in_4bit else False,
                # Enable CPU offloading for memory-constrained scenarios
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            
        if config.torch_dtype:
            import torch
            if config.torch_dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif config.torch_dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
        
        # Handle device mapping - use better defaults for memory-constrained scenarios
        if config.device:
            if config.device == "auto":
                # For quantized models, use a more memory-friendly device mapping
                if config.load_in_4bit or config.load_in_8bit:
                    model_kwargs["device_map"] = "auto"
                    # Enable low memory mode for better memory management
                    model_kwargs["low_cpu_mem_usage"] = True
                else:
                    model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = config.device
        
        # Create HuggingFace pipeline
        llm = HuggingFacePipeline.from_model_id(
            model_id=config.model_name,
            task="text-generation",
            pipeline_kwargs=pipeline_kwargs,
            model_kwargs=model_kwargs,
        )
        
        # Wrap with ChatHuggingFace for better chat interface
        return ChatHuggingFace(llm=llm)


class ModelProviderMixin:
    """Mixin to add model provider functionality to existing classes."""
    
    def _create_llm_from_config(self, model_config: ModelConfig) -> Any:
        """Create LLM instance from model configuration."""
        return LLMProviderFactory.create_llm(model_config)
    
    def _parse_model_config(self, config_dict: Dict[str, Any]) -> ModelConfig:
        """Parse model configuration from dictionary."""
        # Handle legacy format (just model_name as string)
        if isinstance(config_dict.get("model_name"), str) and "provider" not in config_dict:
            # Default to OpenAI for backward compatibility
            config_dict = config_dict.copy()
            config_dict["provider"] = ModelProvider.OPENAI
        
        # Handle new format
        provider = config_dict.get("provider", ModelProvider.OPENAI)
        model_name = config_dict["model_name"]
        temperature = config_dict.get("temperature", 0.7)
        
        return ModelConfig(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=config_dict.get("max_tokens"),
            api_key=config_dict.get("api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=config_dict.get("base_url"),
            device=config_dict.get("device"),
            torch_dtype=config_dict.get("torch_dtype"),
            load_in_8bit=config_dict.get("load_in_8bit", False),
            load_in_4bit=config_dict.get("load_in_4bit", False),
            extra_params=config_dict.get("extra_params", {})
        )


def get_supported_models() -> Dict[str, Dict[str, Any]]:
    """Get information about supported models for each provider."""
    return {
        "openai": {
            "models": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "description": "OpenAI GPT models via API",
            "requirements": ["openai", "langchain_openai"],
            "setup": "Set OPENAI_API_KEY environment variable"
        },
        "ollama": {
            "models": ["llama3", "llama3:8b", "llama3:70b", "mistral", "codellama", "phi3"],
            "description": "Local models via Ollama",
            "requirements": ["langchain_ollama", "ollama server running"],
            "setup": "Install Ollama and pull models: ollama pull llama3"
        },
        "huggingface": {
            "models": ["meta-llama/Llama-2-7b-chat-hf", "microsoft/DialoGPT-medium", "mistralai/Mistral-7B-Instruct-v0.1"],
            "description": "Hugging Face transformers models",
            "requirements": ["transformers", "torch", "accelerate"],
            "setup": "Models downloaded automatically on first use"
        }
    } 