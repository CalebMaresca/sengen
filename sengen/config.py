"""Configuration classes and model provider logic for SenGen."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Type
from enum import Enum
import yaml
from pydantic import BaseModel, Field, create_model

# Model Provider Imports
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    raise ImportError("langchain_openai is required for OpenAI provider")

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"


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
    # Additional parameters
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if isinstance(self.provider, str):
            self.provider = ModelProvider(self.provider)
        if self.extra_params is None:
            self.extra_params = {}


@dataclass
class SenGenConfig:
    """Main configuration for SenGen environment."""
    
    # Scenario configuration
    theme: str
    goal: str
    reward: Dict[str, str] = field(default_factory=lambda: {
        "type": "float",
        "description": "Reward for the agent based on the current state and their previous choice. (r(a_t, s_{t+1}))"
    })
    metrics: Optional[Dict[str, Dict[str, str]]] = None
    metrics_model: Optional[Type[BaseModel]] = None
    cost_correlation: float = 0.5
    max_steps: int = 5
    max_choices: int = 4
    temperature: float = 0.7
    model_name: str = "gpt-4o-mini"
    
    # Model provider configuration
    provider: Optional[str] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = None

    # Type mapping from strings to Python types
    type_mapping = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "bool": bool,
        "boolean": bool,
        "list": List[Any],
        "dict": Dict[str, Any]
    }
    
    def __post_init__(self):
        """Initialize metrics_model if metrics are provided but metrics_model is None."""
        if self.metrics and self.metrics_model is None:
            self._create_metrics_model()
    
    def _create_metrics_model(self):
        """Create the metrics model from metrics configuration."""
        if not self.metrics:
            return
            
        metrics_fields = {}
        
        for metric_name, metric_data in self.metrics.items():
            if isinstance(metric_data, dict):
                # Convert the string type to an actual Python type
                metric_type = self.type_mapping.get(metric_data["type"].lower(), Any)
                
                # Create field with appropriate type and metadata
                metrics_fields[metric_name] = (
                    metric_type, 
                    Field(description=metric_data.get("description", ""))
                )
        
        self.metrics_model = create_model("Metrics", **metrics_fields)
    
    @classmethod
    def from_yaml(cls, path: str) -> "SenGenConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract scenario config (support both old and new format)
        if "scenario" in config:
            config_data = config["scenario"].copy()
        else:
            config_data = config.copy()

        if "metrics" not in config_data:
            config_data["metrics"] = {}

        return cls(**config_data)
    
    def create_output_model(self) -> Type[BaseModel]:
        """Creates a dynamic SenGenOutput model based on the metrics configuration."""
        if not self.metrics:
            raise ValueError("Metrics model must be defined before creating output model")
        
        # Create the Choice model dynamically with the metrics
        choice_model = create_model(
            "Choice",
            text=(str, Field(description="Text of the choice")),
            metrics=(self.metrics_model, Field(description="Metric values for the choice")),
            __doc__="Choice for the agent with associated metrics."
        )
        
        # Define the output model with the dynamic Choice model
        output_model = create_model(
            "SenGenOutput",
            state=(str, Field(description="Current state of the scenario")),
            reward=(self.type_mapping[self.reward["type"]], 
                    Field(description= self.reward["description"])),
            choices=(List[choice_model], Field(description="Available choices for the agent")),
            __doc__="Always use this to structure your response to the user."
        )
        
        return output_model

    def get_model_config(self) -> ModelConfig:
        """Get model configuration for LLM provider."""
        return ModelConfig(
            provider=self.provider or ModelProvider.OPENAI,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            extra_params=self.extra_params or {}
        )


class LLMProviderFactory:
    """Factory for creating LLM instances from different providers."""
    
    @staticmethod
    def create_llm(config: ModelConfig) -> Any:
        """Create an LLM instance based on the provider configuration."""
        
        if config.provider == ModelProvider.OPENAI:
            return LLMProviderFactory._create_openai_llm(config)
        elif config.provider == ModelProvider.OLLAMA:
            return LLMProviderFactory._create_ollama_llm(config)
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
            "base_url": config.base_url or "http://localhost:11434"
        }
        
        if config.max_tokens:
            params["num_predict"] = config.max_tokens
        
        params.update(config.extra_params)
        return ChatOllama(**params)