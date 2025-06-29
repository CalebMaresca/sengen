"""SenGen: A framework for scenario generation and ethical agent evaluation.

Provider Support:
- Scenario Generation: OpenAI and Ollama (structured output required)
- Agent Decisions: All providers (OpenAI, Ollama, HuggingFace)

Use mixed configurations for optimal performance: Ollama for scenarios, OpenAI for agents.
"""

__version__ = "0.1.0"

# Core components
from .core.scenario import ScenarioConfig, ScenarioGenerator
from .core.metrics_tracker import MetricsTracker
from .core.model_providers import (
    ModelProvider, 
    ModelConfig, 
    LLMProviderFactory, 
    get_supported_models
)

# Agent implementations
from .agents.base import Agent, AgentConfig, LLMAgent

# Environment implementations
from .envs.gym_env import SenGenGymEnv

__all__ = [
    "ScenarioConfig",
    "ScenarioGenerator",
    "MetricsTracker",
    "ModelProvider",
    "ModelConfig", 
    "LLMProviderFactory",
    "get_supported_models",
    "Agent",
    "AgentConfig",
    "LLMAgent",
    "SenGenGymEnv",
]
