"""SenGen: A framework for scenario generation and ethical agent evaluation."""

__version__ = "0.1.0"

# Core components
from .core.scenario import ScenarioConfig, ScenarioGenerator
from .core.metrics_tracker import MetricsTracker

# Agent implementations
from .agents.base import Agent, AgentConfig, LLMAgent

# Environment implementations
from .envs.gym_env import SenGenGymEnv

__all__ = [
    "ScenarioConfig",
    "ScenarioGenerator",
    "MetricsTracker",
    "Agent",
    "AgentConfig",
    "LLMAgent",
    "SenGenGymEnv",
]
