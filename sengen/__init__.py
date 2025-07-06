"""SenGen: Scenario Generator for LLM-based RL environments.

A simplified, LangGraph-based implementation for training and evaluating
reinforcement learning agents on ethical decision-making scenarios.

Compatible with OpenAI Gym/Gymnasium for seamless RL framework integration.
"""

__version__ = "0.2.0"

# Main environment - the primary interface for RL
from .env import SenGenEnv

# Configuration classes
from .config import SenGenConfig, ModelProvider, ModelConfig

# Core components (for advanced usage)
from .scenario_generator import ScenarioGenerator

from .metrics_tracker import MetricsTracker

__all__ = [
    # Main environment interface
    "SenGenEnv",
    
    # Configuration
    "SenGenConfig",
    "ModelProvider",
    "ModelConfig",
    
    # Core components
    "ScenarioGenerator",
    "MetricsTracker",
] 