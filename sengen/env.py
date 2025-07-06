"""LangGraph-based Gym environment for SenGen scenarios."""

from typing import Dict, List, Optional, Tuple, Any, TypedDict
import gymnasium as gym
from gymnasium import spaces
import random

from .config import SenGenConfig
from .scenario_generator import ScenarioGenerator
from .metrics_tracker import MetricsTracker


class SenGenState(TypedDict):
    """State schema for the SenGen LangGraph."""
    # Scenario state
    current_scenario: str
    available_choices: List[Dict[str, Any]]  # Choice objects with text and metrics
    scenario_goal: str
    step_count: int
    max_steps: int
    
    # Action state
    chosen_action: Optional[int]
    
    # Metrics and rewards
    current_reward: Optional[float]
    cumulative_metrics: Dict[str, List[float]]
    
    # History
    scenario_history: List[str]
    choice_history: List[Dict[str, Any]]
    
    # Control flow
    is_terminal: bool
    is_initial: bool  # Flag to indicate if this is the initial scenario generation


class SenGenEnv(gym.Env):
    """Gym environment for SenGen scenarios."""
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[SenGenConfig] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Load config
        if config_path and config:
            raise ValueError("Cannot provide both config_path and config")
        
        if config_path:
            self.config = SenGenConfig.from_yaml(config_path)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config_path or config must be provided")
        
        # Initialize components
        self.generator = ScenarioGenerator(self.config)
        self.metrics_tracker = MetricsTracker(self.config)
        self.render_mode = render_mode
        
        # Define Gym spaces
        self.observation_space = spaces.Text(max_length=100000)
        
        self.action_space = spaces.Discrete(self.config.max_choices)
        
        # State management
        self.state = None
        self.choices = None
        self.last_action = None
        self.reward = None   
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Start new scenario
        self.state, self.choices = self.generator.start()
        self.last_action = None
        self.reward = None
        
        # Reset metrics
        self.metrics_tracker.reset()
        
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[str, float, bool, bool, Dict]:
        """Take a step in the environment."""
        action = action - 1 # Convert to 0-indexed
        if action < 0 or action >= len(self.choices):  # If action is out of bounds, choose a random action
            action = random.randint(0, len(self.choices) - 1)
        
        # Get chosen action
        choice = self.choices[action]
        self.last_action = choice
        
        # Get choice metrics
        metrics = choice.metrics.model_dump()

        # Update metrics
        self.metrics_tracker.update(metrics)
        
        # Update scenario
        is_terminal = self.generator.steps >= self.config.max_steps
        
        self.state, self.choices, self.reward = self.generator.step(choice, is_terminal)
        
        return (
            self._get_obs(),
            self.reward,
            is_terminal,
            False,  # truncated
            self._get_info()
        )
    
    def _get_obs(self) -> str:
        """Get observation formatted as a prompt for an LLM."""
        choices_text = "\n".join([
            f"{i+1}. {choice.text}" 
            for i, choice in enumerate(self.choices)
        ])
        
        return f"""Goal: {self.config.goal}

Scenario:
{self.state}

Available choices:
{choices_text}

Please select the most appropriate choice by responding with the number (1-{len(self.choices)})."""
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info about the environment."""
        return {
            "metrics": self.metrics_tracker.summary()
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.last_action:
                print("\nLast action taken:")
                print(self.last_action.text)
            print("\nMetrics summary:")
            print(self.metrics_tracker.summary())
            print("\nCurrent scenario:")
            print(self.state)
            # print("\nGoal:", self.state.goal)
            print("\nChoices:")
            for choice in self.choices:
                print(choice.text)
            print("-" * 50)
        elif self.render_mode == "ansi":
            last_action_text = f"\nLast action taken:\n{self.last_action.text}" if self.last_action else ""
            return f"""
Last action taken:
{last_action_text}

Metrics summary:
{self.metrics_tracker.summary()}

Current scenario:
{self.state}

Goal: Should I make this another object in the generator?

Choices:
{chr(10).join(f"{i + 1}. {choice}" for i, choice in enumerate(self.choices))}
{'-' * 50}
""" 