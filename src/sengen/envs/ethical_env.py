"""Gymnasium environment for ethical scenarios."""

from typing import Dict, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces
from ..core.scenario import ScenarioGenerator, ScenarioConfig, ScenarioState
from ..metrics.ethical_metrics import EthicalMetrics

class EthicalEnv(gym.Env):
    """Environment for ethical scenario-based decision making."""
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        theme: Optional[str] = None,
        max_actions: int = 4,
        max_steps: int = 10,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Load config or create from parameters
        if config_path:
            self.config = ScenarioConfig.from_yaml(config_path)
        else:
            self.config = ScenarioConfig(
                theme=theme or "ethical decision making",
                max_steps=max_steps
            )
        
        # Initialize components
        self.generator = ScenarioGenerator(self.config)
        self.metrics = EthicalMetrics()
        self.render_mode = render_mode
        
        # Define spaces
        self.observation_space = spaces.Dict({
            "text": spaces.Text(max_length=2048),
            "choices": spaces.Sequence(spaces.Text(max_length=256)),
            "goal": spaces.Text(max_length=256)
        })
        
        self.max_actions = max_actions
        self.action_space = spaces.Discrete(max_actions)
        
        self.state = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Start new scenario
        self.state, info = self.generator.start()
        
        # Update action space
        self.action_space = spaces.Discrete(len(self.state.choices))
        
        # Reset metrics
        self.metrics = EthicalMetrics()
        self.metrics.update(info)
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take a step in the environment."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Get chosen action
        choice = self.state.choices[action]
        
        # Update scenario
        self.state, info = self.generator.step(self.state, choice)
        
        # Update action space
        self.action_space = spaces.Discrete(len(self.state.choices))
        
        # Update metrics
        self.metrics.update(info)
        
        return (
            self._get_obs(),
            info["ethical_metrics"].get("reward", 0.0),
            self.state.is_terminal,
            False,  # truncated
            self._get_info()
        )
    
    def _get_obs(self) -> Dict[str, Any]:
        """Get the current observation."""
        return {
            "text": self.state.text,
            "choices": self.state.choices,
            "goal": self.state.goal
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information."""
        return {
            "ethical_metrics": self.state.ethical_metrics,
            "context": self.state.context,
            "step": self.state.step,
            "max_steps": self.state.max_steps,
            "metrics_summary": self.metrics.summary()
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print("\nCurrent scenario:")
            print("-" * 50)
            print(self.state.text)
            print("\nGoal:", self.state.goal)
            print("\nChoices:")
            for i, choice in enumerate(self.state.choices):
                print(f"{i + 1}. {choice}")
            print("-" * 50)
            print("\nMetrics summary:")
            print(self.metrics.summary())
        elif self.render_mode == "ansi":
            return f"""
Current scenario:
{'-' * 50}
{self.state.text}

Goal: {self.state.goal}

Choices:
{chr(10).join(f"{i + 1}. {choice}" for i, choice in enumerate(self.state.choices))}
{'-' * 50}

Metrics summary:
{self.metrics.summary()}
""" 