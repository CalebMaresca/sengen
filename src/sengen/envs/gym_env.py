"""Gymnasium environment for SenGen scenarios."""

from typing import Dict, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces
from ..core.scenario import ScenarioGenerator, ScenarioConfig
from ..core.metrics_tracker import MetricsTracker

class SenGenGymEnv(gym.Env):
    """Environment for SenGen scenarios."""
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        scenario_config: Optional[ScenarioConfig] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Load config or create from parameters
        if config_path and scenario_config:
            raise ValueError("Cannot provide both config_path and scenario_config")
        
        if config_path:
            self.config = ScenarioConfig.from_yaml(config_path)
        elif scenario_config:
            self.config = scenario_config
        else:
            raise ValueError("Either config_path or scenario_config must be provided")
        
        # Initialize components
        self.generator = ScenarioGenerator(self.config)
        self.metrics_tracker = MetricsTracker(self.config)
        self.render_mode = render_mode
        
        # Define spaces
        self.observation_space = spaces.Dict({
            "text": spaces.Text(max_length=2048),
            "choices": spaces.Sequence(spaces.Text(max_length=256))
        })
        
        self.action_space = spaces.Discrete(self.config.max_choices)
        
        self.state = None
        self.choices = None
        self.last_action = None
        self.reward = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Start new scenario
        self.state, self.choices = self.generator.start()
        self.last_action = None
        self.reward = None
        
        # Reset metrics
        self.metrics_tracker.reset()
        
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take a step in the environment."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
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
            metrics
        )
    
    def _get_obs(self) -> Dict[str, Any]:
        """Get the current observation."""
        return {
            "text": self.state,
            "choices": [choice.text for choice in self.choices],
            "goal": self.config.goal
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