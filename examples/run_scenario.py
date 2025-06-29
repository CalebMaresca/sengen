"""Example usage of SenGen package."""

import os
import yaml
from sengen.envs.gym_env import SenGenGymEnv
from sengen.agents.base import LLMAgent, AgentConfig

def main():
    # Load configurations
    config_path = os.path.join(os.path.dirname(__file__), "config_mixed.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = SenGenGymEnv(
        config_path=config_path,
        render_mode="human"
    )
    
    # Create agent
    agent = LLMAgent(AgentConfig(**config["agent"]))
    
    # Run scenario
    obs, info = env.reset()
    total_reward = 0
    
    print("\nStarting scenario simulation...")
    print("=" * 80)
    
    while True:
        # Render current state
        env.render()
        
        # Get agent's action
        action = agent.act(obs)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print("\nScenario completed!")
    print("=" * 80)
    print(f"Total reward: {total_reward}")
    print("\nFinal metrics summary:")
    metrics_summary = env.metrics_tracker.summary()
    for metric_name, metric_data in metrics_summary.items():
        print(f"\n{metric_name}:")
        for key, value in metric_data.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 