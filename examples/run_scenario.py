"""Example usage of SenGen package."""

import os
import yaml
from sengen.envs.ethical_env import EthicalEnv
from sengen.agents.base import LLMAgent, AgentConfig

def main():
    # Load configurations
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = EthicalEnv(
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
    print(info["metrics_summary"])

if __name__ == "__main__":
    main() 