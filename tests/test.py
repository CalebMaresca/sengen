"""Example usage of SenGen package."""

import os
# import yaml
# from sengen.envs.ethical_env import EthicalEnv
# from sengen.agents.base import LLMAgent, AgentConfig
from sengen.core.scenario import ScenarioConfig, ScenarioGenerator
def main():
    # Load configurations
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = ScenarioConfig.from_yaml(config_path)

    SenGenOutput = config.create_output_model()

    # import json
    # print(json.dumps(SenGenOutput.model_json_schema(), indent=2))
    
    # Create environment
    generator = ScenarioGenerator(config)
    print("Step 0:")
    state, choices = generator.start()
    print(state)
    print(choices)
    print("--------------------------------")
    print("Step 1:")
    state, choices = generator.step(choices[0])
    print(state)
    print(choices)
    print("--------------------------------")
    print("History:")
    print(generator.chat_history)

if __name__ == "__main__":
    main() 