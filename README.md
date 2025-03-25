# SenGen: Scenario Generator for LLM RL Agents

SenGen is a Python package designed to generate interactive scenarios for training and evaluating LLM-based reinforcement learning agents, with a particular focus on ethical decision making and AI safety.

## Features

- Dynamic scenario generation using LLMs
- Built-in ethical metrics tracking
- Flexible environment configuration
- Support for custom agent implementations
- Tools for analyzing agent behavior
- Focus on ethical dilemmas and AI safety

## Installation

```bash
pip install sengen
```

## Quick Start

```python
from sengen.envs import EthicalStoryEnv
from sengen.agents import SimpleAgent
from sengen.metrics import EthicalMetrics

# Create an environment
env = EthicalStoryEnv(
    theme="resource allocation",
    max_steps=5
)

# Create an agent
agent = SimpleAgent()

# Run a scenario
obs, info = env.reset()
metrics = EthicalMetrics()

for _ in range(5):
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    metrics.update(info)
    
    if terminated or truncated:
        break

# Get metrics summary
print(metrics.summary())
```

## Configuration

SenGen uses YAML configuration files to define scenario parameters:

```yaml
scenario:
  theme: "AI safety"
  max_steps: 5
  temperature: 0.7

metrics:
  track_utilitarianism: true
  track_deontology: true
  track_virtue_ethics: true

agent:
  model: "gpt-4"
  temperature: 0.2
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 