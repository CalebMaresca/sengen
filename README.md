# SenGen: Scenario Generator for LLM RL Agents

SenGen is a Python package designed to generate interactive scenarios for training and evaluating LLM-based reinforcement learning agents, with a particular focus on ethical decision-making and AI safety.

Note: This project is in early development

## Features

- Dynamic scenario generation using LLMs
- Built-in metrics tracking
- Flexible environment configuration
- Multi-provider LLM support: OpenAI and Ollama

## Installation

Clone this repository, enter into the root directory and then run

```bash
pip install -e .
```

## Quick Start

Run `test_interactive.py` to manually play through scenarios as the agent.

**Note**: For OpenAI models, you will need to set your OpenAI API key as an environment variable on your system.

## Model Provider Support

SenGen can either use the OpenAI api, or local Ollama models to generate the scenarios.

## Configuration

SenGen uses YAML configuration files to define parameters such as model providor, scenario theme, metrics to track, etc. See `sengen/config_example.yaml` for an example.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 