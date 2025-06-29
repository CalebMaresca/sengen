"""Base agent interface and implementations for SenGen."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import yaml
import re

from ..core.model_providers import ModelProviderMixin, ModelConfig, ModelProvider

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    system_prompt: str = "You are a helpful AI assistant trying to acheive the goals given to you by the user while following ethical guidelines."
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    # New fields for model provider support
    provider: Optional[str] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    device: Optional[str] = None
    torch_dtype: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    extra_params: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """Load agent configuration from YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        agent_data = config.get("agent", {}).copy()
        return cls(**agent_data)


class Agent(ABC):
    """Abstract base class for all agents."""
    
    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> int:
        """Choose an action based on the current observation."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the agent's state."""
        pass


class LLMAgent(Agent, ModelProviderMixin):
    """A simple LLM-based agent."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        # Create model config for the provider system
        model_config_dict = {
            "model_name": config.model_name,
            "temperature": config.temperature,
            "provider": config.provider or ModelProvider.OPENAI,
            "max_tokens": config.max_tokens,
            "api_key": config.api_key,
            "base_url": config.base_url,
            "device": config.device,
            "torch_dtype": config.torch_dtype,
            "load_in_8bit": config.load_in_8bit,
            "load_in_4bit": config.load_in_4bit,
            "extra_params": config.extra_params or {}
        }
        
        model_config = self._parse_model_config(model_config_dict)
        self.llm = self._create_llm_from_config(model_config)
        self.model_provider = model_config.provider
        
        self.prompt = ChatPromptTemplate.from_messages([
            ('system', self.config.system_prompt),
            ('human', """Current situation:
{text}

Your goal: {goal}

Available choices:
{choices}

Choose a number between 0 and {highest_choice_number} (inclusive) corresponding to your choice. Respond with ONLY the number of your choice, with no other text or commentary.""")
        ])

        self.chain = self.prompt | self.llm
    
    def act(self, observation: Dict[str, Any]) -> int:
        """Choose an action based on the current observation."""
        response = self.chain.invoke(
            {"text": observation["text"],
            "goal": observation["goal"],
            "choices": "\n".join(f"{i}: {choice}" 
                                for i, choice in enumerate(observation["choices"])),
            "highest_choice_number": len(observation["choices"]) - 1}
        )
        
        # Extract the chosen action number from the response
        try:
            # Handle different response types based on provider
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Find the first integer in the response text
            match = re.search(r'\d+', response_text)
            if match:
                action = int(match.group())
                if 0 <= action < len(observation["choices"]):
                    return action
        except (ValueError, IndexError):
            pass
        
        # Default to first action if parsing fails
        print("Failed to parse action from response:", response)
        return 0
    
    def reset(self) -> None:
        """Reset the agent's state."""
        # No state to reset in this simple implementation
        pass 