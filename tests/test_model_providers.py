"""Tests for model provider functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
import yaml

from sengen.core.model_providers import (
    ModelProvider,
    ModelConfig,
    LLMProviderFactory,
    ModelProviderMixin,
    get_supported_models
)


class TestModelConfig:
    """Test ModelConfig functionality."""
    
    def test_model_config_creation(self):
        """Test creating a ModelConfig instance."""
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.7
        )
        
        assert config.provider == ModelProvider.OPENAI
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.extra_params == {}
    
    def test_model_config_with_string_provider(self):
        """Test creating ModelConfig with string provider."""
        config = ModelConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.7
        )
        
        assert config.provider == ModelProvider.OPENAI
    
    def test_model_config_with_extra_params(self):
        """Test ModelConfig with extra parameters."""
        extra_params = {"top_p": 0.95, "do_sample": True}
        config = ModelConfig(
            provider=ModelProvider.HUGGINGFACE,
            model_name="test-model",
            temperature=0.7,
            extra_params=extra_params
        )
        
        assert config.extra_params == extra_params


class TestLLMProviderFactory:
    """Test LLMProviderFactory functionality."""
    
    @patch('sengen.core.model_providers.ChatOpenAI')
    def test_create_openai_llm(self, mock_chat_openai):
        """Test creating OpenAI LLM."""
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.7,
            api_key="test-key"
        )
        
        mock_instance = Mock()
        mock_chat_openai.return_value = mock_instance
        
        result = LLMProviderFactory.create_llm(config)
        
        mock_chat_openai.assert_called_once_with(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key="test-key"
        )
        assert result == mock_instance
    
    @patch('sengen.core.model_providers.ChatOllama')
    def test_create_ollama_llm(self, mock_chat_ollama):
        """Test creating Ollama LLM."""
        config = ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="llama3",
            temperature=0.7,
            base_url="http://localhost:11434"
        )
        
        mock_instance = Mock()
        mock_chat_ollama.return_value = mock_instance
        
        result = LLMProviderFactory.create_llm(config)
        
        mock_chat_ollama.assert_called_once_with(
            model="llama3",
            temperature=0.7,
            base_url="http://localhost:11434"
        )
        assert result == mock_instance
    
    def test_create_ollama_llm_not_available(self):
        """Test error when Ollama is not available."""
        config = ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="llama3",
            temperature=0.7
        )
        
        with patch('sengen.core.model_providers.ChatOllama', None):
            with pytest.raises(ImportError, match="langchain_ollama is required"):
                LLMProviderFactory.create_llm(config)
    
    @patch('sengen.core.model_providers.ChatHuggingFace')
    @patch('sengen.core.model_providers.HuggingFacePipeline')
    def test_create_huggingface_llm(self, mock_hf_pipeline, mock_chat_hf):
        """Test creating HuggingFace LLM."""
        config = ModelConfig(
            provider=ModelProvider.HUGGINGFACE,
            model_name="test-model",
            temperature=0.7,
            device="cpu",
            max_tokens=256
        )
        
        mock_llm = Mock()
        mock_hf_pipeline.from_model_id.return_value = mock_llm
        mock_instance = Mock()
        mock_chat_hf.return_value = mock_instance
        
        result = LLMProviderFactory.create_llm(config)
        
        mock_hf_pipeline.from_model_id.assert_called_once()
        mock_chat_hf.assert_called_once_with(llm=mock_llm)
        assert result == mock_instance
    
    def test_create_huggingface_llm_not_available(self):
        """Test error when HuggingFace is not available."""
        config = ModelConfig(
            provider=ModelProvider.HUGGINGFACE,
            model_name="test-model",
            temperature=0.7
        )
        
        with patch('sengen.core.model_providers.ChatHuggingFace', None):
            with pytest.raises(ImportError, match="langchain_huggingface is required"):
                LLMProviderFactory.create_llm(config)
    
    def test_unsupported_provider(self):
        """Test error for unsupported provider."""
        # This should raise an error during ModelConfig creation
        with pytest.raises(ValueError, match="is not a valid ModelProvider"):
            config = ModelConfig(
                provider="unsupported",
                model_name="test-model", 
                temperature=0.7
            )


class TestModelProviderMixin:
    """Test ModelProviderMixin functionality."""
    
    def test_parse_model_config_legacy_format(self):
        """Test parsing legacy configuration format."""
        mixin = ModelProviderMixin()
        
        config_dict = {
            "model_name": "gpt-4o-mini",
            "temperature": 0.7
        }
        
        model_config = mixin._parse_model_config(config_dict)
        
        assert model_config.provider == ModelProvider.OPENAI
        assert model_config.model_name == "gpt-4o-mini"
        assert model_config.temperature == 0.7
    
    def test_parse_model_config_new_format(self):
        """Test parsing new configuration format."""
        mixin = ModelProviderMixin()
        
        config_dict = {
            "provider": "ollama",
            "model_name": "llama3",
            "temperature": 0.7,
            "base_url": "http://localhost:11434"
        }
        
        model_config = mixin._parse_model_config(config_dict)
        
        assert model_config.provider == ModelProvider.OLLAMA
        assert model_config.model_name == "llama3"
        assert model_config.temperature == 0.7
        assert model_config.base_url == "http://localhost:11434"
    
    @patch('sengen.core.model_providers.LLMProviderFactory.create_llm')
    def test_create_llm_from_config(self, mock_create_llm):
        """Test creating LLM from configuration."""
        mixin = ModelProviderMixin()
        
        config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.7
        )
        
        mock_llm = Mock()
        mock_create_llm.return_value = mock_llm
        
        result = mixin._create_llm_from_config(config)
        
        mock_create_llm.assert_called_once_with(config)
        assert result == mock_llm


class TestGetSupportedModels:
    """Test get_supported_models function."""
    
    def test_get_supported_models(self):
        """Test getting supported models information."""
        models_info = get_supported_models()
        
        assert "openai" in models_info
        assert "ollama" in models_info
        assert "huggingface" in models_info
        
        # Check structure of returned data
        for provider, info in models_info.items():
            assert "models" in info
            assert "description" in info
            assert "requirements" in info
            assert "setup" in info
            assert isinstance(info["models"], list)
            assert isinstance(info["requirements"], list) 