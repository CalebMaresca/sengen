"""Integration tests for the scenario generation system focusing on metrics."""

import pytest
from unittest.mock import MagicMock, patch
import tempfile
import yaml
import os
import json
from pydantic import ValidationError

from sengen.core.scenario import (
    MetricDefinition,
    Metric,
    ScenarioConfig,
    ScenarioGenerator
)

@pytest.fixture
def complex_metrics_config():
    """Configuration with various metric types for testing."""
    return {
        "scenario": {
            "theme": "Ethical AI",
            "max_steps": 5,
            "temperature": 0.7,
            "model_name": "gpt-4",
            "metrics": {
                "utility_score": {
                    "name": "Utility Score",
                    "description": "Measures the overall positive outcomes (0-10)",
                    "type": "float"
                },
                "harm_risk": {
                    "name": "Harm Risk",
                    "description": "Risk level of causing harm to individuals",
                    "type": "str"
                },
                "rights_respected": {
                    "name": "Rights Respected",
                    "description": "Count of rights respected in the decision",
                    "type": "int"
                },
                "fairness": {
                    "name": "Fairness",
                    "description": "Whether the decision is fair to all parties",
                    "type": "bool"
                }
            }
        }
    }

@pytest.fixture
def complex_config_file(complex_metrics_config):
    """Create a temporary config file with complex metrics."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.dump(complex_metrics_config, temp_file)
        temp_path = temp_file.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

@pytest.fixture
def mock_response_factory():
    """Factory to create mock responses with different metric values."""
    def create_response(metrics_list):
        class MockResponse:
            def __init__(self):
                self.state = "Test scenario state with ethical implications."
                self.choices = ["Option A (safe)", "Option B (risky but beneficial)", "Option C (neutral)"]
                self.metrics = metrics_list
        
        return MockResponse()
    
    return create_response


class TestMetricValidation:
    """Tests for metric validation."""
    
    def test_valid_metrics(self, complex_config_file):
        """Test validation accepts correct metrics."""
        config = ScenarioConfig.from_yaml(complex_config_file)
        output_model = config.create_output_model()
        
        # Create valid metrics matching the configuration
        valid_data = {
            "state": "Test scenario",
            "choices": ["A", "B", "C"],
            "metrics": [
                {"name": "Utility Score", "value": 7.5, "description": "Measures the overall positive outcomes (0-10)"},
                {"name": "Harm Risk", "value": "Low", "description": "Risk level of causing harm to individuals"},
                {"name": "Rights Respected", "value": 3, "description": "Count of rights respected in the decision"},
                {"name": "Fairness", "value": True, "description": "Whether the decision is fair to all parties"}
            ]
        }
        
        # Should validate without error
        instance = output_model.model_validate(valid_data)
        assert len(instance.metrics) == 4
    
    def test_missing_metrics(self, complex_config_file):
        """Test validation rejects when metrics are missing."""
        config = ScenarioConfig.from_yaml(complex_config_file)
        output_model = config.create_output_model()
        
        # Create data with a missing metric
        invalid_data = {
            "state": "Test scenario",
            "choices": ["A", "B", "C"],
            "metrics": [
                {"name": "Utility Score", "value": 7.5, "description": "Measures the overall positive outcomes (0-10)"},
                {"name": "Harm Risk", "value": "Low", "description": "Risk level of causing harm to individuals"},
                # Missing Rights Respected
                {"name": "Fairness", "value": True, "description": "Whether the decision is fair to all parties"}
            ]
        }
        
        # Should raise validation error
        with pytest.raises(ValidationError) as excinfo:
            output_model.model_validate(invalid_data)
        assert "Missing required metrics" in str(excinfo.value)
        assert "Rights Respected" in str(excinfo.value)
    
    def test_unexpected_metrics(self, complex_config_file):
        """Test validation rejects when unexpected metrics are provided."""
        config = ScenarioConfig.from_yaml(complex_config_file)
        output_model = config.create_output_model()
        
        # Create data with an extra, unexpected metric
        invalid_data = {
            "state": "Test scenario",
            "choices": ["A", "B", "C"],
            "metrics": [
                {"name": "Utility Score", "value": 7.5, "description": "Measures the overall positive outcomes (0-10)"},
                {"name": "Harm Risk", "value": "Low", "description": "Risk level of causing harm to individuals"},
                {"name": "Rights Respected", "value": 3, "description": "Count of rights respected in the decision"},
                {"name": "Fairness", "value": True, "description": "Whether the decision is fair to all parties"},
                {"name": "Unexpected Metric", "value": 42, "description": "This shouldn't be here"}
            ]
        }
        
        # Should raise validation error
        with pytest.raises(ValidationError) as excinfo:
            output_model.model_validate(invalid_data)
        assert "Unexpected metrics provided" in str(excinfo.value)
        assert "Unexpected Metric" in str(excinfo.value)


class TestMetricTypeHandling:
    """Tests for handling different metric types."""
    
    def test_metric_type_conversion(self, complex_config_file):
        """Test that different metric types are handled correctly."""
        config = ScenarioConfig.from_yaml(complex_config_file)
        
        # Verify all metric types were loaded correctly
        assert config.metrics["utility_score"].type == "float"
        assert config.metrics["harm_risk"].type == "str"
        assert config.metrics["rights_respected"].type == "int"
        assert config.metrics["fairness"].type == "bool"
        
        # Create the output model
        output_model = config.create_output_model()
        
        # Test model serialization/deserialization with different types
        test_data = {
            "state": "Test scenario",
            "choices": ["A", "B", "C"],
            "metrics": [
                {"name": "Utility Score", "value": 7.5, "description": "Measures the overall positive outcomes (0-10)"},
                {"name": "Harm Risk", "value": "Low", "description": "Risk level of causing harm to individuals"},
                {"name": "Rights Respected", "value": 3, "description": "Count of rights respected in the decision"},
                {"name": "Fairness", "value": True, "description": "Whether the decision is fair to all parties"}
            ]
        }
        
        # Create model instance and verify types
        instance = output_model.model_validate(test_data)
        assert isinstance(instance.metrics[0].value, float)
        assert isinstance(instance.metrics[1].value, str)
        assert isinstance(instance.metrics[2].value, int)
        assert isinstance(instance.metrics[3].value, bool)
        
        # Test serialization
        serialized = instance.model_dump_json()
        deserialized = json.loads(serialized)
        assert deserialized["metrics"][0]["value"] == 7.5
        assert deserialized["metrics"][1]["value"] == "Low"
        assert deserialized["metrics"][2]["value"] == 3
        assert deserialized["metrics"][3]["value"] == True


class TestScenarioIntegration:
    """Integration tests for the scenario generation system."""
    
    @patch("sengen.core.scenario.ChatOpenAI")
    def test_metrics_prompt_generation(self, mock_chat_openai, complex_config_file):
        """Test that metrics are properly included in the prompt."""
        config = ScenarioConfig.from_yaml(complex_config_file)
        
        # Mock the LLM
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value.with_structured_output.return_value = mock_llm_instance
        
        generator = ScenarioGenerator(config)
        
        # Extract the system prompt template
        system_message = generator.prompt.messages[0]
        system_template = system_message.prompt.template
        
        # Check that all metrics are included in the prompt
        assert "Utility Score" in system_template
        assert "Harm Risk" in system_template
        assert "Rights Respected" in system_template
        assert "Fairness" in system_template
        
        # Check instructions for metrics format
        assert "provide all metrics as a list in the 'metrics' field" in system_template
        assert "You MUST include exactly these metrics" in system_template
    
    @patch("sengen.core.scenario.ChatOpenAI")
    def test_metrics_evolution(self, mock_chat_openai, complex_config_file, mock_response_factory):
        """Test that metrics evolve through scenario steps."""
        config = ScenarioConfig.from_yaml(complex_config_file)
        
        # Create initial metrics
        initial_metrics = [
            Metric(name="Utility Score", value=5.0, description="Measures the overall positive outcomes (0-10)"),
            Metric(name="Harm Risk", value="Medium", description="Risk level of causing harm to individuals"),
            Metric(name="Rights Respected", value=2, description="Count of rights respected in the decision"),
            Metric(name="Fairness", value=True, description="Whether the decision is fair to all parties")
        ]
        
        # Create evolved metrics after a choice
        evolved_metrics = [
            Metric(name="Utility Score", value=7.5, description="Measures the overall positive outcomes (0-10)"),
            Metric(name="Harm Risk", value="Low", description="Risk level of causing harm to individuals"),
            Metric(name="Rights Respected", value=3, description="Count of rights respected in the decision"),
            Metric(name="Fairness", value=True, description="Whether the decision is fair to all parties")
        ]
        
        # Create mock responses
        initial_response = mock_response_factory(initial_metrics)
        evolved_response = mock_response_factory(evolved_metrics)
        
        # Mock chain responses for start and step
        mock_chain_with_history = MagicMock()
        mock_chain_with_history.invoke.side_effect = [initial_response, evolved_response]
        
        # Create scenario generator with mocks
        with patch("sengen.core.scenario.RunnableWithMessageHistory", return_value=mock_chain_with_history):
            generator = ScenarioGenerator(config)
            
            # Start the scenario
            state, choices, metrics = generator.start()
            
            # Verify initial metrics
            assert metrics[0].name == "Utility Score"
            assert metrics[0].value == 5.0
            assert metrics[1].name == "Harm Risk"
            assert metrics[1].value == "Medium"
            
            # Take a step
            new_state, new_choices, new_metrics = generator.step("Option B (risky but beneficial)")
            
            # Verify evolved metrics
            assert new_metrics[0].name == "Utility Score"
            assert new_metrics[0].value == 7.5  # Increased from 5.0
            assert new_metrics[1].name == "Harm Risk"
            assert new_metrics[1].value == "Low"  # Changed from Medium


class TestEmptyMetricsHandling:
    """Tests for handling cases with no metrics defined."""
    
    @pytest.fixture
    def empty_metrics_config(self):
        """Configuration with no metrics defined."""
        return {
            "scenario": {
                "theme": "Simple Theme",
                "max_steps": 3,
                "temperature": 0.5,
                "model_name": "test-model",
                # No metrics defined
            }
        }
    
    @pytest.fixture
    def empty_metrics_file(self, empty_metrics_config):
        """Create a temporary config file with no metrics."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            yaml.dump(empty_metrics_config, temp_file)
            temp_path = temp_file.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def test_empty_metrics_handling(self, empty_metrics_file):
        """Test that the system handles cases with no metrics defined."""
        config = ScenarioConfig.from_yaml(empty_metrics_file)
        
        # Verify metrics is None or empty
        assert config.metrics is None or len(config.metrics) == 0
        
        # Create output model
        output_model = config.create_output_model()
        
        # Verify model still has state and choices fields
        assert hasattr(output_model, "__annotations__")
        assert "state" in output_model.__annotations__
        assert "choices" in output_model.__annotations__
        assert "metrics" in output_model.__annotations__
        
        # Test instantiation with empty metrics
        instance = output_model(state="Test state", choices=["A", "B"], metrics=[])
        assert instance.state == "Test state"
        assert instance.choices == ["A", "B"]
        assert instance.metrics == [] 