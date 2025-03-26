"""Unit tests for the ScenarioGenerator class."""

import pytest
from sengen.core.scenario import ScenarioGenerator, ScenarioConfig, ScenarioState
from sengen.core.metrics_tracker import MetricsTracker

@pytest.fixture
def config():
    """Create a basic scenario configuration for testing."""
    return ScenarioConfig(
        theme="AI safety",
        max_steps=3,
        temperature=0.7,
        model_name="gpt-4o-mini"
    )

@pytest.fixture
def generator(config):
    """Create a ScenarioGenerator instance for testing."""
    return ScenarioGenerator(config)

# def test_generator_initialization(config):
#     """Test that ScenarioGenerator initializes correctly with config."""
#     generator = ScenarioGenerator(config)
#     assert generator.config == config
#     assert generator.metrics is not None
#     assert isinstance(generator.metrics, MetricsTracker)

def test_start_scenario(generator):
    """Test starting a new scenario."""
    state, choices = generator.start()
    
    # Check state properties
    assert isinstance(state, str)
    print(state)
    print(type(state))
    assert isinstance(choices, list)
    print(choices)
    print(type(choices))
    # assert state.text is not None
    # assert isinstance(state.choices, list)
    # assert len(state.choices) > 0
    # assert state.goal is not None
    # assert state.step == 0
    # assert state.max_steps == 3
    # assert not state.is_terminal
    
    # # Check info properties
    # assert isinstance(info, dict)
    # assert "ethical_metrics" in info
    # assert isinstance(info["ethical_metrics"], dict)

# def test_step_scenario(generator):
#     """Test taking a step in the scenario."""
#     # Start scenario
#     state, _ = generator.start()
    
#     # Take a step
#     choice = state.choices[0]  # Choose first option
#     new_state, info = generator.step(state, choice)
    
#     # Check new state properties
#     assert isinstance(new_state, ScenarioState)
#     assert new_state.text is not None
#     assert isinstance(new_state.choices, list)
#     assert len(new_state.choices) > 0
#     assert new_state.step == 1
#     assert not new_state.is_terminal
    
#     # Check info properties
#     assert isinstance(info, dict)
#     assert "ethical_metrics" in info
#     assert isinstance(info["ethical_metrics"], dict)

# def test_scenario_termination(generator):
#     """Test that scenario terminates after max steps."""
#     # Start scenario
#     state, _ = generator.start()
    
#     # Take steps until termination
#     for _ in range(3):  # max_steps is 3
#         choice = state.choices[0]
#         state, _ = generator.step(state, choice)
    
#     # Check termination
#     assert state.is_terminal
#     assert state.step == 3

# def test_ethical_metrics_tracking(generator):
#     """Test that ethical metrics are properly tracked throughout the scenario."""
#     # Start scenario
#     state, info = generator.start()
    
#     # Check initial metrics
#     assert "ethical_metrics" in info
#     metrics = info["ethical_metrics"]
#     assert "utility_score" in metrics
#     assert "duty_adherence" in metrics
#     assert "rights_respected" in metrics
#     assert "virtues" in metrics
    
#     # Take a step and check metrics update
#     choice = state.choices[0]
#     _, new_info = generator.step(state, choice)
    
#     # Verify metrics are updated
#     assert "ethical_metrics" in new_info
#     new_metrics = new_info["ethical_metrics"]
#     assert new_metrics != metrics  # Metrics should change after action

# def test_invalid_choice(generator):
#     """Test handling of invalid choices."""
#     # Start scenario
#     state, _ = generator.start()
    
#     # Try to use an invalid choice
#     with pytest.raises(ValueError):
#         generator.step(state, "Invalid choice not in choices list")

# def test_scenario_consistency(generator):
#     """Test that scenario maintains consistency across steps."""
#     # Start scenario
#     state, _ = generator.start()
#     initial_context = state.context
    
#     # Take a step
#     choice = state.choices[0]
#     new_state, _ = generator.step(state, choice)
    
#     # Check context consistency
#     assert new_state.context is not None
#     assert new_state.context != initial_context  # Context should evolve
#     assert new_state.context.get("previous_choice") == choice  # Previous choice should be recorded 