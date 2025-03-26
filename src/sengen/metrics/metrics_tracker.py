"""Ethical metrics tracking system for SenGen."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from ..core.scenario import ScenarioConfig
@dataclass
class MetricsTracker:
    """Tracks various metrics throughout a scenario."""
    
    # Dynamic metrics storage
    metrics: Dict[str, List[Any]] = field(default_factory=dict)
    metric_types: Dict[str, str] = field(default_factory=dict)
    
    def __init__(self, config: ScenarioConfig):
        """Initialize metrics tracking based on configuration."""
        self.metrics = {}
        self.metric_types = {}
        if config.metrics:
            metrics_config = config.metrics
            for metric_name, metric_data in metrics_config.items():
                if isinstance(metric_data, dict):
                    self.metric_types[metric_name] = metric_data.get("type", "float")
                    self.metrics[metric_name] = []
    
    def update(self, metrics: Dict) -> None:
        """Update metrics based on environment info."""
        for metric_name in self.metrics:
            value = metrics.get(metric_name)
            if value is not None:
                self.metrics[metric_name].append(value)

    def reset(self) -> None:
        """Reset the metrics tracker."""
        self.metrics = {key: [] for key in self.metrics}
    
    def summary(self) -> Dict:
        """Generate a summary of the metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if not values:
                continue
                
            metric_type = self.metric_types.get(metric_name, "float")
            if metric_type in ["float", "int"]:
                summary[metric_name] = {
                    "mean": np.mean(values),
                    "trend": self._calculate_trend(values)
                }
            elif metric_type == "bool":
                summary[metric_name] = {
                    "true_count": sum(1 for v in values if v),
                    "false_count": sum(1 for v in values if not v)
                }
            else:  # str or other types
                summary[metric_name] = {
                    "values": values,
                    "unique_values": list(set(values))
                }
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a pandas DataFrame for analysis."""
        data = {}
        for metric_name, values in self.metrics.items():
            data[metric_name] = values
        return pd.DataFrame(data)
    
    def _calculate_trend(self, values: List[float]) -> Optional[float]:
        """Calculate the trend of a metric over time."""
        if not values:
            return None
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0] if len(values) > 1 else 0.0 