from typing import Dict, List, Any
from dataclasses import dataclass, field
import numpy as np

from .config import SenGenConfig


@dataclass
class MetricsTracker:
    """Tracks various metrics throughout a scenario."""
    
    # Dynamic metrics storage
    metrics: Dict[str, List[Any]] = field(default_factory=dict)
    metric_types: Dict[str, str] = field(default_factory=dict)
    
    def __init__(self, config: SenGenConfig):
        """Initialize metrics tracking based on configuration."""
        self.metrics = {}
        self.metric_types = {}
        if config.metrics:
            for metric_name, metric_data in config.metrics.items():
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
    
    def get_metrics(self) -> Dict[str, List[Any]]:
        """Get current metrics."""
        return self.metrics.copy()
    
    def get_latest(self) -> Dict[str, Any]:
        """Get the latest metrics."""
        return {key: values[-1] if values else None for key, values in self.metrics.items()}
    
    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "latest": values[-1],
                    "average": np.mean(values) if metric_name in ["float", "int"] else None,
                    "total": np.sum(values) if metric_name in ["float", "int"] else None
                }
            else:
                summary[metric_name] = {
                    "count": 0,
                    "latest": None,
                    "average": None,
                    "total": None
                }
        return summary