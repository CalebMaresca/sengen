"""Ethical metrics tracking system for SenGen."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

@dataclass
class EthicalMetrics:
    """Tracks various ethical metrics throughout a scenario."""
    
    # Utilitarian metrics
    utility_scores: List[float] = field(default_factory=list)
    affected_parties: List[int] = field(default_factory=list)
    
    # Deontological metrics
    duty_adherence: List[float] = field(default_factory=list)
    rights_respected: List[float] = field(default_factory=list)
    
    # Virtue ethics metrics
    character_virtues: Dict[str, List[float]] = field(default_factory=lambda: {
        "honesty": [],
        "compassion": [],
        "justice": [],
        "courage": [],
        "wisdom": []
    })
    
    # Contextual metrics
    stakes: List[float] = field(default_factory=list)  # How high are the stakes
    uncertainty: List[float] = field(default_factory=list)  # Level of uncertainty
    time_pressure: List[float] = field(default_factory=list)  # Time pressure
    
    def update(self, info: Dict) -> None:
        """Update metrics based on environment info."""
        metrics = info.get("ethical_metrics", {})
        
        # Update utilitarian metrics
        self.utility_scores.append(metrics.get("utility_score", 0.0))
        self.affected_parties.append(metrics.get("affected_parties", 0))
        
        # Update deontological metrics
        self.duty_adherence.append(metrics.get("duty_adherence", 0.0))
        self.rights_respected.append(metrics.get("rights_respected", 0.0))
        
        # Update virtue metrics
        for virtue in self.character_virtues:
            self.character_virtues[virtue].append(
                metrics.get("virtues", {}).get(virtue, 0.0)
            )
        
        # Update contextual metrics
        self.stakes.append(metrics.get("stakes", 0.0))
        self.uncertainty.append(metrics.get("uncertainty", 0.0))
        self.time_pressure.append(metrics.get("time_pressure", 0.0))
    
    def summary(self) -> Dict:
        """Generate a summary of the metrics."""
        return {
            "utilitarian": {
                "mean_utility": np.mean(self.utility_scores),
                "total_affected": sum(self.affected_parties),
                "utility_trend": self._calculate_trend(self.utility_scores)
            },
            "deontological": {
                "mean_duty_adherence": np.mean(self.duty_adherence),
                "mean_rights_respected": np.mean(self.rights_respected)
            },
            "virtue_ethics": {
                virtue: np.mean(scores)
                for virtue, scores in self.character_virtues.items()
            },
            "context": {
                "mean_stakes": np.mean(self.stakes),
                "mean_uncertainty": np.mean(self.uncertainty),
                "mean_time_pressure": np.mean(self.time_pressure)
            }
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a pandas DataFrame for analysis."""
        data = {
            "utility_score": self.utility_scores,
            "affected_parties": self.affected_parties,
            "duty_adherence": self.duty_adherence,
            "rights_respected": self.rights_respected,
            "stakes": self.stakes,
            "uncertainty": self.uncertainty,
            "time_pressure": self.time_pressure
        }
        
        # Add virtue metrics
        for virtue, scores in self.character_virtues.items():
            data[f"virtue_{virtue}"] = scores
        
        return pd.DataFrame(data)
    
    def _calculate_trend(self, values: List[float]) -> Optional[float]:
        """Calculate the trend of a metric over time."""
        if not values:
            return None
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0] if len(values) > 1 else 0.0 