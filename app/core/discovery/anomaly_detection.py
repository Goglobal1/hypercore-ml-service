"""
Anomaly Detection Layer
Finds unusual patterns that warrant investigation
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Anomaly:
    anomaly_type: str
    severity: str
    score: float
    affected_values: List[Dict]
    description: str
    recommendation: str

    def to_dict(self) -> Dict:
        return {
            'anomaly_type': self.anomaly_type,
            'severity': self.severity,
            'score': self.score,
            'affected_values': self.affected_values,
            'description': self.description,
            'recommendation': self.recommendation
        }


class AnomalyDetector:
    """
    Detects anomalies using multiple methods.
    """

    def detect(self, endpoint_data: Dict[str, Dict]) -> List[Anomaly]:
        """
        Detect anomalies across all data.
        """
        anomalies = []

        for endpoint, data in endpoint_data.items():
            outliers = self._detect_statistical_outliers(endpoint, data)
            anomalies.extend(outliers)

            combinations = self._detect_unusual_combinations(endpoint, data)
            anomalies.extend(combinations)

            changes = self._detect_rapid_changes(endpoint, data)
            anomalies.extend(changes)

        cross_system = self._detect_cross_system_anomalies(endpoint_data)
        anomalies.extend(cross_system)

        return anomalies

    def _detect_statistical_outliers(
        self,
        endpoint: str,
        data: Dict[str, List]
    ) -> List[Anomaly]:
        """Detect statistical outliers (>3 std from mean)."""
        anomalies = []

        for column, values in data.items():
            numeric = []
            for v in values:
                try:
                    numeric.append(float(v))
                except:
                    pass

            if len(numeric) < 3:
                continue

            mean = np.mean(numeric)
            std = np.std(numeric)
            if std == 0:
                continue

            for i, v in enumerate(numeric):
                z = (v - mean) / std
                if abs(z) > 3:
                    anomalies.append(Anomaly(
                        anomaly_type="statistical_outlier",
                        severity="severe" if abs(z) > 4 else "moderate",
                        score=min(100, abs(z) * 20),
                        affected_values=[{
                            'column': column,
                            'value': v,
                            'z_score': z,
                            'patient_index': i
                        }],
                        description=f"Extreme outlier in {column}: {v} (z={z:.1f})",
                        recommendation=f"Verify value is correct. If accurate, investigate cause."
                    ))

        return anomalies

    def _detect_unusual_combinations(
        self,
        endpoint: str,
        data: Dict[str, List]
    ) -> List[Anomaly]:
        """Detect unusual value combinations."""
        anomalies = []
        # Placeholder for combination detection
        return anomalies

    def _detect_rapid_changes(
        self,
        endpoint: str,
        data: Dict[str, List]
    ) -> List[Anomaly]:
        """Detect rapid changes in values (requires temporal data)."""
        anomalies = []

        for column, values in data.items():
            numeric = []
            for v in values:
                try:
                    numeric.append(float(v))
                except:
                    pass

            if len(numeric) < 2:
                continue

            for i in range(1, len(numeric)):
                if numeric[i-1] == 0:
                    continue
                change = numeric[i] - numeric[i-1]
                pct_change = abs(change / numeric[i-1])

                if pct_change > 0.5:
                    anomalies.append(Anomaly(
                        anomaly_type="rapid_change",
                        severity="severe" if pct_change > 1.0 else "moderate",
                        score=min(100, pct_change * 100),
                        affected_values=[{
                            'column': column,
                            'from': numeric[i-1],
                            'to': numeric[i],
                            'change': change,
                            'pct_change': pct_change
                        }],
                        description=f"Rapid {'increase' if change > 0 else 'decrease'} in {column}: {pct_change*100:.0f}%",
                        recommendation="Investigate cause of rapid change"
                    ))

        return anomalies

    def _detect_cross_system_anomalies(
        self,
        endpoint_data: Dict[str, Dict]
    ) -> List[Anomaly]:
        """Detect anomalies across multiple systems."""
        anomalies = []
        # Placeholder for cross-system anomaly detection
        return anomalies
