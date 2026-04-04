"""
Endpoint Analysis Layer
Analyzes each of the 24 endpoints for abnormalities
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import statistics


class RiskLevel(Enum):
    NORMAL = "normal"
    WATCH = "watch"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class EndpointResult:
    endpoint: str
    risk_level: RiskLevel
    risk_score: float
    abnormal_values: List[Dict]
    patterns_detected: List[str]
    trend: Optional[str]
    confidence: float
    details: Dict

    def to_dict(self) -> Dict:
        return {
            'endpoint': self.endpoint,
            'risk_level': self.risk_level.value,
            'risk_score': self.risk_score,
            'abnormal_values': self.abnormal_values,
            'patterns_detected': self.patterns_detected,
            'trend': self.trend,
            'confidence': self.confidence,
            'details': self.details
        }


class EndpointAnalyzer:
    """
    Analyzes a single endpoint for abnormalities.
    """

    def __init__(self, reference_ranges: Dict):
        self.reference_ranges = reference_ranges

    def analyze(
        self,
        endpoint: str,
        data: Dict[str, List],
        reference_ranges: Dict
    ) -> EndpointResult:
        """
        Analyze an endpoint with whatever data is available.
        """
        abnormal_values = []
        patterns = []
        risk_scores = []

        for column, values in data.items():
            ref_range = self._find_reference_range(column, reference_ranges)

            for i, value in enumerate(values):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue

                try:
                    numeric_value = float(value)
                except (ValueError, TypeError):
                    continue

                if ref_range:
                    low, high = ref_range
                    if numeric_value < low:
                        severity = self._calculate_severity(numeric_value, low, 'low')
                        abnormal_values.append({
                            'column': column,
                            'value': numeric_value,
                            'reference': f"{low}-{high}",
                            'status': 'low',
                            'severity': severity,
                            'patient_index': i
                        })
                        risk_scores.append(severity * 100)
                    elif numeric_value > high:
                        severity = self._calculate_severity(numeric_value, high, 'high')
                        abnormal_values.append({
                            'column': column,
                            'value': numeric_value,
                            'reference': f"{low}-{high}",
                            'status': 'high',
                            'severity': severity,
                            'patient_index': i
                        })
                        risk_scores.append(severity * 100)

            column_patterns = self._detect_patterns(column, values)
            patterns.extend(column_patterns)

        if risk_scores:
            overall_risk = min(100, max(risk_scores) * 0.6 + np.mean(risk_scores) * 0.4)
        else:
            overall_risk = 0

        if overall_risk >= 80:
            risk_level = RiskLevel.CRITICAL
        elif overall_risk >= 60:
            risk_level = RiskLevel.WARNING
        elif overall_risk >= 40:
            risk_level = RiskLevel.WATCH
        else:
            risk_level = RiskLevel.NORMAL

        trend = self._analyze_trend(data)

        return EndpointResult(
            endpoint=endpoint,
            risk_level=risk_level,
            risk_score=overall_risk,
            abnormal_values=abnormal_values,
            patterns_detected=patterns,
            trend=trend,
            confidence=min(1.0, len(data) / 5),
            details={
                'columns_analyzed': list(data.keys()),
                'values_checked': sum(len(v) for v in data.values()),
                'abnormalities_found': len(abnormal_values)
            }
        )

    def _find_reference_range(self, column: str, reference_ranges: Dict) -> Optional[Tuple[float, float]]:
        """Find reference range for a column."""
        normalized = column.lower().replace(' ', '_').replace('-', '_')
        for key, range_tuple in reference_ranges.items():
            if key in normalized or normalized in key:
                return range_tuple
        return None

    def _calculate_severity(self, value: float, threshold: float, direction: str) -> float:
        """Calculate severity of abnormality (0-1)."""
        if direction == 'low':
            if threshold == 0:
                return 0.5
            deviation = (threshold - value) / threshold
        else:
            deviation = (value - threshold) / threshold

        return min(1.0, max(0, deviation))

    def _detect_patterns(self, column: str, values: List) -> List[str]:
        """Detect clinical patterns in the data."""
        patterns = []

        numeric_values = []
        for v in values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                continue

        if not numeric_values:
            return patterns

        if len(numeric_values) >= 3:
            if all(numeric_values[i] < numeric_values[i+1] for i in range(len(numeric_values)-1)):
                patterns.append(f"rising_trend_{column}")
            if all(numeric_values[i] > numeric_values[i+1] for i in range(len(numeric_values)-1)):
                patterns.append(f"falling_trend_{column}")

        if len(numeric_values) >= 3:
            try:
                mean_val = statistics.mean(numeric_values)
                if mean_val != 0:
                    cv = statistics.stdev(numeric_values) / mean_val
                    if cv > 0.3:
                        patterns.append(f"unstable_{column}")
            except:
                pass

        return patterns

    def _analyze_trend(self, data: Dict[str, List]) -> Optional[str]:
        """Analyze overall trend across all columns."""
        return None


class MultiEndpointAnalyzer:
    """
    Analyzes all available endpoints.
    """

    def __init__(self, reference_ranges: Dict[str, Dict]):
        self.reference_ranges = reference_ranges

    def analyze_all(self, endpoint_data: Dict[str, Dict]) -> Dict[str, EndpointResult]:
        """
        Analyze all endpoints that have data.
        """
        results = {}

        for endpoint, data in endpoint_data.items():
            if not data:
                continue

            ref_ranges = self.reference_ranges.get(endpoint, {}).get('reference_ranges', {})
            analyzer = EndpointAnalyzer(ref_ranges)
            result = analyzer.analyze(endpoint, data, ref_ranges)
            results[endpoint] = result

        return results
