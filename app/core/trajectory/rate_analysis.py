"""
Rate of Change Analysis - Layer 1 of Trajectory System

Detects abnormal rates of change in biomarkers.
Key insight: A biomarker rising 20%/day while still "normal"
is more dangerous than one at threshold but stable.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RateOfChangeResult:
    biomarker: str
    baseline_rate: float      # Normal daily variance (%)
    current_rate: float       # Current daily change (%)
    acceleration: float       # Rate of rate change (2nd derivative)
    z_score: float           # How many std devs from normal
    alert_level: str         # normal, elevated, warning, critical
    days_of_trend: int       # How long has this been trending
    confidence: float        # Statistical confidence


class RateOfChangeAnalyzer:
    """
    Detects abnormal rates of change in biomarkers.
    """

    # Normal daily variance by biomarker (based on clinical literature)
    NORMAL_DAILY_VARIANCE = {
        'procalcitonin': 0.08,   # +/-8% normal daily fluctuation
        'lactate': 0.12,         # +/-12%
        'wbc': 0.10,             # +/-10%
        'crp': 0.15,             # +/-15% (more volatile)
        'troponin': 0.05,        # +/-5% (very stable normally)
        'bnp': 0.10,             # +/-10%
        'creatinine': 0.05,      # +/-5% (very stable)
        'il6': 0.20,             # +/-20% (highly volatile)
        'ferritin': 0.08,        # +/-8%
        'd_dimer': 0.15,         # +/-15%
        'egfr': 0.03,            # +/-3% (very stable)
        'alt': 0.10,
        'ast': 0.10,
        'bilirubin': 0.08,
        'albumin': 0.05,
        'platelets': 0.08,
        'hemoglobin': 0.03,
        'glucose': 0.15,
        'potassium': 0.05,
        'sodium': 0.02,
    }

    # Alert thresholds (multiples of normal variance)
    ALERT_THRESHOLDS = {
        'elevated': 2.0,    # 2x normal variance
        'warning': 3.5,     # 3.5x normal variance
        'critical': 5.0,    # 5x normal variance
    }

    def analyze_patient_trajectory(
        self,
        patient_data: Dict[str, List[float]],
        timestamps: List[float]
    ) -> Dict[str, RateOfChangeResult]:
        """
        Analyze rate of change for all biomarkers in patient data.
        """
        results = {}

        for biomarker, values in patient_data.items():
            if len(values) < 3:
                continue

            # Skip non-numeric or ID columns
            if biomarker.lower() in ['patient_id', 'id', 'day', 'time', 'timestamp', 'outcome', 'label']:
                continue

            result = self._analyze_single_biomarker(
                biomarker, values, timestamps
            )
            if result:
                results[biomarker] = result

        return results

    def _analyze_single_biomarker(
        self,
        biomarker: str,
        values: List[float],
        timestamps: List[float]
    ) -> Optional[RateOfChangeResult]:
        """
        Analyze trajectory of a single biomarker.
        """
        try:
            values = np.array([float(v) for v in values if v is not None and not np.isnan(float(v))])
            timestamps = np.array([float(t) for t in timestamps[:len(values)]])
        except (ValueError, TypeError):
            return None

        if len(values) < 3:
            return None

        # Calculate daily rates of change
        daily_changes = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                time_delta = timestamps[i] - timestamps[i-1]
                if time_delta > 0:
                    pct_change = (values[i] - values[i-1]) / values[i-1]
                    daily_rate = pct_change / time_delta
                    daily_changes.append(daily_rate)

        if not daily_changes:
            return None

        daily_changes = np.array(daily_changes)

        # Current rate (average of recent changes)
        current_rate = np.mean(daily_changes[-3:]) if len(daily_changes) >= 3 else np.mean(daily_changes)

        # Baseline rate (expected normal variance)
        baseline_rate = self.NORMAL_DAILY_VARIANCE.get(biomarker.lower(), 0.10)

        # Calculate acceleration
        if len(daily_changes) >= 3:
            acceleration = np.polyfit(range(len(daily_changes)), daily_changes, 1)[0]
        else:
            acceleration = 0.0

        # Z-score
        z_score = abs(current_rate) / baseline_rate if baseline_rate > 0 else 0

        # Alert level
        if z_score >= self.ALERT_THRESHOLDS['critical']:
            alert_level = 'critical'
        elif z_score >= self.ALERT_THRESHOLDS['warning']:
            alert_level = 'warning'
        elif z_score >= self.ALERT_THRESHOLDS['elevated']:
            alert_level = 'elevated'
        else:
            alert_level = 'normal'

        # Days of consistent trend
        days_of_trend = self._count_trend_days(daily_changes)

        # Statistical confidence
        if len(values) >= 5:
            try:
                _, p_value = stats.pearsonr(timestamps[:len(values)], values)
                confidence = 1.0 - p_value
            except:
                confidence = 0.5 + (len(values) * 0.1)
        else:
            confidence = 0.5 + (len(values) * 0.1)

        return RateOfChangeResult(
            biomarker=biomarker,
            baseline_rate=baseline_rate,
            current_rate=float(current_rate),
            acceleration=float(acceleration),
            z_score=float(z_score),
            alert_level=alert_level,
            days_of_trend=days_of_trend,
            confidence=min(float(confidence), 0.99)
        )

    def _count_trend_days(self, daily_changes: np.ndarray) -> int:
        """Count consecutive days of same-direction change."""
        if len(daily_changes) == 0:
            return 0

        count = 1
        direction = np.sign(daily_changes[-1])

        for i in range(len(daily_changes) - 2, -1, -1):
            if np.sign(daily_changes[i]) == direction:
                count += 1
            else:
                break

        return count
