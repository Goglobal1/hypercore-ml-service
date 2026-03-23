"""
Forecasting and Early Warning Engine - Layers 4-6 of Trajectory System

Predicts when biomarkers will cross thresholds based on trajectory.
Main integration point for the complete trajectory analysis system.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .rate_analysis import RateOfChangeAnalyzer, RateOfChangeResult
from .inflection_detection import InflectionDetector, InflectionPoint
from .pattern_library import PatternLibrary, PatternMatch, DiseasePattern


@dataclass
class ForecastResult:
    biomarker: str
    current_value: float
    threshold: float
    predicted_crossing_day: float
    prediction_confidence: float
    trajectory_model: str


@dataclass
class EarlyWarningReport:
    patient_id: str
    analysis_timestamp: str

    # Overall
    risk_level: str
    confidence: float

    # Timing
    estimated_days_to_event: float
    estimation_range: Tuple[float, float]
    detection_improvement_days: float

    # Patterns
    matched_patterns: List[dict]
    primary_pattern: Optional[str]

    # Details
    rate_changes: Dict[str, dict]
    inflection_points: Dict[str, List[dict]]
    forecasts: Dict[str, dict]

    # Cascade
    earliest_signal_biomarker: str
    earliest_signal_days_ago: float
    signal_propagation_order: List[str]

    # Recommendations
    clinical_recommendations: List[str]
    monitoring_recommendations: Dict[str, str]
    genetic_recommendations: List[str]


class TrajectoryForecaster:
    """Predicts when biomarkers will cross thresholds."""

    THRESHOLDS = {
        'procalcitonin': 0.5,
        'lactate': 2.0,
        'wbc': 12.0,
        'crp': 10.0,
        'troponin': 0.04,
        'bnp': 100.0,
        'creatinine': 1.3,
        'egfr': 60.0,
        'bilirubin': 1.2,
        'alt': 40.0,
        'ast': 40.0,
        'glucose': 126.0,
        'potassium': 5.0,
    }

    CONCERN_DIRECTION = {
        'egfr': 'below',
        'platelets': 'below',
        'albumin': 'below',
        'hemoglobin': 'below',
    }

    def forecast_threshold_crossing(
        self,
        biomarker: str,
        values: List[float],
        timestamps: List[float]
    ) -> Optional[ForecastResult]:
        """Predict when a biomarker will cross its clinical threshold."""
        if len(values) < 3:
            return None

        threshold = self.THRESHOLDS.get(biomarker.lower())
        if threshold is None:
            return None

        try:
            values = np.array([float(v) for v in values])
            timestamps = np.array([float(t) for t in timestamps[:len(values)]])
        except:
            return None

        current_value = values[-1]
        direction = self.CONCERN_DIRECTION.get(biomarker.lower(), 'above')

        # Already past threshold?
        if direction == 'above' and current_value >= threshold:
            return ForecastResult(
                biomarker=biomarker,
                current_value=float(current_value),
                threshold=threshold,
                predicted_crossing_day=0,
                prediction_confidence=1.0,
                trajectory_model='already_exceeded'
            )
        elif direction == 'below' and current_value <= threshold:
            return ForecastResult(
                biomarker=biomarker,
                current_value=float(current_value),
                threshold=threshold,
                predicted_crossing_day=0,
                prediction_confidence=1.0,
                trajectory_model='already_exceeded'
            )

        # Try linear forecast
        result = self._linear_forecast(values, timestamps, threshold, direction, biomarker)
        if result:
            return result

        # Try exponential forecast
        result = self._exponential_forecast(values, timestamps, threshold, direction, biomarker)
        return result

    def _linear_forecast(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        threshold: float,
        direction: str,
        biomarker: str
    ) -> Optional[ForecastResult]:
        """Linear extrapolation to threshold."""
        try:
            slope, intercept = np.polyfit(timestamps, values, 1)

            predicted = intercept + slope * timestamps
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            if slope == 0:
                return None

            current_time = timestamps[-1]
            time_to_threshold = (threshold - intercept) / slope - current_time

            if direction == 'above' and slope <= 0:
                return None
            if direction == 'below' and slope >= 0:
                return None
            if time_to_threshold < 0:
                return None

            return ForecastResult(
                biomarker=biomarker,
                current_value=float(values[-1]),
                threshold=threshold,
                predicted_crossing_day=float(time_to_threshold),
                prediction_confidence=float(min(r_squared * 0.8, 0.95)),
                trajectory_model='linear'
            )
        except:
            return None

    def _exponential_forecast(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        threshold: float,
        direction: str,
        biomarker: str
    ) -> Optional[ForecastResult]:
        """Exponential extrapolation to threshold."""
        try:
            log_values = np.log(values + 0.01)
            slope, intercept = np.polyfit(timestamps, log_values, 1)

            predicted_log = intercept + slope * timestamps
            ss_res = np.sum((log_values - predicted_log) ** 2)
            ss_tot = np.sum((log_values - np.mean(log_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            if slope == 0:
                return None

            current_time = timestamps[-1]
            log_threshold = np.log(threshold + 0.01)
            time_to_threshold = (log_threshold - intercept) / slope - current_time

            if direction == 'above' and slope <= 0:
                return None
            if direction == 'below' and slope >= 0:
                return None
            if time_to_threshold < 0:
                return None

            return ForecastResult(
                biomarker=biomarker,
                current_value=float(values[-1]),
                threshold=threshold,
                predicted_crossing_day=float(time_to_threshold),
                prediction_confidence=float(min(r_squared * 0.75, 0.90)),
                trajectory_model='exponential'
            )
        except:
            return None


class EarlyWarningEngine:
    """
    Main engine that combines all trajectory analysis components.

    This is the KEY to getting WEEKS of warning instead of DAYS.
    """

    def __init__(self):
        self.rate_analyzer = RateOfChangeAnalyzer()
        self.inflection_detector = InflectionDetector()
        self.pattern_library = PatternLibrary()
        self.forecaster = TrajectoryForecaster()

    def analyze_patient(
        self,
        patient_id: str,
        patient_data: Dict[str, List[float]],
        timestamps: List[float]
    ) -> EarlyWarningReport:
        """
        Complete trajectory analysis for a patient.

        Returns early warning report with estimated days to event.
        """
        # Step 1: Rate of change analysis
        rate_changes = self.rate_analyzer.analyze_patient_trajectory(
            patient_data, timestamps
        )

        # Step 2: Inflection point detection
        inflection_points = self.inflection_detector.detect_inflection_points(
            patient_data, timestamps
        )

        # Step 3: Pattern matching
        matched_patterns = self.pattern_library.match_patterns(
            patient_data, inflection_points, rate_changes
        )

        # Step 4: Forecasting
        forecasts = {}
        for biomarker, values in patient_data.items():
            if biomarker.lower() in ['patient_id', 'id', 'day', 'time', 'outcome', 'label']:
                continue
            forecast = self.forecaster.forecast_threshold_crossing(
                biomarker, values, timestamps
            )
            if forecast:
                forecasts[biomarker] = forecast

        # Step 5: Find earliest signal
        earliest_biomarker, earliest_days = self._find_earliest_signal(inflection_points)

        # Step 6: Calculate risk
        risk_level, confidence = self._calculate_overall_risk(
            rate_changes, matched_patterns, forecasts
        )

        # Step 7: Estimate timing
        days_to_event, estimation_range = self._estimate_event_timing(matched_patterns, forecasts)

        # Step 8: Detection improvement
        detection_improvement = earliest_days  # Days of early warning vs threshold

        # Step 9: Recommendations
        clinical_recs, monitoring_recs, genetic_recs = self._generate_recommendations(
            rate_changes, matched_patterns, risk_level
        )

        # Step 10: Signal propagation
        propagation_order = self._determine_signal_propagation(inflection_points)

        # Convert to serializable format
        return EarlyWarningReport(
            patient_id=patient_id,
            analysis_timestamp=datetime.now().isoformat(),
            risk_level=risk_level,
            confidence=confidence,
            estimated_days_to_event=days_to_event,
            estimation_range=estimation_range,
            detection_improvement_days=detection_improvement,
            matched_patterns=[{
                'pattern': p.pattern.value,
                'name': p.pattern_name,
                'confidence': p.confidence,
                'matched_features': p.matched_features,
                'estimated_days': p.estimated_days_to_event,
                'recommendations': p.recommended_actions
            } for p in matched_patterns],
            primary_pattern=matched_patterns[0].pattern_name if matched_patterns else None,
            rate_changes={k: {
                'biomarker': v.biomarker,
                'current_rate': v.current_rate,
                'z_score': v.z_score,
                'alert_level': v.alert_level,
                'days_of_trend': v.days_of_trend,
                'confidence': v.confidence
            } for k, v in rate_changes.items()},
            inflection_points={k: [{
                'day_index': p.day_index,
                'days_ago': p.days_ago,
                'value_at_inflection': p.value_at_inflection,
                'current_value': p.current_value,
                'change_since': p.change_since_inflection,
                'type': p.inflection_type,
                'significance': p.significance
            } for p in points] for k, points in inflection_points.items()},
            forecasts={k: {
                'biomarker': v.biomarker,
                'current_value': v.current_value,
                'threshold': v.threshold,
                'predicted_crossing_day': v.predicted_crossing_day,
                'confidence': v.prediction_confidence,
                'model': v.trajectory_model
            } for k, v in forecasts.items()},
            earliest_signal_biomarker=earliest_biomarker,
            earliest_signal_days_ago=earliest_days,
            signal_propagation_order=propagation_order,
            clinical_recommendations=clinical_recs,
            monitoring_recommendations=monitoring_recs,
            genetic_recommendations=genetic_recs
        )

    def _find_earliest_signal(self, inflection_points: Dict[str, List]) -> Tuple[str, float]:
        """Find which biomarker showed the earliest warning."""
        earliest = ('unknown', 0.0)

        for biomarker, points in inflection_points.items():
            if points:
                max_days = max(p.days_ago for p in points)
                if max_days > earliest[1]:
                    earliest = (biomarker, max_days)

        return earliest

    def _calculate_overall_risk(
        self,
        rate_changes: Dict[str, RateOfChangeResult],
        patterns: List[PatternMatch],
        forecasts: Dict[str, ForecastResult]
    ) -> Tuple[str, float]:
        """Calculate overall risk level."""
        scores = []

        for biomarker, result in rate_changes.items():
            if result.alert_level == 'critical':
                scores.append(0.9)
            elif result.alert_level == 'warning':
                scores.append(0.7)
            elif result.alert_level == 'elevated':
                scores.append(0.5)

        for pattern in patterns:
            scores.append(pattern.confidence)

        for biomarker, forecast in forecasts.items():
            if forecast.predicted_crossing_day > 0:
                risk_from_timing = max(0, 1 - (forecast.predicted_crossing_day / 14))
                scores.append(risk_from_timing * forecast.prediction_confidence)

        if not scores:
            return 'low', 0.3

        avg_score = float(np.mean(scores))

        if avg_score >= 0.75:
            return 'critical', avg_score
        elif avg_score >= 0.55:
            return 'high', avg_score
        elif avg_score >= 0.35:
            return 'moderate', avg_score
        else:
            return 'low', avg_score

    def _estimate_event_timing(
        self,
        patterns: List[PatternMatch],
        forecasts: Dict[str, ForecastResult]
    ) -> Tuple[float, Tuple[float, float]]:
        """Estimate days to event."""
        estimates = []

        for pattern in patterns:
            if pattern.estimated_days_to_event:
                estimates.append(pattern.estimated_days_to_event * pattern.confidence)

        for biomarker, forecast in forecasts.items():
            if forecast.predicted_crossing_day > 0:
                estimates.append(forecast.predicted_crossing_day)

        if not estimates:
            return 14.0, (7.0, 21.0)

        mean_estimate = float(np.mean(estimates))
        min_estimate = max(float(np.min(estimates)), 0.5)
        max_estimate = float(np.max(estimates))

        return mean_estimate, (min_estimate, max_estimate)

    def _generate_recommendations(
        self,
        rate_changes: Dict[str, RateOfChangeResult],
        patterns: List[PatternMatch],
        risk_level: str
    ) -> Tuple[List[str], Dict[str, str], List[str]]:
        """Generate recommendations."""
        clinical = []
        monitoring = {}
        genetic = []

        if risk_level == 'critical':
            clinical.append("IMMEDIATE: Alert care team - critical trajectory detected")
            clinical.append("Consider ICU bed availability")
        elif risk_level == 'high':
            clinical.append("URGENT: Schedule same-day clinical review")
            clinical.append("Prepare escalation protocols")
        elif risk_level == 'moderate':
            clinical.append("Increase monitoring frequency")
            clinical.append("Schedule clinical review within 24-48 hours")

        for biomarker, result in rate_changes.items():
            if result.alert_level in ['warning', 'critical']:
                if biomarker.lower() in ['procalcitonin', 'lactate', 'crp', 'wbc']:
                    clinical.append("Infection markers trending: consider cultures, empiric antibiotics")
                    genetic.append("Pharmacogenomics: CYP2D6, DPYD for antibiotic dosing")
                elif biomarker.lower() in ['troponin', 'bnp']:
                    clinical.append("Cardiac markers trending: order ECG, echo")
                    genetic.append("Pharmacogenomics: CYP2C19, VKORC1 for anticoagulation")
                elif biomarker.lower() in ['creatinine', 'egfr']:
                    clinical.append("Renal markers trending: assess volume, hold nephrotoxins")
                    genetic.append("Pharmacogenomics: SLCO1B1, ABCB1 for drug clearance")

                if result.alert_level == 'critical':
                    monitoring[biomarker] = "every 4 hours"
                else:
                    monitoring[biomarker] = "every 8 hours"

        for pattern in patterns[:2]:
            clinical.extend(pattern.recommended_actions[:2])

        return list(set(clinical))[:8], monitoring, list(set(genetic))[:4]

    def _determine_signal_propagation(self, inflection_points: Dict[str, List]) -> List[str]:
        """Determine order signals propagated."""
        biomarker_times = []
        for biomarker, points in inflection_points.items():
            if points:
                earliest = max(p.days_ago for p in points)
                biomarker_times.append((biomarker, earliest))

        sorted_biomarkers = sorted(biomarker_times, key=lambda x: x[1], reverse=True)
        return [b[0] for b in sorted_biomarkers]
