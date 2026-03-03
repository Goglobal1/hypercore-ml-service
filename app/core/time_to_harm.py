"""
Time-to-Harm Prediction Engine

Uses trajectory analysis to estimate time until clinical event.
This is the Synthetic Intelligence (SI) component that answers:
"If current trends continue, when will this patient reach critical state?"
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import math


class HarmType(str, Enum):
    """Types of clinical harm we predict."""
    SEPSIS_ONSET = "sepsis_onset"
    CARDIAC_EVENT = "cardiac_event"
    RESPIRATORY_FAILURE = "respiratory_failure"
    ACUTE_KIDNEY_INJURY = "acute_kidney_injury"
    HEPATIC_FAILURE = "hepatic_failure"
    NEUROLOGICAL_DECLINE = "neurological_decline"
    MULTI_ORGAN_FAILURE = "multi_organ_failure"
    ICU_TRANSFER = "icu_transfer"
    MORTALITY = "mortality"
    GENERIC_DETERIORATION = "generic_deterioration"


@dataclass
class TimeToHarmPrediction:
    """Prediction result for time-to-harm."""
    harm_type: HarmType
    hours_to_harm: float
    confidence: float
    trajectory_velocity: float  # Rate of change per hour
    critical_threshold: float   # Value at which harm occurs
    current_value: float
    projected_value_24h: float
    projected_value_48h: float
    key_drivers: List[str]
    intervention_window: str    # "immediate", "urgent", "monitor", "stable"
    intervention_window_hours: float
    rationale: str
    recommendations: List[str]


# Domain-specific critical thresholds (from clinical literature + MIMIC patterns)
CRITICAL_THRESHOLDS = {
    "sepsis": {
        "lactate": {"critical": 4.0, "warning": 2.0, "unit": "mmol/L", "direction": "rising", "weight": 1.0},
        "map": {"critical": 65.0, "warning": 70.0, "unit": "mmHg", "direction": "falling", "weight": 0.9},
        "crp": {"critical": 100.0, "warning": 50.0, "unit": "mg/L", "direction": "rising", "weight": 0.7},
        "wbc": {"critical": 12.0, "warning": 10.0, "unit": "K/uL", "direction": "rising", "weight": 0.6},
        "temperature": {"critical": 38.3, "warning": 37.8, "unit": "C", "direction": "rising", "weight": 0.5},
        "procalcitonin": {"critical": 2.0, "warning": 0.5, "unit": "ng/mL", "direction": "rising", "weight": 0.8},
        "heart_rate": {"critical": 120.0, "warning": 100.0, "unit": "bpm", "direction": "rising", "weight": 0.5},
        "respiratory_rate": {"critical": 24.0, "warning": 20.0, "unit": "/min", "direction": "rising", "weight": 0.5}
    },
    "cardiac": {
        "troponin": {"critical": 0.04, "warning": 0.01, "unit": "ng/mL", "direction": "rising", "weight": 1.0},
        "troponin_i": {"critical": 0.04, "warning": 0.01, "unit": "ng/mL", "direction": "rising", "weight": 1.0},
        "troponin_t": {"critical": 0.1, "warning": 0.03, "unit": "ng/mL", "direction": "rising", "weight": 1.0},
        "bnp": {"critical": 400.0, "warning": 100.0, "unit": "pg/mL", "direction": "rising", "weight": 0.9},
        "nt_probnp": {"critical": 900.0, "warning": 300.0, "unit": "pg/mL", "direction": "rising", "weight": 0.9},
        "heart_rate": {"critical": 130.0, "warning": 110.0, "unit": "bpm", "direction": "rising", "weight": 0.7},
        "systolic_bp": {"critical": 90.0, "warning": 100.0, "unit": "mmHg", "direction": "falling", "weight": 0.8},
        "diastolic_bp": {"critical": 60.0, "warning": 65.0, "unit": "mmHg", "direction": "falling", "weight": 0.6},
        "ck_mb": {"critical": 25.0, "warning": 5.0, "unit": "ng/mL", "direction": "rising", "weight": 0.7}
    },
    "kidney": {
        "creatinine": {"critical": 2.0, "warning": 1.3, "unit": "mg/dL", "direction": "rising", "weight": 1.0},
        "bun": {"critical": 40.0, "warning": 25.0, "unit": "mg/dL", "direction": "rising", "weight": 0.8},
        "potassium": {"critical": 5.5, "warning": 5.0, "unit": "mEq/L", "direction": "rising", "weight": 0.9},
        "gfr": {"critical": 30.0, "warning": 60.0, "unit": "mL/min", "direction": "falling", "weight": 0.9},
        "urine_output": {"critical": 0.5, "warning": 1.0, "unit": "mL/kg/h", "direction": "falling", "weight": 0.8},
        "sodium": {"critical": 125.0, "warning": 130.0, "unit": "mEq/L", "direction": "falling", "weight": 0.5}
    },
    "respiratory": {
        "spo2": {"critical": 90.0, "warning": 94.0, "unit": "%", "direction": "falling", "weight": 1.0},
        "pao2": {"critical": 60.0, "warning": 80.0, "unit": "mmHg", "direction": "falling", "weight": 0.9},
        "respiratory_rate": {"critical": 30.0, "warning": 24.0, "unit": "/min", "direction": "rising", "weight": 0.8},
        "fio2": {"critical": 0.6, "warning": 0.4, "unit": "fraction", "direction": "rising", "weight": 0.7},
        "pao2_fio2": {"critical": 200.0, "warning": 300.0, "unit": "ratio", "direction": "falling", "weight": 0.9},
        "pco2": {"critical": 50.0, "warning": 45.0, "unit": "mmHg", "direction": "rising", "weight": 0.6}
    },
    "hepatic": {
        "alt": {"critical": 1000.0, "warning": 200.0, "unit": "U/L", "direction": "rising", "weight": 0.8},
        "ast": {"critical": 1000.0, "warning": 200.0, "unit": "U/L", "direction": "rising", "weight": 0.8},
        "bilirubin": {"critical": 4.0, "warning": 2.0, "unit": "mg/dL", "direction": "rising", "weight": 0.9},
        "inr": {"critical": 2.0, "warning": 1.5, "unit": "ratio", "direction": "rising", "weight": 0.9},
        "albumin": {"critical": 2.5, "warning": 3.0, "unit": "g/dL", "direction": "falling", "weight": 0.7},
        "ammonia": {"critical": 100.0, "warning": 50.0, "unit": "umol/L", "direction": "rising", "weight": 0.8}
    },
    "neurological": {
        "gcs": {"critical": 8.0, "warning": 12.0, "unit": "score", "direction": "falling", "weight": 1.0},
        "icp": {"critical": 22.0, "warning": 15.0, "unit": "mmHg", "direction": "rising", "weight": 0.9},
        "cpp": {"critical": 60.0, "warning": 70.0, "unit": "mmHg", "direction": "falling", "weight": 0.9}
    },
    "hematologic": {
        "hemoglobin": {"critical": 7.0, "warning": 9.0, "unit": "g/dL", "direction": "falling", "weight": 0.9},
        "platelets": {"critical": 50.0, "warning": 100.0, "unit": "K/uL", "direction": "falling", "weight": 0.8},
        "inr": {"critical": 2.5, "warning": 1.5, "unit": "ratio", "direction": "rising", "weight": 0.8},
        "fibrinogen": {"critical": 100.0, "warning": 200.0, "unit": "mg/dL", "direction": "falling", "weight": 0.7}
    }
}

# Recommendations based on domain and urgency
RECOMMENDATIONS = {
    "sepsis": {
        "immediate": [
            "Initiate Sepsis Bundle within 1 hour",
            "Obtain blood cultures before antibiotics",
            "Administer broad-spectrum antibiotics",
            "Begin fluid resuscitation (30 mL/kg crystalloid)",
            "Consider vasopressors if MAP < 65 after fluids"
        ],
        "urgent": [
            "Repeat lactate measurement in 2-4 hours",
            "Monitor urine output hourly",
            "Reassess fluid responsiveness",
            "Consider infectious disease consult"
        ],
        "monitor": [
            "Continue trending inflammatory markers",
            "Reassess in 4-6 hours",
            "Monitor for clinical deterioration"
        ]
    },
    "cardiac": {
        "immediate": [
            "Obtain 12-lead ECG stat",
            "Activate cardiac catheterization lab if STEMI",
            "Administer aspirin and anticoagulation",
            "Cardiology consult stat"
        ],
        "urgent": [
            "Serial troponins every 3-6 hours",
            "Continuous cardiac monitoring",
            "Echocardiogram within 24 hours",
            "Risk stratify with HEART or TIMI score"
        ],
        "monitor": [
            "Continue cardiac monitoring",
            "Trend cardiac biomarkers",
            "Optimize rate and rhythm control"
        ]
    },
    "kidney": {
        "immediate": [
            "Evaluate for obstruction (bladder scan, renal ultrasound)",
            "Hold nephrotoxic medications",
            "Nephrology consult for dialysis evaluation",
            "Assess volume status and optimize perfusion"
        ],
        "urgent": [
            "Strict I/O monitoring",
            "Avoid contrast if possible",
            "Renally dose all medications",
            "Trend creatinine every 6-12 hours"
        ],
        "monitor": [
            "Daily creatinine and electrolytes",
            "Maintain euvolemia",
            "Review medication list for nephrotoxins"
        ]
    },
    "respiratory": {
        "immediate": [
            "Prepare for intubation",
            "Apply high-flow oxygen or NIV",
            "Obtain ABG stat",
            "Chest X-ray and consider CT if PE suspected"
        ],
        "urgent": [
            "Increase oxygen supplementation",
            "Respiratory therapy evaluation",
            "Consider bronchodilators if wheezing",
            "Monitor closely for fatigue"
        ],
        "monitor": [
            "Continue SpO2 monitoring",
            "Incentive spirometry",
            "Mobilize if appropriate"
        ]
    }
}


class TimeToHarmEngine:
    """
    Predicts time until clinical harm based on trajectory analysis.

    Uses linear extrapolation with confidence intervals based on:
    - Trajectory consistency (R^2 of trend)
    - Historical patterns from reference data
    - Domain-specific velocity thresholds
    - Multi-marker convergence
    """

    def __init__(self, reference_data: Optional[Dict] = None):
        """
        Args:
            reference_data: Optional pre-loaded MIMIC reference trajectories
        """
        self.reference_data = reference_data or {}
        self.thresholds = CRITICAL_THRESHOLDS
        self.recommendations = RECOMMENDATIONS

    def calculate_velocity(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Tuple[float, float, float]:
        """
        Calculate rate of change per hour using linear regression.

        Returns:
            (velocity, r_squared, std_error) - velocity in units/hour, fit quality, standard error
        """
        if len(values) < 2:
            return 0.0, 0.0, 0.0

        # Convert to hours from first timestamp
        hours = []
        base_ts = timestamps[0]
        if isinstance(base_ts, str):
            base_ts = datetime.fromisoformat(base_ts.replace('Z', '+00:00'))

        for t in timestamps:
            if isinstance(t, str):
                t = datetime.fromisoformat(t.replace('Z', '+00:00'))
            delta = (t - base_ts).total_seconds() / 3600.0
            hours.append(delta)

        # Linear regression
        n = len(values)
        sum_x = sum(hours)
        sum_y = sum(values)
        sum_xy = sum(h * v for h, v in zip(hours, values))
        sum_x2 = sum(h * h for h in hours)

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0, 0.0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # R-squared for confidence
        y_mean = sum_y / n
        ss_tot = sum((v - y_mean) ** 2 for v in values)

        predictions = [intercept + slope * h for h in hours]
        ss_res = sum((v - p) ** 2 for v, p in zip(values, predictions))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        # Standard error
        if n > 2 and ss_tot > 0:
            std_error = math.sqrt(ss_res / (n - 2)) if ss_res > 0 else 0.0
        else:
            std_error = 0.0

        return slope, r_squared, std_error

    def predict_time_to_threshold(
        self,
        current_value: float,
        velocity: float,
        threshold: float,
        direction: str
    ) -> Optional[float]:
        """
        Predict hours until value reaches threshold.

        Returns:
            Hours until threshold, or None if not trending toward threshold
        """
        if direction == "rising":
            if current_value >= threshold:
                return 0.0  # Already critical
            if velocity <= 0.001:  # Not rising meaningfully
                return None
            return (threshold - current_value) / velocity
        else:  # falling
            if current_value <= threshold:
                return 0.0  # Already critical
            if velocity >= -0.001:  # Not falling meaningfully
                return None
            return (current_value - threshold) / abs(velocity)

    def _normalize_biomarker_name(self, name: str) -> str:
        """Normalize biomarker names for matching."""
        return name.lower().replace('-', '_').replace(' ', '_')

    def _get_domain_thresholds(self, domain: str) -> Dict:
        """Get thresholds for a domain, with fallback."""
        domain_lower = domain.lower().replace('deterioration_', '').replace('_injury', '')

        # Direct match
        if domain_lower in self.thresholds:
            return self.thresholds[domain_lower]

        # Partial match
        for key in self.thresholds:
            if key in domain_lower or domain_lower in key:
                return self.thresholds[key]

        # Default to sepsis (most comprehensive)
        return self.thresholds.get("sepsis", {})

    def _determine_intervention_window(self, hours: float) -> Tuple[str, float]:
        """Determine intervention window based on time to harm."""
        if hours <= 0:
            return "immediate", 0.0
        elif hours <= 6:
            return "immediate", hours
        elif hours <= 24:
            return "urgent", hours
        elif hours <= 72:
            return "monitor", hours
        else:
            return "stable", hours

    def _get_recommendations(self, domain: str, window: str) -> List[str]:
        """Get recommendations based on domain and urgency."""
        domain_lower = domain.lower().replace('deterioration_', '').replace('_injury', '')

        domain_recs = self.recommendations.get(domain_lower, {})
        if domain_recs:
            return domain_recs.get(window, domain_recs.get("monitor", []))

        # Generic recommendations
        if window == "immediate":
            return ["Immediate physician evaluation", "Prepare for escalation of care"]
        elif window == "urgent":
            return ["Close monitoring", "Notify care team", "Reassess in 2-4 hours"]
        else:
            return ["Continue current monitoring", "Reassess per protocol"]

    def _domain_to_harm_type(self, domain: str) -> HarmType:
        """Map domain to harm type."""
        domain_lower = domain.lower()

        mapping = {
            "sepsis": HarmType.SEPSIS_ONSET,
            "cardiac": HarmType.CARDIAC_EVENT,
            "deterioration_cardiac": HarmType.CARDIAC_EVENT,
            "kidney": HarmType.ACUTE_KIDNEY_INJURY,
            "kidney_injury": HarmType.ACUTE_KIDNEY_INJURY,
            "respiratory": HarmType.RESPIRATORY_FAILURE,
            "respiratory_failure": HarmType.RESPIRATORY_FAILURE,
            "hepatic": HarmType.HEPATIC_FAILURE,
            "hepatic_dysfunction": HarmType.HEPATIC_FAILURE,
            "neurological": HarmType.NEUROLOGICAL_DECLINE,
            "hematologic": HarmType.GENERIC_DETERIORATION,
            "multi_system": HarmType.MULTI_ORGAN_FAILURE
        }

        for key, harm_type in mapping.items():
            if key in domain_lower:
                return harm_type

        return HarmType.GENERIC_DETERIORATION

    def _generate_rationale(
        self,
        domain: str,
        most_urgent: Dict,
        all_predictions: List[Dict],
        window: str
    ) -> str:
        """Generate clinical rationale text."""
        marker = most_urgent["marker"]
        hours = most_urgent["hours_to_harm"]
        current = most_urgent["current_value"]
        threshold = most_urgent["threshold"]
        velocity = most_urgent["velocity"]
        direction = "rising toward" if most_urgent["direction"] == "rising" else "falling toward"

        # Format velocity for display
        velocity_str = f"{abs(velocity):.2f}/hour"

        base = f"{marker.upper()} is {direction} critical threshold"
        current_vs_threshold = f"(current: {current:.2f}, critical: {threshold:.2f}, velocity: {velocity_str})"

        if hours <= 0:
            urgency = "CRITICAL THRESHOLD REACHED. Immediate intervention required."
        elif hours < 6:
            urgency = f"Projected to reach critical level in {hours:.1f} hours. Immediate evaluation recommended."
        elif hours < 24:
            urgency = f"Projected to reach critical level in {hours:.1f} hours. Urgent evaluation needed."
        elif hours < 72:
            urgency = f"May reach critical level in {hours:.1f} hours if trend continues. Close monitoring advised."
        else:
            urgency = f"Stable trajectory. Continue routine monitoring."

        # Add supporting markers
        if len(all_predictions) > 1:
            other_markers = [p["marker"] for p in all_predictions if p["marker"] != marker][:3]
            if other_markers:
                supporting = f" Additional concerning trends: {', '.join(other_markers)}."
            else:
                supporting = ""
        else:
            supporting = ""

        return f"{base} {current_vs_threshold}. {urgency}{supporting}"

    def predict(
        self,
        patient_id: str,
        domain: str,
        biomarker_trajectories: Dict[str, List[Dict]],
        current_timestamp: Optional[datetime] = None
    ) -> TimeToHarmPrediction:
        """
        Main prediction entry point.

        Args:
            patient_id: Patient identifier
            domain: Clinical domain (sepsis, cardiac, kidney, respiratory, etc.)
            biomarker_trajectories: Dict of biomarker -> [{"timestamp": dt, "value": float}, ...]
            current_timestamp: Reference time (defaults to now)

        Returns:
            TimeToHarmPrediction with estimated time to clinical harm
        """
        if current_timestamp is None:
            current_timestamp = datetime.now(timezone.utc)
        elif isinstance(current_timestamp, str):
            current_timestamp = datetime.fromisoformat(current_timestamp.replace('Z', '+00:00'))

        domain_thresholds = self._get_domain_thresholds(domain)

        predictions_by_marker = []
        key_drivers = []

        for marker, trajectory in biomarker_trajectories.items():
            normalized_marker = self._normalize_biomarker_name(marker)

            # Find matching threshold config
            thresh_config = None
            for thresh_name, config in domain_thresholds.items():
                if self._normalize_biomarker_name(thresh_name) == normalized_marker:
                    thresh_config = config
                    break

            if thresh_config is None:
                continue

            if len(trajectory) < 2:
                continue

            # Extract values and timestamps
            values = [p["value"] for p in trajectory]
            timestamps = []
            for p in trajectory:
                ts = p["timestamp"]
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                timestamps.append(ts)

            # Calculate velocity
            velocity, r_squared, std_error = self.calculate_velocity(values, timestamps)
            current_value = values[-1]

            # Predict time to critical threshold
            hours_to_critical = self.predict_time_to_threshold(
                current_value=current_value,
                velocity=velocity,
                threshold=thresh_config["critical"],
                direction=thresh_config["direction"]
            )

            # Predict time to warning threshold
            hours_to_warning = self.predict_time_to_threshold(
                current_value=current_value,
                velocity=velocity,
                threshold=thresh_config["warning"],
                direction=thresh_config["direction"]
            )

            # Use the more urgent (critical if available, else warning)
            hours_to_harm = hours_to_critical
            threshold_used = thresh_config["critical"]

            if hours_to_harm is not None and hours_to_harm < 168:  # Within 7-day window
                weight = thresh_config.get("weight", 1.0)
                confidence = r_squared * weight

                predictions_by_marker.append({
                    "marker": marker,
                    "hours_to_harm": hours_to_harm,
                    "hours_to_warning": hours_to_warning,
                    "velocity": velocity,
                    "confidence": confidence,
                    "r_squared": r_squared,
                    "current_value": current_value,
                    "threshold": threshold_used,
                    "warning_threshold": thresh_config["warning"],
                    "direction": thresh_config["direction"],
                    "unit": thresh_config["unit"],
                    "weight": weight
                })

                # Build driver description
                direction_word = "^" if thresh_config["direction"] == "rising" else "v"
                driver_str = f"{marker} {direction_word} ({current_value:.1f} -> {threshold_used} in ~{hours_to_harm:.0f}h)"
                key_drivers.append(driver_str)

        # Handle case with no concerning trajectories
        if not predictions_by_marker:
            return TimeToHarmPrediction(
                harm_type=self._domain_to_harm_type(domain),
                hours_to_harm=999.0,
                confidence=0.0,
                trajectory_velocity=0.0,
                critical_threshold=0.0,
                current_value=0.0,
                projected_value_24h=0.0,
                projected_value_48h=0.0,
                key_drivers=[],
                intervention_window="stable",
                intervention_window_hours=999.0,
                rationale="No concerning trajectory detected in current data. Continue routine monitoring.",
                recommendations=["Continue current care plan", "Routine monitoring per protocol"]
            )

        # Find most urgent marker (weighted by confidence and time)
        def urgency_score(p):
            # Lower score = more urgent
            time_factor = p["hours_to_harm"] if p["hours_to_harm"] > 0 else 0.1
            confidence_factor = 1.0 / (p["confidence"] + 0.1)
            return time_factor * confidence_factor

        predictions_by_marker.sort(key=urgency_score)
        most_urgent = predictions_by_marker[0]

        # Calculate weighted average confidence
        total_weight = sum(p["confidence"] for p in predictions_by_marker)
        if total_weight > 0:
            avg_confidence = total_weight / len(predictions_by_marker)
        else:
            avg_confidence = 0.0

        # Boost confidence if multiple markers converging
        if len(predictions_by_marker) >= 3:
            avg_confidence = min(0.95, avg_confidence * 1.2)
        elif len(predictions_by_marker) >= 2:
            avg_confidence = min(0.90, avg_confidence * 1.1)

        # Project values
        projected_24h = most_urgent["current_value"] + (most_urgent["velocity"] * 24)
        projected_48h = most_urgent["current_value"] + (most_urgent["velocity"] * 48)

        # Determine intervention window
        window, window_hours = self._determine_intervention_window(most_urgent["hours_to_harm"])

        # Get recommendations
        recommendations = self._get_recommendations(domain, window)

        # Generate rationale
        rationale = self._generate_rationale(
            domain=domain,
            most_urgent=most_urgent,
            all_predictions=predictions_by_marker,
            window=window
        )

        return TimeToHarmPrediction(
            harm_type=self._domain_to_harm_type(domain),
            hours_to_harm=round(most_urgent["hours_to_harm"], 1),
            confidence=round(avg_confidence, 2),
            trajectory_velocity=round(most_urgent["velocity"], 4),
            critical_threshold=most_urgent["threshold"],
            current_value=most_urgent["current_value"],
            projected_value_24h=round(projected_24h, 2),
            projected_value_48h=round(projected_48h, 2),
            key_drivers=key_drivers[:5],  # Top 5 drivers
            intervention_window=window,
            intervention_window_hours=round(window_hours, 1),
            rationale=rationale,
            recommendations=recommendations[:5]  # Top 5 recommendations
        )


# === API Helper Functions ===

def predict_time_to_harm(
    patient_id: str,
    domain: str,
    biomarker_trajectories: Dict[str, List[Dict]],
    current_timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    API-friendly function to predict time to harm.

    Args:
        patient_id: Patient identifier
        domain: Clinical domain
        biomarker_trajectories: {
            "lactate": [{"timestamp": "2026-01-07T10:00:00Z", "value": 2.1}, ...],
            "crp": [{"timestamp": "2026-01-07T10:00:00Z", "value": 45.0}, ...]
        }
        current_timestamp: Optional ISO8601 timestamp

    Returns:
        Dict with prediction results
    """
    engine = TimeToHarmEngine()

    ts = None
    if current_timestamp:
        ts = datetime.fromisoformat(current_timestamp.replace('Z', '+00:00'))

    prediction = engine.predict(
        patient_id=patient_id,
        domain=domain,
        biomarker_trajectories=biomarker_trajectories,
        current_timestamp=ts
    )

    return {
        "patient_id": patient_id,
        "domain": domain,
        "harm_type": prediction.harm_type.value,
        "hours_to_harm": prediction.hours_to_harm,
        "confidence": prediction.confidence,
        "trajectory_velocity": prediction.trajectory_velocity,
        "critical_threshold": prediction.critical_threshold,
        "current_value": prediction.current_value,
        "projected_value_24h": prediction.projected_value_24h,
        "projected_value_48h": prediction.projected_value_48h,
        "key_drivers": prediction.key_drivers,
        "intervention_window": prediction.intervention_window,
        "intervention_window_hours": prediction.intervention_window_hours,
        "rationale": prediction.rationale,
        "recommendations": prediction.recommendations
    }


def get_supported_domains() -> List[str]:
    """Return list of supported clinical domains."""
    return list(CRITICAL_THRESHOLDS.keys())


def get_domain_biomarkers(domain: str) -> List[str]:
    """Return list of biomarkers tracked for a domain."""
    engine = TimeToHarmEngine()
    thresholds = engine._get_domain_thresholds(domain)
    return list(thresholds.keys())
