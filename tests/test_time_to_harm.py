"""Tests for Time-to-Harm Prediction Engine."""

import pytest
from datetime import datetime, timedelta, timezone
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.time_to_harm import (
    TimeToHarmEngine,
    predict_time_to_harm,
    get_supported_domains,
    get_domain_biomarkers,
    HarmType
)


class TestVelocityCalculation:
    """Test velocity calculation."""

    def test_rising_velocity(self):
        engine = TimeToHarmEngine()
        now = datetime.now(timezone.utc)

        values = [1.0, 2.0, 3.0]
        timestamps = [
            now - timedelta(hours=2),
            now - timedelta(hours=1),
            now
        ]

        velocity, r_squared, _ = engine.calculate_velocity(values, timestamps)

        assert velocity > 0  # Rising
        assert abs(velocity - 1.0) < 0.01  # ~1.0/hour
        assert r_squared > 0.99  # Perfect linear fit

    def test_falling_velocity(self):
        engine = TimeToHarmEngine()
        now = datetime.now(timezone.utc)

        values = [100.0, 90.0, 80.0]
        timestamps = [
            now - timedelta(hours=2),
            now - timedelta(hours=1),
            now
        ]

        velocity, r_squared, _ = engine.calculate_velocity(values, timestamps)

        assert velocity < 0  # Falling
        assert abs(velocity - (-10.0)) < 0.01  # ~-10.0/hour

    def test_stable_velocity(self):
        engine = TimeToHarmEngine()
        now = datetime.now(timezone.utc)

        values = [5.0, 5.1, 4.9, 5.0]
        timestamps = [
            now - timedelta(hours=3),
            now - timedelta(hours=2),
            now - timedelta(hours=1),
            now
        ]

        velocity, _, _ = engine.calculate_velocity(values, timestamps)

        assert abs(velocity) < 0.1  # Nearly stable


class TestTimeToThreshold:
    """Test time-to-threshold prediction."""

    def test_rising_to_threshold(self):
        engine = TimeToHarmEngine()

        # Current: 2.0, rising 0.5/hour, threshold: 4.0
        # Should reach threshold in 4 hours
        hours = engine.predict_time_to_threshold(
            current_value=2.0,
            velocity=0.5,
            threshold=4.0,
            direction="rising"
        )

        assert hours is not None
        assert abs(hours - 4.0) < 0.01

    def test_already_critical(self):
        engine = TimeToHarmEngine()

        hours = engine.predict_time_to_threshold(
            current_value=5.0,  # Already above threshold
            velocity=0.5,
            threshold=4.0,
            direction="rising"
        )

        assert hours == 0.0  # Already critical

    def test_not_trending_toward_threshold(self):
        engine = TimeToHarmEngine()

        # Falling when should be rising
        hours = engine.predict_time_to_threshold(
            current_value=2.0,
            velocity=-0.5,  # Falling
            threshold=4.0,
            direction="rising"
        )

        assert hours is None  # Not trending toward threshold

    def test_falling_to_threshold(self):
        engine = TimeToHarmEngine()

        # Current: 100, falling 5/hour, threshold: 90
        # Should reach threshold in 2 hours
        hours = engine.predict_time_to_threshold(
            current_value=100.0,
            velocity=-5.0,
            threshold=90.0,
            direction="falling"
        )

        assert hours is not None
        assert abs(hours - 2.0) < 0.01


class TestSepsisTrajectory:
    """Test sepsis deterioration prediction."""

    def test_rising_lactate_prediction(self):
        now = datetime.now(timezone.utc)

        # Lactate rising from 2.0 to 3.0 over 6 hours
        # Velocity = 0.167/hour, threshold = 4.0
        # Time to harm = (4.0 - 3.0) / 0.167 ~ 6 hours
        trajectories = {
            "lactate": [
                {"timestamp": (now - timedelta(hours=6)).isoformat(), "value": 2.0},
                {"timestamp": (now - timedelta(hours=3)).isoformat(), "value": 2.5},
                {"timestamp": now.isoformat(), "value": 3.0}
            ]
        }

        result = predict_time_to_harm("P001", "sepsis", trajectories)

        assert result["hours_to_harm"] < 10  # Should be around 6 hours
        assert result["intervention_window"] in ["immediate", "urgent"]
        assert "lactate" in result["rationale"].lower()
        assert result["harm_type"] == "sepsis_onset"

    def test_multi_marker_sepsis(self):
        now = datetime.now(timezone.utc)

        trajectories = {
            "lactate": [
                {"timestamp": (now - timedelta(hours=6)).isoformat(), "value": 1.5},
                {"timestamp": (now - timedelta(hours=3)).isoformat(), "value": 2.5},
                {"timestamp": now.isoformat(), "value": 3.2}
            ],
            "crp": [
                {"timestamp": (now - timedelta(hours=6)).isoformat(), "value": 30.0},
                {"timestamp": (now - timedelta(hours=3)).isoformat(), "value": 55.0},
                {"timestamp": now.isoformat(), "value": 80.0}
            ],
            "wbc": [
                {"timestamp": (now - timedelta(hours=6)).isoformat(), "value": 8.0},
                {"timestamp": (now - timedelta(hours=3)).isoformat(), "value": 10.0},
                {"timestamp": now.isoformat(), "value": 11.5}
            ]
        }

        result = predict_time_to_harm("P002", "sepsis", trajectories)

        assert result["confidence"] > 0.3  # Multiple markers boost confidence
        assert len(result["key_drivers"]) >= 2  # Multiple drivers
        assert len(result["recommendations"]) > 0


class TestCardiacTrajectory:
    """Test cardiac deterioration prediction."""

    def test_rising_troponin(self):
        now = datetime.now(timezone.utc)

        trajectories = {
            "troponin": [
                {"timestamp": (now - timedelta(hours=6)).isoformat(), "value": 0.01},
                {"timestamp": (now - timedelta(hours=3)).isoformat(), "value": 0.02},
                {"timestamp": now.isoformat(), "value": 0.03}
            ]
        }

        result = predict_time_to_harm("P003", "cardiac", trajectories)

        assert result["harm_type"] == "cardiac_event"
        assert result["hours_to_harm"] < 24
        assert "troponin" in result["rationale"].lower()


class TestKidneyTrajectory:
    """Test kidney injury prediction."""

    def test_rising_creatinine(self):
        now = datetime.now(timezone.utc)

        trajectories = {
            "creatinine": [
                {"timestamp": (now - timedelta(hours=24)).isoformat(), "value": 1.0},
                {"timestamp": (now - timedelta(hours=12)).isoformat(), "value": 1.3},
                {"timestamp": now.isoformat(), "value": 1.6}
            ]
        }

        result = predict_time_to_harm("P004", "kidney", trajectories)

        assert result["harm_type"] == "acute_kidney_injury"
        assert result["hours_to_harm"] < 48


class TestStableTrajectory:
    """Test stable/non-concerning trajectories."""

    def test_stable_values_no_alarm(self):
        now = datetime.now(timezone.utc)

        trajectories = {
            "lactate": [
                {"timestamp": (now - timedelta(hours=6)).isoformat(), "value": 1.0},
                {"timestamp": (now - timedelta(hours=3)).isoformat(), "value": 1.1},
                {"timestamp": now.isoformat(), "value": 1.0}
            ]
        }

        result = predict_time_to_harm("P005", "sepsis", trajectories)

        assert result["intervention_window"] == "stable"
        assert result["hours_to_harm"] > 100

    def test_improving_values(self):
        now = datetime.now(timezone.utc)

        # Lactate decreasing (improving)
        trajectories = {
            "lactate": [
                {"timestamp": (now - timedelta(hours=6)).isoformat(), "value": 3.5},
                {"timestamp": (now - timedelta(hours=3)).isoformat(), "value": 2.5},
                {"timestamp": now.isoformat(), "value": 1.8}
            ]
        }

        result = predict_time_to_harm("P006", "sepsis", trajectories)

        # Should not predict imminent harm since lactate is falling
        assert result["intervention_window"] in ["stable", "monitor"]


class TestInterventionWindows:
    """Test intervention window classification."""

    def test_immediate_window(self):
        now = datetime.now(timezone.utc)

        # Already at critical threshold
        trajectories = {
            "lactate": [
                {"timestamp": (now - timedelta(hours=2)).isoformat(), "value": 3.5},
                {"timestamp": (now - timedelta(hours=1)).isoformat(), "value": 3.8},
                {"timestamp": now.isoformat(), "value": 4.2}  # Above 4.0 critical
            ]
        }

        result = predict_time_to_harm("P007", "sepsis", trajectories)

        assert result["intervention_window"] == "immediate"
        assert result["hours_to_harm"] <= 6


class TestAPIHelpers:
    """Test API helper functions."""

    def test_get_supported_domains(self):
        domains = get_supported_domains()

        assert "sepsis" in domains
        assert "cardiac" in domains
        assert "kidney" in domains
        assert "respiratory" in domains

    def test_get_domain_biomarkers(self):
        sepsis_markers = get_domain_biomarkers("sepsis")

        assert "lactate" in sepsis_markers
        assert "crp" in sepsis_markers
        assert "wbc" in sepsis_markers

        cardiac_markers = get_domain_biomarkers("cardiac")

        assert "troponin" in cardiac_markers
        assert "bnp" in cardiac_markers


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_trajectories(self):
        result = predict_time_to_harm("P008", "sepsis", {})

        assert result["intervention_window"] == "stable"
        assert result["confidence"] == 0.0

    def test_single_point_trajectory(self):
        now = datetime.now(timezone.utc)

        trajectories = {
            "lactate": [
                {"timestamp": now.isoformat(), "value": 2.5}
            ]
        }

        result = predict_time_to_harm("P009", "sepsis", trajectories)

        # Can't calculate velocity with single point
        assert result["intervention_window"] == "stable"

    def test_unknown_biomarker(self):
        now = datetime.now(timezone.utc)

        trajectories = {
            "unknown_marker": [
                {"timestamp": (now - timedelta(hours=2)).isoformat(), "value": 10.0},
                {"timestamp": now.isoformat(), "value": 20.0}
            ]
        }

        result = predict_time_to_harm("P010", "sepsis", trajectories)

        # Unknown marker should be ignored
        assert result["intervention_window"] == "stable"

    def test_unknown_domain_uses_default(self):
        now = datetime.now(timezone.utc)

        trajectories = {
            "lactate": [
                {"timestamp": (now - timedelta(hours=6)).isoformat(), "value": 2.0},
                {"timestamp": now.isoformat(), "value": 3.5}
            ]
        }

        # Unknown domain should fall back to sepsis thresholds
        result = predict_time_to_harm("P011", "unknown_domain", trajectories)

        assert result["hours_to_harm"] < 100  # Should still work


class TestRespiratoryTrajectory:
    """Test respiratory failure prediction."""

    def test_falling_spo2(self):
        now = datetime.now(timezone.utc)

        trajectories = {
            "spo2": [
                {"timestamp": (now - timedelta(hours=4)).isoformat(), "value": 98.0},
                {"timestamp": (now - timedelta(hours=2)).isoformat(), "value": 95.0},
                {"timestamp": now.isoformat(), "value": 92.0}
            ]
        }

        result = predict_time_to_harm("P012", "respiratory", trajectories)

        assert result["harm_type"] == "respiratory_failure"
        assert result["hours_to_harm"] < 24
        assert "spo2" in result["rationale"].lower()


class TestHepaticTrajectory:
    """Test hepatic failure prediction."""

    def test_rising_bilirubin(self):
        now = datetime.now(timezone.utc)

        trajectories = {
            "bilirubin": [
                {"timestamp": (now - timedelta(hours=12)).isoformat(), "value": 1.5},
                {"timestamp": (now - timedelta(hours=6)).isoformat(), "value": 2.5},
                {"timestamp": now.isoformat(), "value": 3.5}
            ]
        }

        result = predict_time_to_harm("P013", "hepatic", trajectories)

        assert result["harm_type"] == "hepatic_failure"
        assert result["hours_to_harm"] < 24


class TestProjectedValues:
    """Test projected value calculations."""

    def test_projected_values_increase(self):
        now = datetime.now(timezone.utc)

        # Rising at 0.5/hour
        trajectories = {
            "lactate": [
                {"timestamp": (now - timedelta(hours=4)).isoformat(), "value": 1.0},
                {"timestamp": (now - timedelta(hours=2)).isoformat(), "value": 2.0},
                {"timestamp": now.isoformat(), "value": 3.0}
            ]
        }

        result = predict_time_to_harm("P014", "sepsis", trajectories)

        # Velocity is ~0.5/hour
        # projected_24h should be ~3.0 + 0.5*24 = 15.0
        # projected_48h should be ~3.0 + 0.5*48 = 27.0
        assert result["projected_value_24h"] > result["current_value"]
        assert result["projected_value_48h"] > result["projected_value_24h"]


class TestRecommendations:
    """Test recommendations generation."""

    def test_sepsis_immediate_recommendations(self):
        now = datetime.now(timezone.utc)

        # Critical lactate level
        trajectories = {
            "lactate": [
                {"timestamp": (now - timedelta(hours=2)).isoformat(), "value": 3.5},
                {"timestamp": (now - timedelta(hours=1)).isoformat(), "value": 4.0},
                {"timestamp": now.isoformat(), "value": 4.5}
            ]
        }

        result = predict_time_to_harm("P015", "sepsis", trajectories)

        assert result["intervention_window"] == "immediate"
        assert len(result["recommendations"]) > 0
        # Should contain sepsis-specific recommendations
        recs_text = " ".join(result["recommendations"]).lower()
        assert "sepsis" in recs_text or "antibiotic" in recs_text or "fluid" in recs_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
