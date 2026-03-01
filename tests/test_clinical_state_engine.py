# test_clinical_state_engine.py
"""
Tests for Clinical State Engine (CSE) - Alert Trigger Contract v1

Run with: pytest tests/test_clinical_state_engine.py -v
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.clinical_state_engine import (
    ClinicalStateEngine,
    ClinicalState,
    AlertSeverity,
    EventType,
    ATCConfig,
    StateStorage,
    evaluate_patient_alert,
    get_patient_state,
    get_alert_history,
    get_atc_config
)


class TestStateMapping:
    """Test risk score to state mapping."""

    def test_s0_stable_range(self):
        engine = ClinicalStateEngine()
        assert engine.map_score_to_state(0.0) == ClinicalState.S0_STABLE
        assert engine.map_score_to_state(0.15) == ClinicalState.S0_STABLE
        assert engine.map_score_to_state(0.29) == ClinicalState.S0_STABLE

    def test_s1_watch_range(self):
        engine = ClinicalStateEngine()
        assert engine.map_score_to_state(0.30) == ClinicalState.S1_WATCH
        assert engine.map_score_to_state(0.40) == ClinicalState.S1_WATCH
        assert engine.map_score_to_state(0.54) == ClinicalState.S1_WATCH

    def test_s2_escalating_range(self):
        engine = ClinicalStateEngine()
        assert engine.map_score_to_state(0.55) == ClinicalState.S2_ESCALATING
        assert engine.map_score_to_state(0.65) == ClinicalState.S2_ESCALATING
        assert engine.map_score_to_state(0.79) == ClinicalState.S2_ESCALATING

    def test_s3_critical_range(self):
        engine = ClinicalStateEngine()
        assert engine.map_score_to_state(0.80) == ClinicalState.S3_CRITICAL
        assert engine.map_score_to_state(0.90) == ClinicalState.S3_CRITICAL
        assert engine.map_score_to_state(1.0) == ClinicalState.S3_CRITICAL

    def test_custom_thresholds(self):
        config = ATCConfig(s0_upper=0.20, s1_upper=0.40, s2_upper=0.60)
        engine = ClinicalStateEngine(config=config)

        assert engine.map_score_to_state(0.15) == ClinicalState.S0_STABLE
        assert engine.map_score_to_state(0.25) == ClinicalState.S1_WATCH
        assert engine.map_score_to_state(0.50) == ClinicalState.S2_ESCALATING
        assert engine.map_score_to_state(0.70) == ClinicalState.S3_CRITICAL


class TestEscalationDetection:
    """Test state transition and escalation detection."""

    def test_escalation_detection(self):
        engine = ClinicalStateEngine()

        # Escalations
        assert engine.is_escalation(ClinicalState.S0_STABLE, ClinicalState.S1_WATCH) == True
        assert engine.is_escalation(ClinicalState.S0_STABLE, ClinicalState.S2_ESCALATING) == True
        assert engine.is_escalation(ClinicalState.S1_WATCH, ClinicalState.S2_ESCALATING) == True
        assert engine.is_escalation(ClinicalState.S2_ESCALATING, ClinicalState.S3_CRITICAL) == True

        # De-escalations
        assert engine.is_escalation(ClinicalState.S3_CRITICAL, ClinicalState.S2_ESCALATING) == False
        assert engine.is_escalation(ClinicalState.S2_ESCALATING, ClinicalState.S1_WATCH) == False

        # Same state
        assert engine.is_escalation(ClinicalState.S1_WATCH, ClinicalState.S1_WATCH) == False

    def test_initial_state_escalation(self):
        engine = ClinicalStateEngine()

        # Starting above stable should fire
        assert engine.is_escalation(None, ClinicalState.S1_WATCH) == True
        assert engine.is_escalation(None, ClinicalState.S2_ESCALATING) == True

        # Starting at stable should not fire
        assert engine.is_escalation(None, ClinicalState.S0_STABLE) == False


class TestVelocityCalculation:
    """Test risk score velocity calculation."""

    def test_velocity_with_history(self):
        engine = ClinicalStateEngine()
        now = datetime.utcnow()

        # Score went from 0.4 to 0.6 in 30 minutes = 0.4/hr
        last_scores = [(now - timedelta(minutes=30), 0.4)]
        velocity = engine.calculate_velocity(0.6, now, last_scores)

        assert abs(velocity - 0.4) < 0.01

    def test_velocity_no_history(self):
        engine = ClinicalStateEngine()
        now = datetime.utcnow()

        velocity = engine.calculate_velocity(0.5, now, [])
        assert velocity == 0.0

    def test_velocity_within_window(self):
        engine = ClinicalStateEngine()
        now = datetime.utcnow()

        # Multiple scores, should use earliest in window
        last_scores = [
            (now - timedelta(minutes=45), 0.3),
            (now - timedelta(minutes=30), 0.4),
            (now - timedelta(minutes=15), 0.5),
        ]
        velocity = engine.calculate_velocity(0.6, now, last_scores)

        # From 0.3 to 0.6 over 45 minutes = 0.4/hr
        assert abs(velocity - 0.4) < 0.01


class TestNoveltyDetection:
    """Test biomarker novelty detection."""

    def test_novelty_new_biomarker(self):
        engine = ClinicalStateEngine()

        current = ["crp", "lactate", "wbc"]
        previous = ["crp", "glucose", "potassium"]

        detected, new_markers = engine.detect_novelty(current, previous)

        assert detected == True
        assert "lactate" in new_markers or "wbc" in new_markers

    def test_no_novelty_same_biomarkers(self):
        engine = ClinicalStateEngine()

        current = ["crp", "lactate", "wbc"]
        previous = ["crp", "lactate", "wbc"]

        detected, new_markers = engine.detect_novelty(current, previous)

        assert detected == False
        assert new_markers == []


class TestAlertFiring:
    """Test alert firing logic."""

    def test_escalation_fires_alert(self):
        engine = ClinicalStateEngine()
        now = datetime.utcnow()

        should_fire, rationale = engine.should_fire_alert(
            from_state=ClinicalState.S1_WATCH,
            to_state=ClinicalState.S2_ESCALATING,
            velocity=0.0,
            novelty_detected=False,
            last_alert_time=now - timedelta(minutes=5),
            current_time=now
        )

        assert should_fire == True
        assert "escalation" in rationale.lower()

    def test_deescalation_no_alert(self):
        engine = ClinicalStateEngine()
        now = datetime.utcnow()

        should_fire, rationale = engine.should_fire_alert(
            from_state=ClinicalState.S2_ESCALATING,
            to_state=ClinicalState.S1_WATCH,
            velocity=0.0,
            novelty_detected=False,
            last_alert_time=now - timedelta(minutes=5),
            current_time=now
        )

        assert should_fire == False
        assert "de-escalation" in rationale.lower()

    def test_cooldown_suppression(self):
        engine = ClinicalStateEngine()
        now = datetime.utcnow()

        should_fire, rationale = engine.should_fire_alert(
            from_state=ClinicalState.S1_WATCH,
            to_state=ClinicalState.S1_WATCH,
            velocity=0.05,  # Below threshold
            novelty_detected=False,
            last_alert_time=now - timedelta(minutes=30),  # Within cooldown
            current_time=now
        )

        assert should_fire == False
        assert "cooldown" in rationale.lower()

    def test_velocity_override(self):
        engine = ClinicalStateEngine()
        now = datetime.utcnow()

        should_fire, rationale = engine.should_fire_alert(
            from_state=ClinicalState.S1_WATCH,
            to_state=ClinicalState.S1_WATCH,
            velocity=0.25,  # Above threshold
            novelty_detected=False,
            last_alert_time=now - timedelta(minutes=30),
            current_time=now
        )

        assert should_fire == True
        assert "velocity" in rationale.lower()

    def test_novelty_override(self):
        engine = ClinicalStateEngine()
        now = datetime.utcnow()

        should_fire, rationale = engine.should_fire_alert(
            from_state=ClinicalState.S1_WATCH,
            to_state=ClinicalState.S1_WATCH,
            velocity=0.05,  # Below threshold
            novelty_detected=True,
            last_alert_time=now - timedelta(minutes=30),
            current_time=now
        )

        assert should_fire == True
        assert "novelty" in rationale.lower() or "biomarker" in rationale.lower()


class TestFullEvaluation:
    """Test full evaluation flow."""

    def setup_method(self):
        """Fresh storage for each test."""
        self.storage = StateStorage()
        self.engine = ClinicalStateEngine(storage=self.storage)

    def test_first_evaluation_watch_state(self):
        now = datetime.utcnow()

        result = self.engine.evaluate(
            patient_id="P001",
            timestamp=now,
            risk_domain="sepsis",
            current_scores={"composite": 0.45},
            contributing_biomarkers=["crp", "lactate", "wbc"]
        )

        assert result["state_now"] == "S1"
        assert result["state_name"] == "Watch"
        assert result["state_transition"] == True
        assert result["alert_event"] is not None
        assert result["severity"] == "WARNING"

    def test_first_evaluation_stable_no_alert(self):
        now = datetime.utcnow()

        result = self.engine.evaluate(
            patient_id="P002",
            timestamp=now,
            risk_domain="sepsis",
            current_scores={"composite": 0.15},
            contributing_biomarkers=["crp"]
        )

        assert result["state_now"] == "S0"
        assert result["alert_event"] is None

    def test_escalation_sequence(self):
        now = datetime.utcnow()

        # First: Watch state
        r1 = self.engine.evaluate(
            patient_id="P003",
            timestamp=now,
            risk_domain="cardiac",
            current_scores={"composite": 0.40}
        )
        assert r1["state_now"] == "S1"
        assert r1["alert_event"] is not None

        # Second: Escalate to Escalating
        r2 = self.engine.evaluate(
            patient_id="P003",
            timestamp=now + timedelta(minutes=30),
            risk_domain="cardiac",
            current_scores={"composite": 0.65}
        )
        assert r2["state_now"] == "S2"
        assert r2["state_from"] == "S1"
        assert r2["state_transition"] == True
        assert r2["alert_event"] is not None

        # Third: Critical
        r3 = self.engine.evaluate(
            patient_id="P003",
            timestamp=now + timedelta(minutes=60),
            risk_domain="cardiac",
            current_scores={"composite": 0.85}
        )
        assert r3["state_now"] == "S3"
        assert r3["alert_event"] is not None
        assert r3["severity"] == "CRITICAL"

    def test_same_episode_tracking(self):
        now = datetime.utcnow()

        r1 = self.engine.evaluate(
            patient_id="P004",
            timestamp=now,
            risk_domain="respiratory",
            current_scores={"composite": 0.45}
        )
        episode_id = r1["episode_id"]

        r2 = self.engine.evaluate(
            patient_id="P004",
            timestamp=now + timedelta(minutes=30),
            risk_domain="respiratory",
            current_scores={"composite": 0.60}
        )

        # Same episode
        assert r2["episode_id"] == episode_id


class TestAPIHelpers:
    """Test convenience API functions."""

    def test_evaluate_patient_alert(self):
        result = evaluate_patient_alert(
            patient_id="TEST001",
            timestamp=datetime.utcnow().isoformat(),
            risk_domain="sepsis",
            current_scores={"risk": 0.55}
        )

        assert "state_now" in result
        assert "severity" in result

    def test_get_atc_config(self):
        config = get_atc_config()

        assert "s0_upper" in config
        assert "s1_upper" in config
        assert "default_cooldown_minutes" in config
        assert config["s0_upper"] == 0.30


class TestAuditLog:
    """Test audit logging."""

    def setup_method(self):
        self.storage = StateStorage()
        self.engine = ClinicalStateEngine(storage=self.storage)

    def test_events_logged(self):
        now = datetime.utcnow()

        self.engine.evaluate(
            patient_id="P010",
            timestamp=now,
            risk_domain="sepsis",
            current_scores={"composite": 0.50}
        )

        events = self.storage.get_events(patient_id="P010")
        assert len(events) == 1
        assert events[0].event_type in [EventType.ALERT_FIRED, EventType.ALERT_SUPPRESSED]

    def test_multiple_events_logged(self):
        now = datetime.utcnow()

        for i in range(5):
            self.engine.evaluate(
                patient_id="P011",
                timestamp=now + timedelta(minutes=i * 30),
                risk_domain="sepsis",
                current_scores={"composite": 0.40 + i * 0.1}
            )

        events = self.storage.get_events(patient_id="P011")
        assert len(events) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
