"""
Clinical State Engine - Unified Implementation
Merges hypercore-ml-service + cse.py with ALL features.
"""

import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone

from .models import (
    ClinicalState,
    AlertType,
    AlertSeverity,
    ConfidenceLevel,
    SuppressionReason,
    BreakRule,
    EventType,
    AlertEvent,
    EpisodeState,
    PatientState,
    EvaluationResult,
    BreakRuleResult,
    generate_event_id,
    generate_episode_id,
)
from .config import (
    DomainConfig,
    get_domain_config,
    get_biomarker_thresholds,
    RATIONALE_TEMPLATES,
    SUGGESTED_ACTIONS,
    get_recommendations,
    get_site_config,
)
from .storage import StorageBackend, get_storage

logger = logging.getLogger(__name__)


# =============================================================================
# CLINICAL STATE ENGINE
# =============================================================================

class ClinicalStateEngine:
    """
    Core alert evaluation engine implementing merged ATC v1 + cse.py features.

    Features:
    - 4-state clinical model (S0-S3)
    - All domain-specific thresholds from hypercore
    - All break rules (velocity, novelty, TTH shortening, dwell escalation)
    - Domain-specific cooldowns
    - Episode lifecycle management
    - Comprehensive audit logging
    - Alert type classification (INTERRUPTIVE, NON_INTERRUPTIVE, NONE)
    """

    def __init__(
        self,
        config: Optional[DomainConfig] = None,
        storage: Optional[StorageBackend] = None,
    ):
        self.config = config or DomainConfig()
        self.storage = storage or get_storage()

    def map_score_to_state(self, risk_score: float, config: DomainConfig = None) -> ClinicalState:
        """Map a risk score to clinical state using configured thresholds."""
        cfg = config or self.config
        if risk_score < cfg.s0_upper:
            return ClinicalState.S0_STABLE
        elif risk_score < cfg.s1_upper:
            return ClinicalState.S1_WATCH
        elif risk_score < cfg.s2_upper:
            return ClinicalState.S2_ESCALATING
        else:
            return ClinicalState.S3_CRITICAL

    def calculate_velocity(
        self,
        current_score: float,
        current_time: datetime,
        last_scores: List[Tuple[datetime, float]],
        window_hours: float = 1.0,
    ) -> float:
        """Calculate score velocity (change per hour)."""
        if not last_scores:
            return 0.0

        # Find scores within velocity window
        window_start = current_time - timedelta(hours=window_hours)
        recent_scores = [(t, s) for t, s in last_scores if t >= window_start]

        if not recent_scores:
            # Use most recent score regardless of time
            last_time, last_score = last_scores[-1]
            hours_elapsed = (current_time - last_time).total_seconds() / 3600.0
            if hours_elapsed > 0.01:
                return (current_score - last_score) / hours_elapsed
            return 0.0

        # Use earliest score in window
        earliest_time, earliest_score = recent_scores[0]
        hours_elapsed = (current_time - earliest_time).total_seconds() / 3600.0

        if hours_elapsed > 0.01:
            return (current_score - earliest_score) / hours_elapsed
        return 0.0

    def detect_novelty(
        self,
        current_biomarkers: List[str],
        previous_biomarkers: List[str],
    ) -> Tuple[bool, List[str]]:
        """Detect if new biomarkers have entered the contributing set (top 3)."""
        current_set = set(b.lower() for b in current_biomarkers[:3])
        previous_set = set(b.lower() for b in previous_biomarkers[:3])
        new_markers = current_set - previous_set

        return len(new_markers) > 0, list(new_markers)

    def check_tth_shortening(
        self,
        current_tth_hours: Optional[float],
        previous_tth_hours: Optional[float],
        threshold: float = 0.25,
    ) -> BreakRuleResult:
        """Check if time-to-harm decreased by more than threshold (25% default)."""
        if current_tth_hours is None or previous_tth_hours is None:
            return BreakRuleResult(triggered=False, rule=None)

        if previous_tth_hours <= 0:
            return BreakRuleResult(triggered=False, rule=None)

        decrease_pct = (previous_tth_hours - current_tth_hours) / previous_tth_hours

        if decrease_pct >= threshold:
            return BreakRuleResult(
                triggered=True,
                rule=BreakRule.TTH_SHORTENING,
                details={
                    "previous_tth_hours": previous_tth_hours,
                    "current_tth_hours": current_tth_hours,
                    "decrease_pct": round(decrease_pct * 100, 1),
                }
            )
        return BreakRuleResult(triggered=False, rule=None)

    def check_dwell_escalation(
        self,
        current_state: ClinicalState,
        time_in_state_hours: float,
        acknowledged: bool,
        dwell_threshold_hours: float = 4.0,
    ) -> BreakRuleResult:
        """Check for dwell escalation (S2+ for >4hr without ack)."""
        if current_state.severity_level < 2:
            return BreakRuleResult(triggered=False, rule=None)

        if acknowledged:
            return BreakRuleResult(triggered=False, rule=None)

        if time_in_state_hours >= dwell_threshold_hours:
            return BreakRuleResult(
                triggered=True,
                rule=BreakRule.DWELL_ESCALATION,
                details={
                    "state": current_state.value,
                    "hours_in_state": round(time_in_state_hours, 2),
                    "threshold_hours": dwell_threshold_hours,
                }
            )
        return BreakRuleResult(triggered=False, rule=None)

    def check_all_break_rules(
        self,
        velocity: float,
        novelty_detected: bool,
        new_markers: List[str],
        current_tth_hours: Optional[float],
        previous_tth_hours: Optional[float],
        current_state: ClinicalState,
        time_in_state_hours: float,
        acknowledged: bool,
        config: DomainConfig,
    ) -> List[BreakRuleResult]:
        """Check all break rules and return results."""
        results = []

        # 1. Velocity spike
        if config.velocity_override_enabled and abs(velocity) > config.velocity_threshold:
            results.append(BreakRuleResult(
                triggered=True,
                rule=BreakRule.VELOCITY_SPIKE,
                details={
                    "velocity": round(velocity, 4),
                    "threshold": config.velocity_threshold,
                }
            ))
        else:
            results.append(BreakRuleResult(triggered=False, rule=BreakRule.VELOCITY_SPIKE))

        # 2. Novelty detection
        if config.novelty_detection_enabled and novelty_detected:
            results.append(BreakRuleResult(
                triggered=True,
                rule=BreakRule.NOVELTY_DETECTION,
                details={"new_markers": new_markers}
            ))
        else:
            results.append(BreakRuleResult(triggered=False, rule=BreakRule.NOVELTY_DETECTION))

        # 3. TTH shortening
        if config.tth_shortening_enabled:
            tth_result = self.check_tth_shortening(
                current_tth_hours,
                previous_tth_hours,
                config.tth_shortening_threshold,
            )
            results.append(tth_result)
        else:
            results.append(BreakRuleResult(triggered=False, rule=BreakRule.TTH_SHORTENING))

        # 4. Dwell escalation
        if config.dwell_escalation_enabled:
            dwell_result = self.check_dwell_escalation(
                current_state,
                time_in_state_hours,
                acknowledged,
                config.dwell_escalation_hours,
            )
            results.append(dwell_result)
        else:
            results.append(BreakRuleResult(triggered=False, rule=BreakRule.DWELL_ESCALATION))

        return results

    def determine_alert_type(
        self,
        from_state: Optional[ClinicalState],
        to_state: ClinicalState,
        break_rules: List[BreakRuleResult],
        within_cooldown: bool,
    ) -> Tuple[AlertType, Optional[SuppressionReason]]:
        """
        Determine alert type and suppression reason.

        INTERRUPTIVE transitions:
        - S2→S3, S0→S3, S1→S3
        - Dwell escalation at S2+
        - TTH shortening at S2+

        NON_INTERRUPTIVE transitions:
        - S0→S1, S1→S2, S0→S2
        - Velocity spike
        - Novelty detection

        NONE:
        - Same state, no break rule
        - Downward transition (not to S0)
        - Within cooldown (suppressed)
        """
        # Check if any break rule triggered
        break_triggered = any(br.triggered for br in break_rules)
        triggered_rules = [br.rule for br in break_rules if br.triggered]

        # First assessment (before new state)
        if from_state is None:
            # Initial assessment
            if to_state.severity_level >= 3:
                return AlertType.INTERRUPTIVE, None
            elif to_state.severity_level >= 1:
                return AlertType.NON_INTERRUPTIVE, None
            else:
                return AlertType.NONE, None

        # State escalation
        if to_state.severity_level > from_state.severity_level:
            # Jump to S3 from any state
            if to_state.severity_level >= 3:
                return AlertType.INTERRUPTIVE, None
            # S0→S2 or S1→S2
            elif to_state.severity_level >= 2:
                return AlertType.NON_INTERRUPTIVE, None
            # S0→S1
            else:
                return AlertType.NON_INTERRUPTIVE, None

        # Same state
        if to_state == from_state:
            # Check break rules
            if BreakRule.DWELL_ESCALATION in triggered_rules or BreakRule.TTH_SHORTENING in triggered_rules:
                if to_state.severity_level >= 2:
                    return AlertType.INTERRUPTIVE, None

            if BreakRule.VELOCITY_SPIKE in triggered_rules or BreakRule.NOVELTY_DETECTION in triggered_rules:
                return AlertType.NON_INTERRUPTIVE, None

            # Within cooldown - suppress
            if within_cooldown:
                return AlertType.NONE, SuppressionReason.COOLDOWN_ACTIVE

            # Same state, no break
            return AlertType.NONE, SuppressionReason.SAME_STATE_NO_BREAK

        # State de-escalation
        if to_state.severity_level < from_state.severity_level:
            if to_state == ClinicalState.S0_STABLE:
                # Resolved to S0 - this is good news but not an alert
                return AlertType.NONE, SuppressionReason.DE_ESCALATION
            else:
                # Downward but not to S0
                return AlertType.NONE, SuppressionReason.DOWNWARD_TRANSITION_NOT_RESOLVE

        return AlertType.NONE, None

    def manage_episode(
        self,
        patient_id: str,
        risk_domain: str,
        timestamp: datetime,
        from_state: Optional[ClinicalState],
        to_state: ClinicalState,
        last_episode: Optional[EpisodeState],
        config: DomainConfig,
    ) -> Optional[EpisodeState]:
        """
        Manage episode lifecycle.

        Episode opens on:
        - State transition from S0 to S1+
        - Initial assessment at S1+

        Episode closes on:
        - Resolve to S0
        - Higher-state transition (new episode)
        - Max duration exceeded
        - Manual acknowledgment (if configured)
        """
        # No previous episode
        if last_episode is None:
            if to_state.severity_level >= 1:
                # Open new episode
                return EpisodeState(
                    episode_id=generate_episode_id(),
                    patient_id=patient_id,
                    risk_domain=risk_domain,
                    opened_at=timestamp,
                    opened_state=to_state,
                    highest_state=to_state,
                )
            return None

        # Existing episode
        episode = last_episode

        # Check for episode closure conditions

        # 1. Resolve to S0
        if to_state == ClinicalState.S0_STABLE and from_state and from_state.severity_level > 0:
            episode.closed_at = timestamp
            episode.closed_reason = "resolved_to_stable"
            self.storage.save_episode(episode)
            return None

        # 2. Max duration exceeded
        if episode.duration_hours >= config.max_episode_duration_hours:
            episode.closed_at = timestamp
            episode.closed_reason = "max_duration_exceeded"
            self.storage.save_episode(episode)
            # Open new episode if still elevated
            if to_state.severity_level >= 1:
                return EpisodeState(
                    episode_id=generate_episode_id(),
                    patient_id=patient_id,
                    risk_domain=risk_domain,
                    opened_at=timestamp,
                    opened_state=to_state,
                    highest_state=to_state,
                )
            return None

        # 3. Check if been at S0 for too long (episode break)
        if last_episode.highest_state == ClinicalState.S0_STABLE:
            # Been stable - check if should start new episode
            if to_state.severity_level >= 1:
                return EpisodeState(
                    episode_id=generate_episode_id(),
                    patient_id=patient_id,
                    risk_domain=risk_domain,
                    opened_at=timestamp,
                    opened_state=to_state,
                    highest_state=to_state,
                )

        # Update highest state
        if to_state.severity_level > episode.highest_state.severity_level:
            episode.highest_state = to_state

        return episode

    def generate_rationale(
        self,
        domain: str,
        state: ClinicalState,
        velocity: float,
        contributing_biomarkers: List[str],
        break_rules: List[BreakRuleResult],
        new_markers: List[str],
    ) -> Tuple[str, str, str]:
        """
        Generate clinical rationale, headline, and suggested action.

        Returns: (headline, rationale, suggested_action)
        """
        templates = RATIONALE_TEMPLATES.get(domain.lower(), RATIONALE_TEMPLATES["default"])

        # Format driver list
        drivers_str = ", ".join(contributing_biomarkers[:3]) if contributing_biomarkers else "multiple markers"
        velocity_str = f"{velocity:+.2f}"

        # Determine headline based on trigger type
        triggered_rules = [br.rule for br in break_rules if br.triggered]

        if BreakRule.TTH_SHORTENING in triggered_rules:
            pct = next((br.details.get("decrease_pct", 0) for br in break_rules
                       if br.rule == BreakRule.TTH_SHORTENING), 0)
            headline = templates.get("tth_shortening", "Deterioration accelerating").format(pct=pct)
        elif BreakRule.DWELL_ESCALATION in triggered_rules:
            hours = next((br.details.get("hours_in_state", 0) for br in break_rules
                         if br.rule == BreakRule.DWELL_ESCALATION), 0)
            headline = templates.get("dwell", "Sustained elevated state").format(hours=round(hours, 1))
        elif BreakRule.VELOCITY_SPIKE in triggered_rules:
            headline = templates.get("velocity", "Rapid change detected").format(velocity=velocity_str)
        elif BreakRule.NOVELTY_DETECTION in triggered_rules:
            new_str = ", ".join(new_markers[:3])
            headline = templates.get("novelty", "New marker detected").format(new_markers=new_str)
        else:
            # State-based headline
            headline = templates.get(state.value, templates.get("S1", "Risk alert"))

        # Build detailed rationale
        state_desc = {
            "S0": "stable",
            "S1": "watch",
            "S2": "escalating",
            "S3": "critical"
        }.get(state.value, "elevated")

        rationale_parts = [
            f"Patient risk state is {state_desc} ({state.value}).",
        ]
        if contributing_biomarkers:
            rationale_parts.append(f"Primary drivers: {drivers_str}.")
        if abs(velocity) > 0.01:
            rationale_parts.append(f"Score velocity: {velocity_str}/hour.")
        if new_markers:
            rationale_parts.append(f"New markers detected: {', '.join(new_markers)}.")

        rationale = " ".join(rationale_parts)

        # Suggested action
        suggested_action = SUGGESTED_ACTIONS.get(state.value, "Review patient status.")

        return headline, rationale, suggested_action

    def evaluate(
        self,
        patient_id: str,
        timestamp: datetime,
        risk_domain: str,
        current_scores: Dict[str, float],
        contributing_biomarkers: Optional[List[str]] = None,
        current_tth_hours: Optional[float] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Main evaluation entry point.

        Args:
            patient_id: Unique patient identifier
            timestamp: Observation timestamp
            risk_domain: Risk category (sepsis, cardiac, etc.)
            current_scores: Dict of score_name -> value (uses max)
            contributing_biomarkers: Top biomarkers driving the score
            current_tth_hours: Current time-to-harm prediction (if available)
            config_override: Optional config overrides

        Returns:
            Complete EvaluationResult with state, alert decision, and all details
        """
        start_time = time.time()

        # Get configuration (with site overrides if available)
        site_config = get_site_config()
        if site_config:
            config = site_config.get_domain_config(risk_domain)
        else:
            config = get_domain_config(risk_domain)

        # Apply any explicit overrides
        if config_override:
            for key, value in config_override.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Calculate aggregate risk score
        if not current_scores:
            risk_score = 0.0
        else:
            risk_score = max(current_scores.values())

        contributing_biomarkers = contributing_biomarkers or []

        # Get last known state
        last_state = self.storage.get_patient_state(patient_id, risk_domain)

        # Map to new state
        new_state = self.map_score_to_state(risk_score, config)

        # Calculate velocity
        last_scores = last_state.last_scores if last_state else []
        velocity = self.calculate_velocity(
            risk_score, timestamp, last_scores, config.velocity_window_hours
        )

        # Detect novelty
        previous_biomarkers = last_state.contributing_biomarkers if last_state else []
        novelty_detected, new_markers = self.detect_novelty(
            contributing_biomarkers, previous_biomarkers
        )

        # Get previous TTH
        previous_tth_hours = last_state.last_tth_hours if last_state else None

        # Calculate time in current state
        if last_state and last_state.current_state == new_state:
            time_in_state = (timestamp - last_state.last_score_time).total_seconds() / 3600.0
            time_in_state += last_state.time_in_current_state_hours
        else:
            time_in_state = 0.0

        # Check episode acknowledgment
        last_episode = last_state.episode if last_state else None
        acknowledged = last_episode.acknowledged if last_episode else False

        # Check all break rules
        break_rules = self.check_all_break_rules(
            velocity=velocity,
            novelty_detected=novelty_detected,
            new_markers=new_markers,
            current_tth_hours=current_tth_hours,
            previous_tth_hours=previous_tth_hours,
            current_state=new_state,
            time_in_state_hours=time_in_state,
            acknowledged=acknowledged,
            config=config,
        )

        # Check cooldown
        from_state = last_state.current_state if last_state else None
        last_alert_time = last_episode.last_alert_time if last_episode else None
        cooldown_minutes = config.get_cooldown_for_state(new_state.severity_level)

        within_cooldown = False
        cooldown_remaining = 0.0
        if last_alert_time and from_state == new_state:
            minutes_since = (timestamp - last_alert_time).total_seconds() / 60.0
            if minutes_since < cooldown_minutes:
                within_cooldown = True
                cooldown_remaining = cooldown_minutes - minutes_since

        # Determine alert type
        alert_type, suppression_reason = self.determine_alert_type(
            from_state=from_state,
            to_state=new_state,
            break_rules=break_rules,
            within_cooldown=within_cooldown,
        )

        # State transition flag
        state_transition = from_state != new_state if from_state else True

        # Manage episode
        episode = self.manage_episode(
            patient_id=patient_id,
            risk_domain=risk_domain,
            timestamp=timestamp,
            from_state=from_state,
            to_state=new_state,
            last_episode=last_episode,
            config=config,
        )

        # Generate clinical content
        headline, rationale_text, suggested_action = self.generate_rationale(
            domain=risk_domain,
            state=new_state,
            velocity=velocity,
            contributing_biomarkers=contributing_biomarkers,
            break_rules=break_rules,
            new_markers=new_markers,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            risk_score=risk_score,
            velocity=velocity,
            break_rules=break_rules,
            contributing_biomarkers=contributing_biomarkers,
        )

        # Build severity
        severity = AlertSeverity.from_state(new_state)

        # Create alert event
        alert_fired = alert_type != AlertType.NONE
        event_type = EventType.ALERT_FIRED if alert_fired else EventType.ALERT_SUPPRESSED

        alert_event = AlertEvent(
            event_id=generate_event_id(),
            event_type=event_type,
            timestamp=timestamp,
            patient_id=patient_id,
            risk_domain=risk_domain,
            episode_id=episode.episode_id if episode else None,
            state_previous=from_state,
            state_current=new_state,
            state_transition=state_transition,
            risk_score=risk_score,
            alert_fired=alert_fired,
            alert_type=alert_type,
            severity=severity,
            suppression_reason=suppression_reason,
            break_rules_checked=[br.rule.value if br.rule else "none" for br in break_rules],
            break_rules_triggered=[br.rule.value for br in break_rules if br.triggered],
            contributing_biomarkers=contributing_biomarkers,
            velocity=velocity,
            confidence=confidence,
            confidence_level=ConfidenceLevel.from_score(confidence),
            thresholds_used={
                "s0_upper": config.s0_upper,
                "s1_upper": config.s1_upper,
                "s2_upper": config.s2_upper,
                "velocity_threshold": config.velocity_threshold,
            },
            rationale=f"{alert_type.value}: {suppression_reason.value if suppression_reason else 'fired'}",
            clinical_headline=headline,
            clinical_rationale=rationale_text,
            suggested_action=suggested_action,
            recommendations=get_recommendations(
                risk_domain,
                "immediate" if new_state.severity_level >= 3 else
                "urgent" if new_state.severity_level >= 2 else "monitor"
            ),
            time_to_harm_hours=current_tth_hours,
            intervention_window=(
                "immediate" if current_tth_hours and current_tth_hours <= 6 else
                "urgent" if current_tth_hours and current_tth_hours <= 24 else
                "monitor" if current_tth_hours and current_tth_hours <= 72 else
                "stable"
            ) if current_tth_hours else None,
            evaluation_duration_ms=(time.time() - start_time) * 1000,
            cooldown_remaining_minutes=cooldown_remaining,
        )

        # Log event
        self.storage.log_event(alert_event)

        # Update episode if alert fired
        if episode and alert_fired:
            episode.alert_count += 1
            episode.last_alert_time = timestamp

        # Update patient state
        new_scores = [(timestamp, risk_score)]
        if last_state:
            cutoff = timestamp - timedelta(hours=24)
            new_scores = [(t, s) for t, s in last_state.last_scores if t >= cutoff]
            new_scores.append((timestamp, risk_score))

        updated_state = PatientState(
            patient_id=patient_id,
            risk_domain=risk_domain,
            current_state=new_state,
            risk_score=risk_score,
            episode=episode,
            last_score_time=timestamp,
            last_scores=new_scores,
            contributing_biomarkers=contributing_biomarkers,
            last_tth_hours=current_tth_hours,
            time_in_current_state_hours=time_in_state if not state_transition else 0.0,
        )
        self.storage.save_patient_state(updated_state)

        # Build result
        return EvaluationResult(
            patient_id=patient_id,
            risk_domain=risk_domain,
            timestamp=timestamp,
            risk_score=risk_score,
            state_now=new_state,
            state_previous=from_state,
            state_transition=state_transition,
            alert_fired=alert_fired,
            alert_type=alert_type,
            alert_event=alert_event if alert_fired else None,
            suppression_reason=suppression_reason,
            break_rules=break_rules,
            severity=severity,
            confidence=confidence,
            clinical_headline=headline,
            clinical_rationale=rationale_text,
            suggested_action=suggested_action,
            contributing_biomarkers=contributing_biomarkers,
            episode=episode,
            time_to_harm={
                "hours_to_harm": current_tth_hours,
                "intervention_window": alert_event.intervention_window,
            } if current_tth_hours else None,
            evaluation_duration_ms=alert_event.evaluation_duration_ms,
        )

    def _calculate_confidence(
        self,
        risk_score: float,
        velocity: float,
        break_rules: List[BreakRuleResult],
        contributing_biomarkers: List[str],
    ) -> float:
        """Calculate confidence score (0.0 - 0.99)."""
        # Base confidence from risk score magnitude
        base_confidence = min(0.6, risk_score * 0.7)

        # Boost for velocity (trend strength)
        if abs(velocity) > 0.1:
            base_confidence += 0.1
        elif abs(velocity) > 0.05:
            base_confidence += 0.05

        # Boost for multiple biomarkers
        n_biomarkers = len(contributing_biomarkers)
        if n_biomarkers >= 3:
            base_confidence += 0.15
        elif n_biomarkers >= 2:
            base_confidence += 0.10
        elif n_biomarkers >= 1:
            base_confidence += 0.05

        # Boost for multiple break rules triggered
        n_triggered = sum(1 for br in break_rules if br.triggered)
        if n_triggered >= 2:
            base_confidence += 0.1
        elif n_triggered >= 1:
            base_confidence += 0.05

        return min(0.99, max(0.0, base_confidence))


# =============================================================================
# API HELPER FUNCTION
# =============================================================================

def evaluate_patient(
    patient_id: str,
    timestamp: str,
    risk_domain: str,
    current_scores: Dict[str, float],
    contributing_biomarkers: Optional[List[str]] = None,
    current_tth_hours: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    API-friendly function to evaluate a patient.

    Args:
        patient_id: Unique patient identifier
        timestamp: ISO8601 timestamp string
        risk_domain: Risk category
        current_scores: Dict of score_name -> value
        contributing_biomarkers: Top biomarkers
        current_tth_hours: Time-to-harm prediction
        config: Optional configuration overrides

    Returns:
        Evaluation result as dictionary
    """
    # Parse timestamp
    if isinstance(timestamp, str):
        try:
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            ts = datetime.now(timezone.utc)
    else:
        ts = timestamp

    engine = get_engine()
    result = engine.evaluate(
        patient_id=patient_id,
        timestamp=ts,
        risk_domain=risk_domain,
        current_scores=current_scores,
        contributing_biomarkers=contributing_biomarkers,
        current_tth_hours=current_tth_hours,
        config_override=config,
    )

    return result.to_dict()


# =============================================================================
# GLOBAL ENGINE INSTANCE
# =============================================================================

_engine: Optional[ClinicalStateEngine] = None


def get_engine() -> ClinicalStateEngine:
    """Get the global engine instance."""
    global _engine
    if _engine is None:
        _engine = ClinicalStateEngine()
    return _engine


def set_engine(engine: ClinicalStateEngine) -> None:
    """Set a custom engine instance."""
    global _engine
    _engine = engine
