"""
Alert System Models - Unified Implementation
Merges hypercore-ml-service + cse.py features

All enums, data classes, and type definitions for the alert system.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid


# =============================================================================
# CLINICAL STATE MODEL (4-State)
# =============================================================================

class ClinicalState(str, Enum):
    """Four-state clinical model with severity levels."""
    S0_STABLE = "S0"
    S1_WATCH = "S1"
    S2_ESCALATING = "S2"
    S3_CRITICAL = "S3"

    @property
    def severity_level(self) -> int:
        """Numeric severity for comparison (0-3)."""
        return {"S0": 0, "S1": 1, "S2": 2, "S3": 3}[self.value]

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        return {
            "S0": "Stable",
            "S1": "Watch",
            "S2": "Escalating",
            "S3": "Critical"
        }[self.value]

    @classmethod
    def from_score(cls, score: float, thresholds: Dict[str, float] = None) -> "ClinicalState":
        """Map risk score to clinical state using thresholds."""
        thresholds = thresholds or {"s0_upper": 0.30, "s1_upper": 0.55, "s2_upper": 0.80}
        if score < thresholds.get("s0_upper", 0.30):
            return cls.S0_STABLE
        elif score < thresholds.get("s1_upper", 0.55):
            return cls.S1_WATCH
        elif score < thresholds.get("s2_upper", 0.80):
            return cls.S2_ESCALATING
        else:
            return cls.S3_CRITICAL


# =============================================================================
# ALERT CLASSIFICATION
# =============================================================================

class AlertType(str, Enum):
    """Alert type classification."""
    INTERRUPTIVE = "interruptive"      # Requires immediate action (pager, popup)
    NON_INTERRUPTIVE = "non_interruptive"  # Informational (banner, queue)
    NONE = "none"                      # No alert (suppressed)


class AlertSeverity(str, Enum):
    """Alert severity levels mapped to states."""
    INFO = "INFO"          # S0 level
    WARNING = "WARNING"    # S1 level
    URGENT = "URGENT"      # S2 level
    CRITICAL = "CRITICAL"  # S3 level

    @classmethod
    def from_state(cls, state: ClinicalState) -> "AlertSeverity":
        """Map clinical state to severity."""
        mapping = {
            ClinicalState.S0_STABLE: cls.INFO,
            ClinicalState.S1_WATCH: cls.WARNING,
            ClinicalState.S2_ESCALATING: cls.URGENT,
            ClinicalState.S3_CRITICAL: cls.CRITICAL,
        }
        return mapping.get(state, cls.INFO)


class ConfidenceLevel(str, Enum):
    """Confidence levels for findings (6-tier system)."""
    VERY_LOW = "very_low"      # 0.00 - 0.19
    LOW = "low"                 # 0.20 - 0.39
    MODERATE = "moderate"       # 0.40 - 0.59
    HIGH = "high"               # 0.60 - 0.79
    VERY_HIGH = "very_high"     # 0.80 - 0.94
    DEFINITIVE = "definitive"   # 0.95 - 0.99

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Map numeric confidence to level."""
        if score < 0.20:
            return cls.VERY_LOW
        elif score < 0.40:
            return cls.LOW
        elif score < 0.60:
            return cls.MODERATE
        elif score < 0.80:
            return cls.HIGH
        elif score < 0.95:
            return cls.VERY_HIGH
        else:
            return cls.DEFINITIVE


class SuppressionReason(str, Enum):
    """Reasons for suppressing an alert."""
    COOLDOWN_ACTIVE = "cooldown_active"
    SAME_STATE_NO_BREAK = "same_state_no_break"
    DOWNWARD_TRANSITION_NOT_RESOLVE = "downward_transition_not_resolve"
    DE_ESCALATION = "de_escalation"


class BreakRule(str, Enum):
    """Break rules that override cooldown/suppression."""
    VELOCITY_SPIKE = "velocity_spike"
    NOVELTY_DETECTION = "novelty_detection"
    TTH_SHORTENING = "tth_shortening"
    DWELL_ESCALATION = "dwell_escalation"


class HarmType(str, Enum):
    """Types of clinical harm for TTH prediction."""
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


class EventType(str, Enum):
    """Audit event types."""
    ALERT_FIRED = "ALERT_FIRED"
    ALERT_SUPPRESSED = "ALERT_SUPPRESSED"
    STATE_CHANGE = "STATE_CHANGE"
    EPISODE_OPENED = "EPISODE_OPENED"
    EPISODE_CLOSED = "EPISODE_CLOSED"
    ACKNOWLEDGMENT = "ACKNOWLEDGMENT"
    ESCALATION = "ESCALATION"


# =============================================================================
# RISK DOMAINS
# =============================================================================

class RiskDomain(str, Enum):
    """All supported risk domains (merged from both systems)."""
    # From hypercore
    SEPSIS = "sepsis"
    CARDIAC = "deterioration_cardiac"
    KIDNEY = "kidney_injury"
    RESPIRATORY = "respiratory_failure"
    HEPATIC = "hepatic_dysfunction"
    NEUROLOGICAL = "neurological"
    METABOLIC = "metabolic"
    HEMATOLOGIC = "hematologic"
    ONCOLOGY = "oncology_inception"
    MULTI_SYSTEM = "multi_system"
    # From cse.py
    DETERIORATION = "deterioration"
    INFECTION = "infection"
    OUTBREAK = "outbreak"
    TRIAL_CONFOUNDER = "trial_confounder"
    CUSTOM = "custom"
    # Fallback
    UNKNOWN = "unknown"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BreakRuleResult:
    """Result of checking a break rule."""
    triggered: bool
    rule: Optional[BreakRule]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "rule": self.rule.value if self.rule else None,
            "details": self.details
        }


@dataclass
class EpisodeState:
    """Tracks an alert episode lifecycle."""
    episode_id: str
    patient_id: str
    risk_domain: str
    opened_at: datetime
    opened_state: ClinicalState
    closed_at: Optional[datetime] = None
    closed_reason: Optional[str] = None
    highest_state: ClinicalState = field(default=ClinicalState.S0_STABLE)
    alert_count: int = 0
    last_alert_time: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def duration_hours(self) -> float:
        end = self.closed_at or datetime.now(timezone.utc)
        return (end - self.opened_at).total_seconds() / 3600.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "patient_id": self.patient_id,
            "risk_domain": self.risk_domain,
            "opened_at": self.opened_at.isoformat(),
            "opened_state": self.opened_state.value,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "closed_reason": self.closed_reason,
            "highest_state": self.highest_state.value,
            "alert_count": self.alert_count,
            "last_alert_time": self.last_alert_time.isoformat() if self.last_alert_time else None,
            "is_open": self.is_open,
            "duration_hours": round(self.duration_hours, 2),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


@dataclass
class PatientState:
    """Persisted state for a patient + risk domain."""
    patient_id: str
    risk_domain: str
    current_state: ClinicalState
    risk_score: float
    episode: Optional[EpisodeState]
    last_score_time: datetime
    last_scores: List[Tuple[datetime, float]] = field(default_factory=list)
    contributing_biomarkers: List[str] = field(default_factory=list)
    last_tth_hours: Optional[float] = None
    time_in_current_state_hours: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "risk_domain": self.risk_domain,
            "current_state": self.current_state.value,
            "risk_score": round(self.risk_score, 4),
            "episode": self.episode.to_dict() if self.episode else None,
            "last_score_time": self.last_score_time.isoformat(),
            "contributing_biomarkers": self.contributing_biomarkers,
            "last_tth_hours": self.last_tth_hours,
            "time_in_current_state_hours": round(self.time_in_current_state_hours, 2),
        }


@dataclass
class AlertEvent:
    """Complete alert event record for audit logging."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    patient_id: str
    risk_domain: str
    episode_id: Optional[str]
    # State info
    state_previous: Optional[ClinicalState]
    state_current: ClinicalState
    state_transition: bool
    risk_score: float
    # Alert decision
    alert_fired: bool
    alert_type: AlertType
    severity: AlertSeverity
    # Suppression
    suppression_reason: Optional[SuppressionReason]
    # Break rules
    break_rules_checked: List[str] = field(default_factory=list)
    break_rules_triggered: List[str] = field(default_factory=list)
    # Details
    contributing_biomarkers: List[str] = field(default_factory=list)
    velocity: float = 0.0
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.MODERATE
    thresholds_used: Dict[str, float] = field(default_factory=dict)
    # Clinical content
    rationale: str = ""
    clinical_headline: str = ""
    clinical_rationale: str = ""
    suggested_action: str = ""
    recommendations: List[str] = field(default_factory=list)
    # TTH integration
    time_to_harm_hours: Optional[float] = None
    intervention_window: Optional[str] = None
    # Timing
    evaluation_duration_ms: float = 0.0
    cooldown_remaining_minutes: float = 0.0
    # Routing
    routed_to: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "patient_id": self.patient_id,
            "risk_domain": self.risk_domain,
            "episode_id": self.episode_id,
            "state_previous": self.state_previous.value if self.state_previous else None,
            "state_current": self.state_current.value,
            "state_transition": self.state_transition,
            "risk_score": round(self.risk_score, 4),
            "alert_fired": self.alert_fired,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "suppression_reason": self.suppression_reason.value if self.suppression_reason else None,
            "break_rules_checked": self.break_rules_checked,
            "break_rules_triggered": self.break_rules_triggered,
            "contributing_biomarkers": self.contributing_biomarkers,
            "velocity": round(self.velocity, 4),
            "confidence": round(self.confidence, 4),
            "confidence_level": self.confidence_level.value,
            "thresholds_used": self.thresholds_used,
            "rationale": self.rationale,
            "clinical_headline": self.clinical_headline,
            "clinical_rationale": self.clinical_rationale,
            "suggested_action": self.suggested_action,
            "recommendations": self.recommendations,
            "time_to_harm_hours": round(self.time_to_harm_hours, 1) if self.time_to_harm_hours else None,
            "intervention_window": self.intervention_window,
            "evaluation_duration_ms": round(self.evaluation_duration_ms, 2),
            "cooldown_remaining_minutes": round(self.cooldown_remaining_minutes, 1),
            "routed_to": self.routed_to,
        }


@dataclass
class AcknowledgmentRecord:
    """Record of an alert acknowledgment."""
    ack_id: str
    alert_id: str
    patient_id: str
    episode_id: str
    acknowledged_by: str
    acknowledged_at: datetime
    action_taken: Optional[str] = None
    notes: Optional[str] = None
    close_episode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ack_id": self.ack_id,
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "episode_id": self.episode_id,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat(),
            "action_taken": self.action_taken,
            "notes": self.notes,
            "close_episode": self.close_episode,
        }


@dataclass
class EvaluationResult:
    """Complete result of a patient evaluation."""
    # Core result
    patient_id: str
    risk_domain: str
    timestamp: datetime
    risk_score: float
    state_now: ClinicalState
    state_previous: Optional[ClinicalState]
    state_transition: bool
    # Alert decision
    alert_fired: bool
    alert_type: AlertType
    alert_event: Optional[AlertEvent]
    # Suppression info
    suppression_reason: Optional[SuppressionReason]
    # Break rules
    break_rules: List[BreakRuleResult]
    # Clinical content
    severity: AlertSeverity
    confidence: float
    clinical_headline: str
    clinical_rationale: str
    suggested_action: str
    contributing_biomarkers: List[str]
    # Episode
    episode: Optional[EpisodeState]
    # TTH
    time_to_harm: Optional[Dict[str, Any]]
    # Timing
    evaluation_duration_ms: float
    # Agent findings (populated by pipeline)
    agent_findings: Optional[Dict[str, Any]] = None
    # Cascade detection (multi-omic early warning)
    cascade_detection: Optional[Dict[str, Any]] = None
    # Unified intelligence layer integration
    unified_intelligence: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "risk_domain": self.risk_domain,
            "timestamp": self.timestamp.isoformat(),
            "risk_score": round(self.risk_score, 4),
            "state_now": self.state_now.value,
            "state_name": self.state_now.display_name,
            "state_previous": self.state_previous.value if self.state_previous else None,
            "state_transition": self.state_transition,
            "alert_fired": self.alert_fired,
            "alert_type": self.alert_type.value,
            "alert_event": self.alert_event.to_dict() if self.alert_event else None,
            "suppression_reason": self.suppression_reason.value if self.suppression_reason else None,
            "break_rules": [br.to_dict() for br in self.break_rules],
            "severity": self.severity.value,
            "confidence": round(self.confidence, 4),
            "clinical_headline": self.clinical_headline,
            "clinical_rationale": self.clinical_rationale,
            "suggested_action": self.suggested_action,
            "contributing_biomarkers": self.contributing_biomarkers,
            "episode": self.episode.to_dict() if self.episode else None,
            "time_to_harm": self.time_to_harm,
            "evaluation_duration_ms": round(self.evaluation_duration_ms, 2),
            "agent_findings": self.agent_findings,
            "cascade_detection": self.cascade_detection,
            "unified_intelligence": self.unified_intelligence,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_event_id() -> str:
    """Generate a unique event ID."""
    return f"evt_{uuid.uuid4().hex[:12]}"


def generate_episode_id() -> str:
    """Generate a unique episode ID."""
    return f"ep_{uuid.uuid4().hex[:8]}"


def generate_ack_id() -> str:
    """Generate a unique acknowledgment ID."""
    return f"ack_{uuid.uuid4().hex[:8]}"
