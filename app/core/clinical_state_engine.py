# clinical_state_engine.py
# Location: app/core/clinical_state_engine.py
"""
Clinical State Engine (CSE) - Alert Trigger Contract v1 Implementation

This module implements the 4-state clinical alerting model with:
- State mapping from risk scores
- State transition detection
- Dedupe/cooldown logic
- Audit logging

ALERT TRIGGER CONTRACT (ATC) v1 SPECIFICATION
=============================================

## 1. FOUR-STATE MODEL

| State | Code | Risk Score Range | Clinical Meaning |
|-------|------|------------------|------------------|
| S0    | STABLE | 0.00 - 0.29 | Normal baseline, routine monitoring |
| S1    | WATCH | 0.30 - 0.54 | Elevated concern, enhanced monitoring |
| S2    | ESCALATING | 0.55 - 0.79 | Active deterioration, intervention needed |
| S3    | CRITICAL | 0.80 - 1.00 | Immediate action required |

## 2. STATE MAPPING RULES

Configurable thresholds (defaults shown):
- S0_UPPER = 0.30
- S1_UPPER = 0.55
- S2_UPPER = 0.80
- S3_UPPER = 1.00

Multi-domain aggregation:
- When multiple risk domains exist, use MAX(domain_scores) for state mapping
- OR use weighted average if domain weights are specified

## 3. TRANSITION RULES

Alert-firing transitions (require notification):
- S0 -> S2 (skip-level escalation)
- S0 -> S3 (critical jump)
- S1 -> S2 (standard escalation)
- S1 -> S3 (critical jump from watch)
- S2 -> S3 (escalation to critical)

Non-firing transitions:
- S3 -> S2 (de-escalation) - log only
- S2 -> S1 (de-escalation) - log only
- S1 -> S0 (return to stable) - log only
- Same state (no change) - log only

Re-alert rules (within same state):
- Re-alert if velocity spike: delta_score / delta_time > VELOCITY_THRESHOLD
- Re-alert if novelty detected: new biomarker pattern not seen in previous 24h
- Otherwise, suppress within COOLDOWN_WINDOW

## 4. ESCALATION/BREAK RULES

Re-alert conditions within episode:
1. State escalates (higher state number)
2. Velocity spike: |score_change| > 0.15 within 1 hour
3. Novelty: new contributing biomarker enters top-3 drivers

Break conditions (end current episode, start new):
1. State returns to S0 for > 4 hours
2. Manual clinician acknowledgment
3. Episode duration exceeds MAX_EPISODE_DURATION (default: 72h)

## 5. DEDUPE RULES

Event definition:
- Unique event = (patient_id, risk_domain, state, episode_id)
- Episode boundary: state returns to S0 for > EPISODE_BREAK_HOURS

Time window:
- Default cooldown: 60 minutes for same-state, same-domain
- Escalation overrides cooldown
- Velocity spike overrides cooldown

Suppression logging:
- All suppressed alerts logged to audit trail
- Reason captured: "cooldown", "same_state", "de-escalation"

## 6. AUDIT LOGGING SCHEMA

Alert Event (fired):
```json
{
    "event_id": "uuid",
    "event_type": "ALERT_FIRED",
    "timestamp": "ISO8601",
    "patient_id": "string",
    "risk_domain": "string",
    "episode_id": "string",
    "state_from": "S0|S1|S2|S3",
    "state_to": "S0|S1|S2|S3",
    "risk_score": 0.0-1.0,
    "contributing_biomarkers": ["array"],
    "velocity": 0.0,
    "severity": "INFO|WARNING|URGENT|CRITICAL",
    "rationale": "string",
    "suggested_cooldown_minutes": 60
}
```

Suppressed Candidate:
```json
{
    "event_id": "uuid",
    "event_type": "ALERT_SUPPRESSED",
    "timestamp": "ISO8601",
    "patient_id": "string",
    "risk_domain": "string",
    "episode_id": "string",
    "state_current": "S0|S1|S2|S3",
    "risk_score": 0.0-1.0,
    "suppression_reason": "cooldown|same_state|de-escalation",
    "last_alert_timestamp": "ISO8601",
    "time_since_last_alert_minutes": 0
}
```
"""

from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass, asdict, field
import uuid
import json
from collections import defaultdict

# Import domain classifier (optional, for enhanced mode)
try:
    from app.core.domain_classifier import classify_domains, get_primary_domain
    DOMAIN_CLASSIFIER_AVAILABLE = True
except ImportError:
    DOMAIN_CLASSIFIER_AVAILABLE = False
    classify_domains = None
    get_primary_domain = None


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ClinicalState(str, Enum):
    """Four-state clinical model."""
    S0_STABLE = "S0"
    S1_WATCH = "S1"
    S2_ESCALATING = "S2"
    S3_CRITICAL = "S3"

    @property
    def severity_level(self) -> int:
        """Numeric severity for comparison."""
        return {"S0": 0, "S1": 1, "S2": 2, "S3": 3}[self.value]

    @property
    def display_name(self) -> str:
        return {
            "S0": "Stable",
            "S1": "Watch",
            "S2": "Escalating",
            "S3": "Critical"
        }[self.value]


class AlertSeverity(str, Enum):
    """Alert severity levels mapped to states."""
    INFO = "INFO"
    WARNING = "WARNING"
    URGENT = "URGENT"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    """Audit event types."""
    ALERT_FIRED = "ALERT_FIRED"
    ALERT_SUPPRESSED = "ALERT_SUPPRESSED"
    STATE_CHANGE = "STATE_CHANGE"
    EPISODE_START = "EPISODE_START"
    EPISODE_END = "EPISODE_END"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ATCConfig:
    """Alert Trigger Contract configuration."""
    # State thresholds (upper bounds)
    s0_upper: float = 0.30
    s1_upper: float = 0.55
    s2_upper: float = 0.80
    s3_upper: float = 1.00

    # Cooldown settings
    default_cooldown_minutes: int = 60
    escalation_cooldown_minutes: int = 15
    critical_cooldown_minutes: int = 5

    # Velocity thresholds
    velocity_threshold: float = 0.15  # score change per hour
    velocity_window_hours: float = 1.0

    # Episode settings
    episode_break_hours: float = 4.0
    max_episode_duration_hours: float = 72.0

    # Re-alert settings
    novelty_detection_enabled: bool = True
    velocity_override_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = ATCConfig()


# =============================================================================
# DOMAIN-SPECIFIC CONFIGURATIONS
# =============================================================================

DOMAIN_CONFIGS: Dict[str, ATCConfig] = {
    # Sepsis - aggressive alerting due to rapid deterioration risk
    "sepsis": ATCConfig(
        s0_upper=0.25,
        s1_upper=0.50,
        s2_upper=0.75,
        default_cooldown_minutes=30,
        escalation_cooldown_minutes=10,
        critical_cooldown_minutes=5,
        velocity_threshold=0.12
    ),

    # Cardiac - balanced thresholds
    "deterioration_cardiac": ATCConfig(
        s0_upper=0.30,
        s1_upper=0.55,
        s2_upper=0.80,
        default_cooldown_minutes=45,
        escalation_cooldown_minutes=15,
        critical_cooldown_minutes=5,
        velocity_threshold=0.15
    ),

    # Kidney injury - slightly higher thresholds (slower progression typical)
    "kidney_injury": ATCConfig(
        s0_upper=0.35,
        s1_upper=0.60,
        s2_upper=0.82,
        default_cooldown_minutes=60,
        escalation_cooldown_minutes=20,
        critical_cooldown_minutes=10,
        velocity_threshold=0.10
    ),

    # Respiratory failure - aggressive like sepsis
    "respiratory_failure": ATCConfig(
        s0_upper=0.25,
        s1_upper=0.50,
        s2_upper=0.75,
        default_cooldown_minutes=30,
        escalation_cooldown_minutes=10,
        critical_cooldown_minutes=5,
        velocity_threshold=0.15
    ),

    # Hepatic dysfunction - moderate alerting
    "hepatic_dysfunction": ATCConfig(
        s0_upper=0.32,
        s1_upper=0.58,
        s2_upper=0.82,
        default_cooldown_minutes=60,
        escalation_cooldown_minutes=20,
        critical_cooldown_minutes=10,
        velocity_threshold=0.12
    ),

    # Neurological - aggressive due to time-sensitive nature
    "neurological": ATCConfig(
        s0_upper=0.25,
        s1_upper=0.50,
        s2_upper=0.75,
        default_cooldown_minutes=20,
        escalation_cooldown_minutes=10,
        critical_cooldown_minutes=3,
        velocity_threshold=0.10
    ),

    # Metabolic - moderate alerting
    "metabolic": ATCConfig(
        s0_upper=0.30,
        s1_upper=0.55,
        s2_upper=0.80,
        default_cooldown_minutes=45,
        escalation_cooldown_minutes=15,
        critical_cooldown_minutes=10,
        velocity_threshold=0.15
    ),

    # Hematologic - moderate alerting
    "hematologic": ATCConfig(
        s0_upper=0.30,
        s1_upper=0.55,
        s2_upper=0.80,
        default_cooldown_minutes=60,
        escalation_cooldown_minutes=20,
        critical_cooldown_minutes=10,
        velocity_threshold=0.12
    ),

    # Oncology inception - higher thresholds (requires pattern confirmation)
    "oncology_inception": ATCConfig(
        s0_upper=0.40,
        s1_upper=0.65,
        s2_upper=0.85,
        default_cooldown_minutes=120,
        escalation_cooldown_minutes=60,
        critical_cooldown_minutes=30,
        velocity_threshold=0.08
    ),

    # Multi-system - most aggressive (multiple organ involvement)
    "multi_system": ATCConfig(
        s0_upper=0.20,
        s1_upper=0.45,
        s2_upper=0.70,
        default_cooldown_minutes=20,
        escalation_cooldown_minutes=10,
        critical_cooldown_minutes=3,
        velocity_threshold=0.10
    ),

    # Unknown domain - use defaults
    "unknown": DEFAULT_CONFIG
}


def get_domain_config(domain: str) -> ATCConfig:
    """Get configuration for a specific clinical domain."""
    return DOMAIN_CONFIGS.get(domain.lower(), DEFAULT_CONFIG)


# =============================================================================
# CLINICAL RATIONALE TEMPLATES
# =============================================================================

# Biomarker display names for clinical language
BIOMARKER_DISPLAY_NAMES: Dict[str, str] = {
    # Sepsis markers
    "lactate": "lactate",
    "crp": "C-reactive protein",
    "c_reactive_protein": "C-reactive protein",
    "wbc": "white blood cell count",
    "white_blood_cell": "white blood cell count",
    "procalcitonin": "procalcitonin",
    "pct": "procalcitonin",
    "il6": "interleukin-6",

    # Cardiac markers
    "troponin": "troponin",
    "troponin_i": "troponin I",
    "troponin_t": "troponin T",
    "bnp": "BNP",
    "nt_probnp": "NT-proBNP",
    "ck_mb": "CK-MB",

    # Kidney markers
    "creatinine": "creatinine",
    "bun": "BUN",
    "egfr": "eGFR",
    "cystatin_c": "cystatin C",

    # Respiratory markers
    "pao2": "PaO2",
    "spo2": "SpO2",
    "fio2": "FiO2",
    "pf_ratio": "P/F ratio",
    "paco2": "PaCO2",

    # Hepatic markers
    "alt": "ALT",
    "ast": "AST",
    "bilirubin": "bilirubin",
    "inr": "INR",
    "albumin": "albumin",

    # Neurological markers
    "gcs": "Glasgow Coma Scale",
    "icp": "intracranial pressure",

    # Metabolic markers
    "glucose": "glucose",
    "sodium": "sodium",
    "potassium": "potassium",
    "ph": "blood pH",

    # Hematologic markers
    "hemoglobin": "hemoglobin",
    "platelets": "platelet count",
    "fibrinogen": "fibrinogen",
    "d_dimer": "D-dimer"
}


# Clinical headline templates by domain and state
RATIONALE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "sepsis": {
        "S1": "Early infection markers detected - recommend enhanced monitoring",
        "S2": "Sepsis indicators escalating - consider cultures and source control",
        "S3": "Severe sepsis pattern - immediate sepsis bundle initiation recommended",
        "escalation": "Sepsis markers worsening: {drivers} trending up ({velocity}/hr)",
        "velocity": "Rapid inflammatory marker rise detected ({velocity}/hr)",
        "novelty": "New sepsis marker active: {new_markers} now contributing"
    },
    "deterioration_cardiac": {
        "S1": "Cardiac stress indicators elevated - monitor for ACS/HF",
        "S2": "Significant myocardial injury pattern - cardiology consultation advised",
        "S3": "Critical cardiac event likely - immediate evaluation required",
        "escalation": "Cardiac markers rising: {drivers} ({velocity}/hr)",
        "velocity": "Rapid cardiac marker elevation ({velocity}/hr)",
        "novelty": "New cardiac marker active: {new_markers}"
    },
    "kidney_injury": {
        "S1": "Early renal function decline - optimize volume/nephrotoxin exposure",
        "S2": "Acute kidney injury developing - consider nephrology input",
        "S3": "Severe AKI - urgent renal support evaluation needed",
        "escalation": "Renal function declining: {drivers} ({velocity}/hr)",
        "velocity": "Rapid creatinine/urea rise ({velocity}/hr)",
        "novelty": "New renal marker involved: {new_markers}"
    },
    "respiratory_failure": {
        "S1": "Oxygenation impairment noted - assess respiratory status",
        "S2": "Respiratory failure developing - prepare for escalation",
        "S3": "Critical hypoxemia - immediate ventilatory support needed",
        "escalation": "Oxygenation worsening: {drivers} ({velocity}/hr)",
        "velocity": "Rapid respiratory decline ({velocity}/hr)",
        "novelty": "New respiratory marker: {new_markers}"
    },
    "hepatic_dysfunction": {
        "S1": "Liver enzyme elevation - monitor for hepatotoxicity",
        "S2": "Hepatic dysfunction progressing - assess etiology",
        "S3": "Acute liver failure pattern - hepatology consultation urgent",
        "escalation": "Liver markers rising: {drivers} ({velocity}/hr)",
        "velocity": "Rapid liver enzyme elevation ({velocity}/hr)",
        "novelty": "New hepatic marker: {new_markers}"
    },
    "neurological": {
        "S1": "Neurological changes detected - frequent neuro checks",
        "S2": "Significant neurological decline - urgent evaluation needed",
        "S3": "Critical neurological event - immediate neurology/neurosurgery",
        "escalation": "Neuro status declining: {drivers} ({velocity}/hr)",
        "velocity": "Rapid neurological change ({velocity}/hr)",
        "novelty": "New neurological finding: {new_markers}"
    },
    "metabolic": {
        "S1": "Metabolic derangement detected - correct electrolytes",
        "S2": "Significant metabolic imbalance - frequent monitoring",
        "S3": "Critical metabolic emergency - immediate correction needed",
        "escalation": "Metabolic markers abnormal: {drivers} ({velocity}/hr)",
        "velocity": "Rapid metabolic change ({velocity}/hr)",
        "novelty": "New metabolic abnormality: {new_markers}"
    },
    "hematologic": {
        "S1": "Hematologic abnormality noted - monitor counts",
        "S2": "Significant cytopenias/coagulopathy - hematology input",
        "S3": "Critical hematologic emergency - immediate intervention",
        "escalation": "Blood counts/coagulation worsening: {drivers}",
        "velocity": "Rapid hematologic change ({velocity}/hr)",
        "novelty": "New hematologic marker: {new_markers}"
    },
    "oncology_inception": {
        "S1": "Tumor marker elevation - further workup recommended",
        "S2": "Pattern concerning for malignancy - expedite evaluation",
        "S3": "High-probability malignancy - urgent oncology referral",
        "escalation": "Tumor markers rising: {drivers} ({velocity}/hr)",
        "velocity": "Rapid tumor marker rise ({velocity}/hr)",
        "novelty": "New tumor marker elevated: {new_markers}"
    },
    "multi_system": {
        "S1": "Multi-organ stress detected - comprehensive assessment needed",
        "S2": "Multi-organ dysfunction developing - ICU evaluation",
        "S3": "Multi-organ failure - immediate critical care intervention",
        "escalation": "Multiple systems declining: {drivers}",
        "velocity": "Rapid multi-system deterioration ({velocity}/hr)",
        "novelty": "Additional organ system involved: {new_markers}"
    },
    # Default for unknown domains
    "default": {
        "S1": "Risk indicators elevated - enhanced monitoring recommended",
        "S2": "Risk score escalating - clinical evaluation advised",
        "S3": "Critical risk level - immediate evaluation required",
        "escalation": "Risk markers worsening: {drivers} ({velocity}/hr)",
        "velocity": "Rapid risk increase ({velocity}/hr)",
        "novelty": "New risk factor identified: {new_markers}"
    }
}


def format_biomarker_list(biomarkers: List[str], max_display: int = 3) -> str:
    """Format biomarker list for clinical display."""
    display_names = []
    for marker in biomarkers[:max_display]:
        normalized = marker.lower().strip().replace("-", "_").replace(" ", "_")
        display_name = BIOMARKER_DISPLAY_NAMES.get(normalized, marker)
        display_names.append(display_name)

    if len(biomarkers) > max_display:
        display_names.append(f"+{len(biomarkers) - max_display} more")

    return ", ".join(display_names)


def generate_clinical_rationale(
    domain: str,
    state: ClinicalState,
    velocity: float,
    contributing_biomarkers: List[str],
    is_escalation: bool = False,
    is_velocity_trigger: bool = False,
    is_novelty_trigger: bool = False,
    new_markers: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Generate clinical-language rationale and headline for alert.

    Returns:
        {
            "headline": "Short clinical summary for display",
            "rationale": "Detailed explanation for clinicians",
            "suggested_action": "Recommended clinical action"
        }
    """
    templates = RATIONALE_TEMPLATES.get(domain, RATIONALE_TEMPLATES["default"])
    drivers_str = format_biomarker_list(contributing_biomarkers)
    velocity_str = f"{velocity:+.2f}"

    # Determine headline based on trigger type
    if is_escalation:
        headline = templates["escalation"].format(
            drivers=drivers_str,
            velocity=velocity_str
        )
    elif is_velocity_trigger:
        headline = templates["velocity"].format(velocity=velocity_str)
    elif is_novelty_trigger and new_markers:
        new_markers_str = format_biomarker_list(new_markers)
        headline = templates["novelty"].format(new_markers=new_markers_str)
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
        f"Primary drivers: {drivers_str}." if contributing_biomarkers else "",
        f"Score velocity: {velocity_str}/hour." if abs(velocity) > 0.01 else ""
    ]

    if is_novelty_trigger and new_markers:
        new_markers_str = format_biomarker_list(new_markers)
        rationale_parts.append(f"New markers detected: {new_markers_str}.")

    rationale = " ".join(p for p in rationale_parts if p)

    # Suggested action based on state
    action_map = {
        "S0": "Continue routine monitoring.",
        "S1": "Increase monitoring frequency. Review recent labs and vitals.",
        "S2": "Evaluate patient bedside. Consider specialist consultation.",
        "S3": "Immediate bedside evaluation. Activate rapid response if appropriate."
    }
    suggested_action = action_map.get(state.value, "Review patient status.")

    return {
        "headline": headline,
        "rationale": rationale,
        "suggested_action": suggested_action
    }


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class PatientState:
    """Persisted state for a patient + risk domain."""
    patient_id: str
    risk_domain: str
    current_state: ClinicalState
    risk_score: float
    episode_id: str
    episode_start: datetime
    last_alert_time: Optional[datetime]
    last_score_time: datetime
    last_scores: List[Tuple[datetime, float]] = field(default_factory=list)
    contributing_biomarkers: List[str] = field(default_factory=list)
    alert_count_in_episode: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "risk_domain": self.risk_domain,
            "current_state": self.current_state.value,
            "risk_score": self.risk_score,
            "episode_id": self.episode_id,
            "episode_start": self.episode_start.isoformat(),
            "last_alert_time": self.last_alert_time.isoformat() if self.last_alert_time else None,
            "last_score_time": self.last_score_time.isoformat(),
            "contributing_biomarkers": self.contributing_biomarkers,
            "alert_count_in_episode": self.alert_count_in_episode
        }


@dataclass
class AlertEvent:
    """Alert event record for audit logging."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    patient_id: str
    risk_domain: str
    episode_id: str
    state_from: Optional[ClinicalState]
    state_to: ClinicalState
    risk_score: float
    contributing_biomarkers: List[str]
    velocity: float
    severity: AlertSeverity
    rationale: str
    suggested_cooldown_minutes: int
    suppression_reason: Optional[str] = None
    last_alert_timestamp: Optional[datetime] = None
    time_since_last_alert_minutes: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "patient_id": self.patient_id,
            "risk_domain": self.risk_domain,
            "episode_id": self.episode_id,
            "state_from": self.state_from.value if self.state_from else None,
            "state_to": self.state_to.value,
            "risk_score": round(self.risk_score, 4),
            "contributing_biomarkers": self.contributing_biomarkers,
            "velocity": round(self.velocity, 4),
            "severity": self.severity.value,
            "rationale": self.rationale,
            "suggested_cooldown_minutes": self.suggested_cooldown_minutes,
            "suppression_reason": self.suppression_reason,
            "last_alert_timestamp": self.last_alert_timestamp.isoformat() if self.last_alert_timestamp else None,
            "time_since_last_alert_minutes": round(self.time_since_last_alert_minutes, 1) if self.time_since_last_alert_minutes else None
        }


# =============================================================================
# STORAGE (IN-MEMORY - REPLACE WITH PERSISTENT STORAGE IN PRODUCTION)
# =============================================================================

class StateStorage:
    """In-memory storage for patient states and alert logs.

    In production, replace with:
    - Redis for state (fast lookup, TTL support)
    - PostgreSQL/TimescaleDB for audit logs
    - Firebase/DynamoDB for serverless deployments
    """

    def __init__(self):
        # patient_id:risk_domain -> PatientState
        self._states: Dict[str, PatientState] = {}
        # Ordered list of all events
        self._audit_log: List[AlertEvent] = []

    def _key(self, patient_id: str, risk_domain: str) -> str:
        return f"{patient_id}:{risk_domain}"

    def get_state(self, patient_id: str, risk_domain: str) -> Optional[PatientState]:
        """Retrieve last known state for patient + domain."""
        return self._states.get(self._key(patient_id, risk_domain))

    def save_state(self, state: PatientState) -> None:
        """Persist patient state."""
        self._states[self._key(state.patient_id, state.risk_domain)] = state

    def log_event(self, event: AlertEvent) -> None:
        """Append event to audit log."""
        self._audit_log.append(event)

    def get_events(
        self,
        patient_id: Optional[str] = None,
        risk_domain: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AlertEvent]:
        """Query audit log with filters."""
        results = self._audit_log

        if patient_id:
            results = [e for e in results if e.patient_id == patient_id]
        if risk_domain:
            results = [e for e in results if e.risk_domain == risk_domain]
        if since:
            results = [e for e in results if e.timestamp >= since]

        return results[-limit:]

    def clear_patient(self, patient_id: str) -> None:
        """Clear all state for a patient (for testing)."""
        keys_to_remove = [k for k in self._states if k.startswith(f"{patient_id}:")]
        for k in keys_to_remove:
            del self._states[k]


# Global storage instance
_storage = StateStorage()


def get_storage() -> StateStorage:
    """Get the global storage instance."""
    return _storage


# =============================================================================
# CLINICAL STATE ENGINE
# =============================================================================

class ClinicalStateEngine:
    """
    Core alert evaluation engine implementing ATC v1.

    Responsibilities:
    - Map risk scores to clinical states
    - Detect state transitions
    - Apply dedupe/cooldown logic
    - Generate alert events
    - Maintain audit trail
    """

    def __init__(self, config: ATCConfig = None, storage: StateStorage = None):
        self.config = config or DEFAULT_CONFIG
        self.storage = storage or get_storage()

    def map_score_to_state(self, risk_score: float) -> ClinicalState:
        """Map a risk score to clinical state using configured thresholds."""
        if risk_score < self.config.s0_upper:
            return ClinicalState.S0_STABLE
        elif risk_score < self.config.s1_upper:
            return ClinicalState.S1_WATCH
        elif risk_score < self.config.s2_upper:
            return ClinicalState.S2_ESCALATING
        else:
            return ClinicalState.S3_CRITICAL

    def state_to_severity(self, state: ClinicalState) -> AlertSeverity:
        """Map clinical state to alert severity."""
        return {
            ClinicalState.S0_STABLE: AlertSeverity.INFO,
            ClinicalState.S1_WATCH: AlertSeverity.WARNING,
            ClinicalState.S2_ESCALATING: AlertSeverity.URGENT,
            ClinicalState.S3_CRITICAL: AlertSeverity.CRITICAL
        }[state]

    def calculate_velocity(
        self,
        current_score: float,
        current_time: datetime,
        last_scores: List[Tuple[datetime, float]]
    ) -> float:
        """Calculate score velocity (change per hour)."""
        if not last_scores:
            return 0.0

        # Find scores within velocity window
        window_start = current_time - timedelta(hours=self.config.velocity_window_hours)
        recent_scores = [(t, s) for t, s in last_scores if t >= window_start]

        if not recent_scores:
            # Use most recent score regardless of time
            last_time, last_score = last_scores[-1]
            hours_elapsed = (current_time - last_time).total_seconds() / 3600.0
            if hours_elapsed > 0:
                return (current_score - last_score) / hours_elapsed
            return 0.0

        # Use earliest score in window
        earliest_time, earliest_score = recent_scores[0]
        hours_elapsed = (current_time - earliest_time).total_seconds() / 3600.0

        if hours_elapsed > 0:
            return (current_score - earliest_score) / hours_elapsed
        return 0.0

    def detect_novelty(
        self,
        current_biomarkers: List[str],
        previous_biomarkers: List[str]
    ) -> Tuple[bool, List[str]]:
        """Detect if new biomarkers have entered the contributing set."""
        current_set = set(current_biomarkers[:3])  # Top 3
        previous_set = set(previous_biomarkers[:3])
        new_markers = current_set - previous_set

        return len(new_markers) > 0, list(new_markers)

    def is_escalation(
        self,
        from_state: Optional[ClinicalState],
        to_state: ClinicalState
    ) -> bool:
        """Check if transition represents escalation."""
        if from_state is None:
            return to_state.severity_level >= ClinicalState.S1_WATCH.severity_level
        return to_state.severity_level > from_state.severity_level

    def should_fire_alert(
        self,
        from_state: Optional[ClinicalState],
        to_state: ClinicalState,
        velocity: float,
        novelty_detected: bool,
        last_alert_time: Optional[datetime],
        current_time: datetime
    ) -> Tuple[bool, str]:
        """
        Determine if alert should fire based on ATC rules.

        Returns: (should_fire, rationale)
        """
        # First alert for this patient/domain
        if from_state is None:
            if to_state.severity_level >= ClinicalState.S1_WATCH.severity_level:
                return True, f"Initial assessment at {to_state.display_name} state"
            return False, "Initial assessment at Stable - no alert needed"

        # State escalation always fires
        if self.is_escalation(from_state, to_state):
            return True, f"State escalation from {from_state.display_name} to {to_state.display_name}"

        # De-escalation - log but don't alert
        if to_state.severity_level < from_state.severity_level:
            return False, f"De-escalation from {from_state.display_name} to {to_state.display_name}"

        # Same state - check override conditions
        if from_state == to_state:
            # Velocity spike override
            if self.config.velocity_override_enabled and abs(velocity) > self.config.velocity_threshold:
                return True, f"Velocity spike ({velocity:.3f}/hr) exceeds threshold in {to_state.display_name} state"

            # Novelty override
            if self.config.novelty_detection_enabled and novelty_detected:
                return True, f"New biomarker pattern detected in {to_state.display_name} state"

            # Cooldown check
            if last_alert_time:
                minutes_since = (current_time - last_alert_time).total_seconds() / 60.0
                cooldown = self._get_cooldown_for_state(to_state)

                if minutes_since < cooldown:
                    return False, f"Within cooldown ({minutes_since:.1f}/{cooldown} min) for {to_state.display_name} state"

            # Outside cooldown, re-alert for non-stable states
            if to_state != ClinicalState.S0_STABLE:
                return True, f"Cooldown expired, re-alerting for {to_state.display_name} state"

        return False, "No alert condition met"

    def _get_cooldown_for_state(self, state: ClinicalState) -> int:
        """Get appropriate cooldown for state severity."""
        if state == ClinicalState.S3_CRITICAL:
            return self.config.critical_cooldown_minutes
        elif state == ClinicalState.S2_ESCALATING:
            return self.config.escalation_cooldown_minutes
        return self.config.default_cooldown_minutes

    def evaluate(
        self,
        patient_id: str,
        timestamp: datetime,
        risk_domain: str,
        current_scores: Dict[str, float],
        contributing_biomarkers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main evaluation entry point.

        Args:
            patient_id: Unique patient identifier
            timestamp: Observation timestamp
            risk_domain: Risk category (e.g., "sepsis", "cardiac", "respiratory")
            current_scores: Dict of score_name -> value (uses max or weighted avg)
            contributing_biomarkers: Top biomarkers driving the score

        Returns:
            Evaluation result with state, transition info, and any fired alert
        """
        # Calculate aggregate score (max strategy)
        if not current_scores:
            risk_score = 0.0
        else:
            risk_score = max(current_scores.values())

        contributing_biomarkers = contributing_biomarkers or []

        # Get last known state
        last_state = self.storage.get_state(patient_id, risk_domain)

        # Map to new state
        new_state = self.map_score_to_state(risk_score)

        # Calculate velocity
        last_scores = last_state.last_scores if last_state else []
        velocity = self.calculate_velocity(risk_score, timestamp, last_scores)

        # Detect novelty
        previous_biomarkers = last_state.contributing_biomarkers if last_state else []
        novelty_detected, new_markers = self.detect_novelty(
            contributing_biomarkers, previous_biomarkers
        )

        # Determine episode
        if last_state:
            # Check if we need new episode
            hours_in_stable = 0
            if last_state.current_state == ClinicalState.S0_STABLE:
                hours_in_stable = (timestamp - last_state.last_score_time).total_seconds() / 3600.0

            episode_duration = (timestamp - last_state.episode_start).total_seconds() / 3600.0

            if (hours_in_stable > self.config.episode_break_hours or
                    episode_duration > self.config.max_episode_duration_hours):
                # Start new episode
                episode_id = str(uuid.uuid4())[:8]
                episode_start = timestamp
                alert_count = 0
            else:
                episode_id = last_state.episode_id
                episode_start = last_state.episode_start
                alert_count = last_state.alert_count_in_episode
        else:
            # New patient/domain
            episode_id = str(uuid.uuid4())[:8]
            episode_start = timestamp
            alert_count = 0

        # Determine if alert should fire
        from_state = last_state.current_state if last_state else None
        last_alert_time = last_state.last_alert_time if last_state else None

        should_fire, rationale = self.should_fire_alert(
            from_state=from_state,
            to_state=new_state,
            velocity=velocity,
            novelty_detected=novelty_detected,
            last_alert_time=last_alert_time,
            current_time=timestamp
        )

        # State transition flag
        state_transition = from_state != new_state if from_state else True

        # Build alert event
        event_id = str(uuid.uuid4())
        severity = self.state_to_severity(new_state)
        cooldown = self._get_cooldown_for_state(new_state)

        if should_fire:
            alert_count += 1
            event = AlertEvent(
                event_id=event_id,
                event_type=EventType.ALERT_FIRED,
                timestamp=timestamp,
                patient_id=patient_id,
                risk_domain=risk_domain,
                episode_id=episode_id,
                state_from=from_state,
                state_to=new_state,
                risk_score=risk_score,
                contributing_biomarkers=contributing_biomarkers,
                velocity=velocity,
                severity=severity,
                rationale=rationale,
                suggested_cooldown_minutes=cooldown
            )
            new_alert_time = timestamp
        else:
            # Log suppressed
            minutes_since = None
            if last_alert_time:
                minutes_since = (timestamp - last_alert_time).total_seconds() / 60.0

            event = AlertEvent(
                event_id=event_id,
                event_type=EventType.ALERT_SUPPRESSED,
                timestamp=timestamp,
                patient_id=patient_id,
                risk_domain=risk_domain,
                episode_id=episode_id,
                state_from=from_state,
                state_to=new_state,
                risk_score=risk_score,
                contributing_biomarkers=contributing_biomarkers,
                velocity=velocity,
                severity=severity,
                rationale=rationale,
                suggested_cooldown_minutes=cooldown,
                suppression_reason=self._extract_suppression_reason(rationale),
                last_alert_timestamp=last_alert_time,
                time_since_last_alert_minutes=minutes_since
            )
            new_alert_time = last_alert_time

        # Log event
        self.storage.log_event(event)

        # Update score history (keep last 24 hours)
        new_scores = [(timestamp, risk_score)]
        if last_state:
            cutoff = timestamp - timedelta(hours=24)
            new_scores = [(t, s) for t, s in last_state.last_scores if t >= cutoff]
            new_scores.append((timestamp, risk_score))

        # Save updated state
        updated_state = PatientState(
            patient_id=patient_id,
            risk_domain=risk_domain,
            current_state=new_state,
            risk_score=risk_score,
            episode_id=episode_id,
            episode_start=episode_start,
            last_alert_time=new_alert_time,
            last_score_time=timestamp,
            last_scores=new_scores,
            contributing_biomarkers=contributing_biomarkers,
            alert_count_in_episode=alert_count
        )
        self.storage.save_state(updated_state)

        # Generate clinical rationale
        is_escalation = self.is_escalation(from_state, new_state)
        is_velocity_trigger = abs(velocity) > self.config.velocity_threshold
        is_novelty_trigger = novelty_detected

        clinical_rationale = generate_clinical_rationale(
            domain=risk_domain,
            state=new_state,
            velocity=velocity,
            contributing_biomarkers=contributing_biomarkers,
            is_escalation=is_escalation,
            is_velocity_trigger=is_velocity_trigger,
            is_novelty_trigger=is_novelty_trigger,
            new_markers=new_markers if novelty_detected else None
        )

        # Domain discovery (if available)
        domain_discovery = None
        if DOMAIN_CLASSIFIER_AVAILABLE and classify_domains and contributing_biomarkers:
            try:
                domains = classify_domains(contributing_biomarkers)
                if domains:
                    domain_discovery = {
                        "detected_domains": domains,
                        "primary_domain": domains[0]["domain"] if domains else risk_domain,
                        "confidence": domains[0]["confidence"] if domains else 0.0
                    }
            except Exception:
                pass  # Domain discovery is optional

        # Build response
        result = {
            "patient_id": patient_id,
            "risk_domain": risk_domain,
            "timestamp": timestamp.isoformat(),
            "risk_score": round(risk_score, 4),
            "state_now": new_state.value,
            "state_name": new_state.display_name,
            "state_transition": state_transition,
            "state_from": from_state.value if from_state else None,
            "velocity": round(velocity, 4),
            "novelty_detected": novelty_detected,
            "new_biomarkers": new_markers if novelty_detected else [],
            "episode_id": episode_id,
            "severity": severity.value,
            "rationale": rationale,
            "suggested_cooldown_minutes": cooldown,
            # Enhanced fields
            "clinical_headline": clinical_rationale["headline"],
            "clinical_rationale": clinical_rationale["rationale"],
            "suggested_action": clinical_rationale["suggested_action"],
            "contributing_biomarkers": contributing_biomarkers,
            "domain_config_used": {
                "thresholds": {
                    "s0_upper": self.config.s0_upper,
                    "s1_upper": self.config.s1_upper,
                    "s2_upper": self.config.s2_upper
                },
                "cooldowns": {
                    "default": self.config.default_cooldown_minutes,
                    "escalation": self.config.escalation_cooldown_minutes,
                    "critical": self.config.critical_cooldown_minutes
                },
                "velocity_threshold": self.config.velocity_threshold
            }
        }

        # Add domain discovery if available
        if domain_discovery:
            result["domain_discovery"] = domain_discovery

        if should_fire:
            result["alert_event"] = event.to_dict()
        else:
            result["alert_event"] = None
            result["suppressed_event"] = event.to_dict()

        return result

    def _extract_suppression_reason(self, rationale: str) -> str:
        """Extract suppression reason category from rationale."""
        rationale_lower = rationale.lower()
        if "cooldown" in rationale_lower:
            return "cooldown"
        if "de-escalation" in rationale_lower:
            return "de-escalation"
        if "same" in rationale_lower or "no change" in rationale_lower:
            return "same_state"
        return "other"


# =============================================================================
# API HELPERS
# =============================================================================

def evaluate_patient_alert(
    patient_id: str,
    timestamp: str,
    risk_domain: str,
    current_scores: Dict[str, float],
    contributing_biomarkers: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    feature_values: Optional[Dict[str, float]] = None,
    auto_discover_domain: bool = False
) -> Dict[str, Any]:
    """
    Convenience function for API endpoint.

    Parses timestamp string and delegates to engine.

    Args:
        patient_id: Unique patient identifier
        timestamp: ISO8601 timestamp string
        risk_domain: Risk category (e.g., "sepsis", "cardiac") or "auto" for auto-discovery
        current_scores: Dict of score_name -> value
        contributing_biomarkers: Top biomarkers driving the score
        config: Optional custom configuration override
        feature_values: Optional feature values for domain discovery
        auto_discover_domain: If True or risk_domain is "auto", auto-classify domain

    Returns:
        Evaluation result with enhanced schema
    """
    # Parse timestamp
    if isinstance(timestamp, str):
        try:
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            ts = datetime.now(timezone.utc)
    else:
        ts = timestamp

    # Auto-discover domain if requested
    effective_domain = risk_domain
    domain_auto_discovered = False

    if (auto_discover_domain or risk_domain == "auto") and DOMAIN_CLASSIFIER_AVAILABLE and get_primary_domain:
        if contributing_biomarkers:
            try:
                discovered_domain, confidence, drivers = get_primary_domain(
                    contributing_biomarkers,
                    feature_values or {}
                )
                if confidence > 0.3:
                    effective_domain = discovered_domain
                    domain_auto_discovered = True
            except Exception:
                pass  # Fall back to provided domain

    # Use domain-specific config if no explicit config provided
    if config:
        engine_config = ATCConfig(**{
            k: v for k, v in config.items()
            if hasattr(DEFAULT_CONFIG, k)
        })
    else:
        # Try domain-specific config
        engine_config = get_domain_config(effective_domain)

    engine = ClinicalStateEngine(config=engine_config)
    result = engine.evaluate(
        patient_id=patient_id,
        timestamp=ts,
        risk_domain=effective_domain,
        current_scores=current_scores,
        contributing_biomarkers=contributing_biomarkers
    )

    # Add auto-discovery metadata if used
    if domain_auto_discovered:
        result["domain_auto_discovered"] = True
        result["original_risk_domain"] = risk_domain

    return result


def get_patient_state(patient_id: str, risk_domain: str) -> Optional[Dict[str, Any]]:
    """Get current state for a patient + domain."""
    state = get_storage().get_state(patient_id, risk_domain)
    return state.to_dict() if state else None


def get_alert_history(
    patient_id: Optional[str] = None,
    risk_domain: Optional[str] = None,
    since_hours: float = 24,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Get alert history with filters."""
    since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    events = get_storage().get_events(
        patient_id=patient_id,
        risk_domain=risk_domain,
        since=since,
        limit=limit
    )
    return [e.to_dict() for e in events]


def get_atc_config() -> Dict[str, Any]:
    """Get current ATC configuration."""
    return DEFAULT_CONFIG.to_dict()


def get_all_domain_configs() -> Dict[str, Dict[str, Any]]:
    """Get all domain-specific configurations."""
    return {domain: config.to_dict() for domain, config in DOMAIN_CONFIGS.items()}


def get_domain_config_for_api(domain: str) -> Dict[str, Any]:
    """Get configuration for a specific domain."""
    config = get_domain_config(domain)
    return {
        "domain": domain,
        "config": config.to_dict(),
        "is_default": config == DEFAULT_CONFIG
    }
