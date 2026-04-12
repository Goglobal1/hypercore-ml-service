"""
Evolution System Schemas - HyperCore
====================================

CENTRAL NERVOUS SYSTEM for DiviScan.

Every agent (3 active, scaling to 96+) uses these schemas to:
1. EMIT evolution signals after operations
2. RECEIVE parameter updates from the Evolution Controller
3. LEARN continuously from outcomes

Core patterns adapted from GAIR-NLP ASI-Evolve with healthcare additions:
- Capability tiers (1-9) for graduated autonomy
- Three-lane architecture (Production/Shadow/Promotion)
- FDA-ready audit trails
- Utility Gate integration

Reference: HyperCore Implementation Guide - Appendix F
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Union
import uuid


# =============================================================================
# ENUMS
# =============================================================================

class Lane(str, Enum):
    """Three-lane architecture lanes."""
    PRODUCTION = "production"  # Validated, human-approved capabilities
    SHADOW = "shadow"          # AI-generated candidates under evaluation
    PROMOTION = "promotion"    # Queued for human review and approval


class CapabilityTier(IntEnum):
    """
    Capability tiers for graduated autonomy (1-9 scale).

    Lower tiers = more constrained, higher confidence required
    Higher tiers = more autonomy, used for well-validated capabilities

    Tiers 7-9 require explicit human approval for promotion.
    """
    TIER_1 = 1  # Minimal autonomy - every action requires approval
    TIER_2 = 2  # Basic - can suggest, human must approve
    TIER_3 = 3  # Guided - can act with immediate human oversight
    TIER_4 = 4  # Supervised - periodic human review
    TIER_5 = 5  # Standard - normal operational autonomy
    TIER_6 = 6  # Elevated - expanded action space
    TIER_7 = 7  # Advanced - significant autonomy (requires approval)
    TIER_8 = 8  # Expert - near-full autonomy (requires approval)
    TIER_9 = 9  # Maximum - full autonomy (requires approval + escalation)

    @classmethod
    def requires_escalation(cls, tier: int) -> bool:
        """Check if tier requires escalation for promotion."""
        return tier >= cls.TIER_7

    @classmethod
    def from_score(cls, utility_score: float, confidence: float) -> "CapabilityTier":
        """
        Derive tier from utility score and confidence.

        Args:
            utility_score: Combined utility score (0-1)
            confidence: Confidence level (0-1)

        Returns:
            Appropriate CapabilityTier
        """
        combined = (utility_score * 0.6) + (confidence * 0.4)

        if combined >= 0.95:
            return cls.TIER_9
        elif combined >= 0.90:
            return cls.TIER_8
        elif combined >= 0.85:
            return cls.TIER_7
        elif combined >= 0.75:
            return cls.TIER_6
        elif combined >= 0.65:
            return cls.TIER_5
        elif combined >= 0.55:
            return cls.TIER_4
        elif combined >= 0.45:
            return cls.TIER_3
        elif combined >= 0.35:
            return cls.TIER_2
        else:
            return cls.TIER_1


class ApprovalStatus(str, Enum):
    """Status of evolution node in approval workflow."""
    DRAFT = "draft"                    # Initial creation in shadow
    EVALUATING = "evaluating"          # Under automated evaluation
    PENDING_REVIEW = "pending_review"  # Awaiting human review
    UNDER_REVIEW = "under_review"      # Human is actively reviewing
    APPROVED = "approved"              # Approved for promotion
    REJECTED = "rejected"              # Rejected, will not promote
    PROMOTED = "promoted"              # Successfully promoted to production
    DEPRECATED = "deprecated"          # Superseded by newer version
    ROLLED_BACK = "rolled_back"        # Removed from production


class EscalationLevel(str, Enum):
    """Escalation levels for high-impact decisions."""
    NONE = "none"              # No escalation needed
    SOFT = "soft"              # Flagged for attention, not blocking
    HARD = "hard"              # Requires human approval before proceeding
    CRITICAL = "critical"      # Requires senior/multi-person approval


class EvolutionNodeType(str, Enum):
    """Types of evolution nodes."""
    HYPOTHESIS = "hypothesis"      # New approach/algorithm
    REFINEMENT = "refinement"      # Improvement to existing node
    COMBINATION = "combination"    # Merger of multiple parent nodes
    ROLLBACK = "rollback"          # Reversion to previous version
    BASELINE = "baseline"          # Initial seed/baseline capability


class DeploymentDomain(str, Enum):
    """Deployment domains with different safety requirements."""
    PHARMA = "pharma"              # Pharmaceutical/drug discovery
    CLINICAL = "clinical"          # Clinical decision support
    RESEARCH = "research"          # Research/academic use
    ADMINISTRATIVE = "administrative"  # Non-clinical operations


class SignalType(str, Enum):
    """Types of evolution signals agents can emit."""
    # Prediction/Decision signals
    PREDICTION = "prediction"          # Agent made a prediction
    DECISION = "decision"              # Agent made a decision
    CLASSIFICATION = "classification"  # Agent classified something

    # Outcome signals (feedback loop)
    OUTCOME_OBSERVED = "outcome_observed"  # Ground truth received
    OUTCOME_PARTIAL = "outcome_partial"    # Partial feedback received

    # Performance signals
    LATENCY = "latency"                # Response time measurement
    ERROR = "error"                    # Error occurred
    RESOURCE_USAGE = "resource_usage"  # CPU/memory/GPU usage

    # Quality signals
    CALIBRATION = "calibration"        # Calibration measurement
    CONFIDENCE = "confidence"          # Confidence distribution
    UNCERTAINTY = "uncertainty"        # Uncertainty quantification

    # Safety signals
    SAFETY_CHECK = "safety_check"      # Safety evaluation result
    ESCALATION = "escalation"          # Escalation triggered
    ANOMALY = "anomaly"                # Anomaly detected

    # Experiment signals
    EXPERIMENT_START = "experiment_start"    # Shadow experiment started
    EXPERIMENT_END = "experiment_end"        # Shadow experiment completed
    HYPOTHESIS_GENERATED = "hypothesis_generated"  # New hypothesis created
    LESSON_EXTRACTED = "lesson_extracted"    # Lesson extracted from analysis


class ParameterType(str, Enum):
    """Types of parameter updates the controller can send."""
    THRESHOLD = "threshold"            # Decision thresholds
    WEIGHT = "weight"                  # Model/ensemble weights
    POLICY = "policy"                  # Decision policies
    CONSTRAINT = "constraint"          # Operating constraints
    FEATURE = "feature"                # Feature flags/toggles
    HYPERPARAMETER = "hyperparameter"  # Algorithm hyperparameters


class CognitionItemType(str, Enum):
    """Types of items in the cognition store."""
    LESSON = "lesson"                  # Extracted lesson from evolution
    REGULATORY = "regulatory"          # Regulatory guidance (FDA, etc.)
    CLINICAL = "clinical"              # Clinical protocol/guideline
    SAFETY = "safety"                  # Safety constraint/rule
    METHODOLOGY = "methodology"        # Methodology/best practice
    EVIDENCE = "evidence"              # Clinical evidence/study
    PRIOR = "prior"                    # Prior knowledge/assumption


class SignalSeverity(str, Enum):
    """Severity levels for signals."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# SIGNAL/UPDATE STRUCTURES (Nervous System Core)
# =============================================================================

@dataclass
class EvolutionSignal:
    """
    Signal emitted by agents after operations.

    Every agent in the system emits these signals, which flow to the
    Evolution Controller for learning and adaptation. This is the
    afferent pathway of the nervous system.

    Usage:
        emitter = EvolutionEmitter(agent_id="diagnostic_engine")
        emitter.emit(SignalType.PREDICTION, {
            "prediction": 0.87,
            "confidence": 0.92,
            "features_used": ["age", "biomarker_x"],
        })
    """
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Source identification
    agent_id: str = ""                # e.g., "diagnostic_engine", "utility_gate"
    agent_version: str = "1.0.0"
    domain: DeploymentDomain = DeploymentDomain.RESEARCH

    # Signal content
    signal_type: SignalType = SignalType.PREDICTION
    payload: Dict[str, Any] = field(default_factory=dict)

    # Context
    session_id: Optional[str] = None  # Links related signals
    request_id: Optional[str] = None  # Original request ID
    patient_id: Optional[str] = None  # Anonymized patient reference

    # Outcome tracking (filled in later when outcome known)
    outcome: Optional[Dict[str, Any]] = None
    outcome_timestamp: Optional[str] = None

    # Performance metrics
    latency_ms: Optional[float] = None
    resource_usage: Optional[Dict[str, float]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_outcome(self, outcome: Dict[str, Any]) -> "EvolutionSignal":
        """Return a copy of this signal with outcome attached."""
        return EvolutionSignal(
            signal_id=self.signal_id,
            timestamp=self.timestamp,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            domain=self.domain,
            signal_type=self.signal_type,
            payload=self.payload,
            session_id=self.session_id,
            request_id=self.request_id,
            patient_id=self.patient_id,
            outcome=outcome,
            outcome_timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=self.latency_ms,
            resource_usage=self.resource_usage,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "domain": self.domain.value,
            "signal_type": self.signal_type.value,
            "payload": self.payload,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "patient_id": self.patient_id,
            "outcome": self.outcome,
            "outcome_timestamp": self.outcome_timestamp,
            "latency_ms": self.latency_ms,
            "resource_usage": self.resource_usage,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionSignal":
        return cls(
            signal_id=data.get("signal_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            agent_id=data.get("agent_id", ""),
            agent_version=data.get("agent_version", "1.0.0"),
            domain=DeploymentDomain(data.get("domain", "research")),
            signal_type=SignalType(data.get("signal_type", "prediction")),
            payload=data.get("payload", {}),
            session_id=data.get("session_id"),
            request_id=data.get("request_id"),
            patient_id=data.get("patient_id"),
            outcome=data.get("outcome"),
            outcome_timestamp=data.get("outcome_timestamp"),
            latency_ms=data.get("latency_ms"),
            resource_usage=data.get("resource_usage"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ParameterUpdate:
    """
    Parameter update sent from Evolution Controller to agents.

    This is the efferent pathway of the nervous system - how the
    controller adapts agent behavior based on learned signals.

    Usage:
        # Controller sends:
        update = ParameterUpdate(
            target_agent="diagnostic_engine",
            parameter_type=ParameterType.THRESHOLD,
            parameter_name="confidence_threshold",
            old_value=0.85,
            new_value=0.88,
            rationale="Calibration drift detected, raising threshold"
        )

        # Agent receives and applies:
        agent.apply_parameter_update(update)
    """
    update_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Target
    target_agent: str = ""            # Which agent receives this
    target_version: Optional[str] = None  # Specific version, or None for all

    # Parameter specification
    parameter_type: ParameterType = ParameterType.THRESHOLD
    parameter_name: str = ""          # e.g., "confidence_threshold"
    parameter_path: Optional[str] = None  # e.g., "model.layers.0.weight"

    # Values
    old_value: Any = None
    new_value: Any = None

    # Justification (for audit trail)
    rationale: str = ""
    evidence_signals: List[str] = field(default_factory=list)  # Signal IDs
    source_node_id: Optional[str] = None  # Evolution node that triggered this

    # Control
    lane: Lane = Lane.SHADOW          # Which lane this applies to
    effective_immediately: bool = True
    rollback_on_failure: bool = True

    # Approval (for production lane updates)
    requires_approval: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None

    # Status
    applied: bool = False
    applied_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_id": self.update_id,
            "timestamp": self.timestamp,
            "target_agent": self.target_agent,
            "target_version": self.target_version,
            "parameter_type": self.parameter_type.value,
            "parameter_name": self.parameter_name,
            "parameter_path": self.parameter_path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "rationale": self.rationale,
            "evidence_signals": self.evidence_signals,
            "source_node_id": self.source_node_id,
            "lane": self.lane.value,
            "effective_immediately": self.effective_immediately,
            "rollback_on_failure": self.rollback_on_failure,
            "requires_approval": self.requires_approval,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "applied": self.applied,
            "applied_at": self.applied_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterUpdate":
        return cls(
            update_id=data.get("update_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            target_agent=data.get("target_agent", ""),
            target_version=data.get("target_version"),
            parameter_type=ParameterType(data.get("parameter_type", "threshold")),
            parameter_name=data.get("parameter_name", ""),
            parameter_path=data.get("parameter_path"),
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            rationale=data.get("rationale", ""),
            evidence_signals=data.get("evidence_signals", []),
            source_node_id=data.get("source_node_id"),
            lane=Lane(data.get("lane", "shadow")),
            effective_immediately=data.get("effective_immediately", True),
            rollback_on_failure=data.get("rollback_on_failure", True),
            requires_approval=data.get("requires_approval", False),
            approved_by=data.get("approved_by"),
            approved_at=data.get("approved_at"),
            applied=data.get("applied", False),
            applied_at=data.get("applied_at"),
            error=data.get("error"),
        )


@dataclass
class AgentRegistration:
    """
    Registration of an agent with the Evolution Controller.

    Every agent must register to participate in the evolution system.
    This tells the controller what parameters the agent exposes and
    what signals it can emit.
    """
    agent_id: str = ""
    agent_type: str = ""              # e.g., "diagnostic", "utility_gate", "trial_rescue"
    version: str = "1.0.0"

    # What this agent does
    description: str = ""
    domain: DeploymentDomain = DeploymentDomain.RESEARCH

    # What signals this agent emits
    emits_signals: List[str] = field(default_factory=list)  # SignalType values

    # What parameters can be updated
    configurable_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Format: {"param_name": {"type": "float", "min": 0, "max": 1, "default": 0.5}}

    # Current parameter values
    current_parameters: Dict[str, Any] = field(default_factory=dict)

    # Registration metadata
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_heartbeat: Optional[str] = None
    status: str = "active"  # active, inactive, degraded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "version": self.version,
            "description": self.description,
            "domain": self.domain.value,
            "emits_signals": self.emits_signals,
            "configurable_parameters": self.configurable_parameters,
            "current_parameters": self.current_parameters,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRegistration":
        return cls(
            agent_id=data.get("agent_id", ""),
            agent_type=data.get("agent_type", ""),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            domain=DeploymentDomain(data.get("domain", "research")),
            emits_signals=data.get("emits_signals", []),
            configurable_parameters=data.get("configurable_parameters", {}),
            current_parameters=data.get("current_parameters", {}),
            registered_at=data.get("registered_at", datetime.now(timezone.utc).isoformat()),
            last_heartbeat=data.get("last_heartbeat"),
            status=data.get("status", "active"),
        )


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class UtilityBreakdown:
    """Detailed breakdown of utility scoring from Utility Gate."""
    handler_score: float = 0.0
    net_utility: float = 0.0
    rightness: float = 0.0
    novelty: float = 0.0
    convincing: float = 0.0
    calibration: float = 0.0
    actionability: float = 0.0

    # Healthcare-specific scores
    safety_score: float = 0.0
    regulatory_compliance: float = 0.0
    clinical_validity: float = 0.0

    def combined_score(self) -> float:
        """Calculate weighted combined score."""
        return (
            self.handler_score * 0.15 +
            self.net_utility * 0.15 +
            self.safety_score * 0.25 +  # Safety weighted heavily
            self.clinical_validity * 0.20 +
            self.regulatory_compliance * 0.15 +
            self.calibration * 0.10
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "handler_score": self.handler_score,
            "net_utility": self.net_utility,
            "rightness": self.rightness,
            "novelty": self.novelty,
            "convincing": self.convincing,
            "calibration": self.calibration,
            "actionability": self.actionability,
            "safety_score": self.safety_score,
            "regulatory_compliance": self.regulatory_compliance,
            "clinical_validity": self.clinical_validity,
            "combined_score": self.combined_score(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UtilityBreakdown":
        return cls(
            handler_score=data.get("handler_score", 0.0),
            net_utility=data.get("net_utility", 0.0),
            rightness=data.get("rightness", 0.0),
            novelty=data.get("novelty", 0.0),
            convincing=data.get("convincing", 0.0),
            calibration=data.get("calibration", 0.0),
            actionability=data.get("actionability", 0.0),
            safety_score=data.get("safety_score", 0.0),
            regulatory_compliance=data.get("regulatory_compliance", 0.0),
            clinical_validity=data.get("clinical_validity", 0.0),
        )


@dataclass
class EvaluationResult:
    """Results from evaluating an evolution node."""
    success: bool
    eval_score: float = 0.0

    # Detailed metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Safety evaluation
    safety_passed: bool = True
    safety_warnings: List[str] = field(default_factory=list)
    safety_blockers: List[str] = field(default_factory=list)

    # Runtime info
    runtime_seconds: float = 0.0
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None

    # Validation results
    clinical_validation: Optional[Dict[str, Any]] = None
    regulatory_check: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "eval_score": self.eval_score,
            "metrics": self.metrics,
            "safety_passed": self.safety_passed,
            "safety_warnings": self.safety_warnings,
            "safety_blockers": self.safety_blockers,
            "runtime_seconds": self.runtime_seconds,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "clinical_validation": self.clinical_validation,
            "regulatory_check": self.regulatory_check,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        return cls(
            success=data.get("success", False),
            eval_score=data.get("eval_score", 0.0),
            metrics=data.get("metrics", {}),
            safety_passed=data.get("safety_passed", True),
            safety_warnings=data.get("safety_warnings", []),
            safety_blockers=data.get("safety_blockers", []),
            runtime_seconds=data.get("runtime_seconds", 0.0),
            error_message=data.get("error_message"),
            stack_trace=data.get("stack_trace"),
            clinical_validation=data.get("clinical_validation"),
            regulatory_check=data.get("regulatory_check"),
        )


@dataclass
class AuditEntry:
    """Immutable audit log entry for FDA compliance."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # What happened
    action: str = ""
    actor: str = ""  # "system", "ai:researcher", "human:user@email.com"

    # Context
    node_id: Optional[str] = None
    lane: Optional[str] = None

    # Details
    description: str = ""
    rationale: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Change tracking
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None

    # Integrity
    checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "action": self.action,
            "actor": self.actor,
            "node_id": self.node_id,
            "lane": self.lane,
            "description": self.description,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEntry":
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            action=data.get("action", ""),
            actor=data.get("actor", ""),
            node_id=data.get("node_id"),
            lane=data.get("lane"),
            description=data.get("description", ""),
            rationale=data.get("rationale", ""),
            evidence=data.get("evidence", {}),
            previous_state=data.get("previous_state"),
            new_state=data.get("new_state"),
            checksum=data.get("checksum"),
        )


@dataclass
class EvolutionNode:
    """
    Core evolution node - a single capability/hypothesis in the system.

    Adapts ASI-Evolve's Node with healthcare-specific additions:
    - Capability tier for graduated autonomy
    - Lane tracking for three-lane architecture
    - Approval status for promotion workflow
    - Audit trail for FDA compliance
    """

    # Identity
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: int = 1

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Lineage (from ASI-Evolve)
    parent_ids: List[str] = field(default_factory=list)
    node_type: EvolutionNodeType = EvolutionNodeType.HYPOTHESIS

    # Content (from ASI-Evolve)
    motivation: str = ""      # Why this approach was tried
    code: str = ""            # Generated code/algorithm
    analysis: str = ""        # Lessons learned from evaluation

    # Healthcare-specific classification
    domain: DeploymentDomain = DeploymentDomain.RESEARCH
    capability_tier: CapabilityTier = CapabilityTier.TIER_1

    # Three-lane tracking
    lane: Lane = Lane.SHADOW
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT

    # Scoring (enhanced from ASI-Evolve)
    score: float = 0.0                # Primary score for ranking
    utility_breakdown: Optional[UtilityBreakdown] = None
    confidence: float = 0.0           # Confidence in the score

    # Evaluation
    evaluation_result: Optional[EvaluationResult] = None
    visit_count: int = 0              # For UCB1 sampling (from ASI-Evolve)

    # Escalation
    escalation_level: EscalationLevel = EscalationLevel.NONE
    escalation_reason: Optional[str] = None

    # Approval workflow
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_notes: Optional[str] = None
    promoted_at: Optional[str] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Audit (references to AuditEntry IDs)
    audit_trail: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate and set derived fields."""
        if self.utility_breakdown:
            # Auto-calculate tier from utility if not explicitly set
            combined = self.utility_breakdown.combined_score()
            suggested_tier = CapabilityTier.from_score(combined, self.confidence)

            # Only upgrade tier, never downgrade automatically
            if suggested_tier > self.capability_tier:
                self.capability_tier = suggested_tier

        # Check escalation requirements
        if CapabilityTier.requires_escalation(self.capability_tier):
            if self.escalation_level == EscalationLevel.NONE:
                self.escalation_level = EscalationLevel.HARD
                self.escalation_reason = f"Tier {self.capability_tier} requires escalation"

    def get_context_text(self) -> str:
        """Return text for similarity search (from ASI-Evolve pattern)."""
        parts = [self.name, self.motivation, self.analysis]
        return " ".join(p for p in parts if p)

    def can_promote(self) -> tuple[bool, Optional[str]]:
        """Check if node can be promoted to production."""
        if self.lane == Lane.PRODUCTION:
            return False, "Already in production"

        if self.approval_status != ApprovalStatus.APPROVED:
            return False, f"Not approved (status: {self.approval_status.value})"

        if self.evaluation_result and not self.evaluation_result.safety_passed:
            return False, "Safety evaluation not passed"

        if self.evaluation_result and self.evaluation_result.safety_blockers:
            return False, f"Safety blockers: {', '.join(self.evaluation_result.safety_blockers)}"

        if self.escalation_level in (EscalationLevel.HARD, EscalationLevel.CRITICAL):
            if not self.reviewed_by:
                return False, "Requires human review for escalation"

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_ids": self.parent_ids,
            "node_type": self.node_type.value,
            "motivation": self.motivation,
            "code": self.code,
            "analysis": self.analysis,
            "domain": self.domain.value,
            "capability_tier": int(self.capability_tier),
            "lane": self.lane.value,
            "approval_status": self.approval_status.value,
            "score": self.score,
            "utility_breakdown": self.utility_breakdown.to_dict() if self.utility_breakdown else None,
            "confidence": self.confidence,
            "evaluation_result": self.evaluation_result.to_dict() if self.evaluation_result else None,
            "visit_count": self.visit_count,
            "escalation_level": self.escalation_level.value,
            "escalation_reason": self.escalation_reason,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "review_notes": self.review_notes,
            "promoted_at": self.promoted_at,
            "tags": self.tags,
            "metadata": self.metadata,
            "audit_trail": self.audit_trail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionNode":
        """Deserialize from dictionary."""
        utility_data = data.get("utility_breakdown")
        eval_data = data.get("evaluation_result")

        return cls(
            node_id=data.get("node_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            parent_ids=data.get("parent_ids", []),
            node_type=EvolutionNodeType(data.get("node_type", "hypothesis")),
            motivation=data.get("motivation", ""),
            code=data.get("code", ""),
            analysis=data.get("analysis", ""),
            domain=DeploymentDomain(data.get("domain", "research")),
            capability_tier=CapabilityTier(data.get("capability_tier", 1)),
            lane=Lane(data.get("lane", "shadow")),
            approval_status=ApprovalStatus(data.get("approval_status", "draft")),
            score=data.get("score", 0.0),
            utility_breakdown=UtilityBreakdown.from_dict(utility_data) if utility_data else None,
            confidence=data.get("confidence", 0.0),
            evaluation_result=EvaluationResult.from_dict(eval_data) if eval_data else None,
            visit_count=data.get("visit_count", 0),
            escalation_level=EscalationLevel(data.get("escalation_level", "none")),
            escalation_reason=data.get("escalation_reason"),
            reviewed_by=data.get("reviewed_by"),
            reviewed_at=data.get("reviewed_at"),
            review_notes=data.get("review_notes"),
            promoted_at=data.get("promoted_at"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            audit_trail=data.get("audit_trail", []),
        )


@dataclass
class CognitionItem:
    """
    Knowledge item for the cognition store.
    Stores domain knowledge, regulatory guidance, clinical protocols.

    From ASI-Evolve pattern with healthcare additions.
    """
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Content
    title: str = ""
    content: str = ""
    source: str = ""  # e.g., "FDA Guidance 2024", "Clinical Protocol XYZ"

    # Classification
    item_type: CognitionItemType = CognitionItemType.LESSON
    category: str = ""  # "regulatory", "clinical", "safety", "methodology"
    domain: DeploymentDomain = DeploymentDomain.RESEARCH

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Relevance tracking
    retrieval_count: int = 0
    last_retrieved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "item_type": self.item_type.value,
            "category": self.category,
            "domain": self.domain.value,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "retrieval_count": self.retrieval_count,
            "last_retrieved_at": self.last_retrieved_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitionItem":
        return cls(
            item_id=data.get("item_id", str(uuid.uuid4())),
            title=data.get("title", ""),
            content=data.get("content", ""),
            source=data.get("source", ""),
            item_type=CognitionItemType(data.get("item_type", "lesson")),
            category=data.get("category", ""),
            domain=DeploymentDomain(data.get("domain", "research")),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
            retrieval_count=data.get("retrieval_count", 0),
            last_retrieved_at=data.get("last_retrieved_at"),
        )


@dataclass
class PromotionRequest:
    """Request to promote a node from shadow to production."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_id: str = ""

    # Request details
    requested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    requested_by: str = ""  # "ai:researcher" or "human:user@email.com"

    # Justification
    rationale: str = ""
    evidence_summary: str = ""
    risk_assessment: str = ""

    # Review
    status: ApprovalStatus = ApprovalStatus.PENDING_REVIEW
    reviewer: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_decision: Optional[str] = None
    review_notes: Optional[str] = None

    # Escalation
    escalation_level: EscalationLevel = EscalationLevel.NONE
    escalation_approvers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "requested_at": self.requested_at,
            "requested_by": self.requested_by,
            "rationale": self.rationale,
            "evidence_summary": self.evidence_summary,
            "risk_assessment": self.risk_assessment,
            "status": self.status.value,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "review_decision": self.review_decision,
            "review_notes": self.review_notes,
            "escalation_level": self.escalation_level.value,
            "escalation_approvers": self.escalation_approvers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromotionRequest":
        return cls(
            request_id=data.get("request_id", str(uuid.uuid4())),
            node_id=data.get("node_id", ""),
            requested_at=data.get("requested_at", datetime.now(timezone.utc).isoformat()),
            requested_by=data.get("requested_by", ""),
            rationale=data.get("rationale", ""),
            evidence_summary=data.get("evidence_summary", ""),
            risk_assessment=data.get("risk_assessment", ""),
            status=ApprovalStatus(data.get("status", "pending_review")),
            reviewer=data.get("reviewer"),
            reviewed_at=data.get("reviewed_at"),
            review_decision=data.get("review_decision"),
            review_notes=data.get("review_notes"),
            escalation_level=EscalationLevel(data.get("escalation_level", "none")),
            escalation_approvers=data.get("escalation_approvers", []),
        )


# =============================================================================
# TYPE ALIASES
# =============================================================================

NodeId = str
ItemId = str
RequestId = str
EntryId = str
