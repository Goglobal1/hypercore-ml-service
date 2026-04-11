"""
Utility Engine Schemas
Dataclasses and enums for the Utility Gate decision system.

Based on Handler/Feied IEEE framework for clinical decision support.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DeploymentMode(str, Enum):
    """Deployment context determines utility weights and thresholds."""
    HOSPITAL = "hospital"
    PHARMA = "pharma"
    GOVERNMENT = "government"


class DecisionAction(str, Enum):
    """Possible actions from the Utility Gate."""
    SURFACE = "surface"      # Show to user
    SUPPRESS = "suppress"    # Hide from user
    SNOOZE = "snooze"       # Defer for later
    ESCALATE = "escalate"   # Force immediate attention
    REVIEW = "review"       # Queue for manual review


@dataclass(slots=True)
class EvidenceItem:
    """Single piece of evidence supporting a clinical finding."""
    kind: str          # Type: "lab", "vital", "pattern", "genetic", "ml"
    label: str         # Human-readable description
    value: Any         # Actual value
    weight: float = 1.0  # Relative importance (0-1)


@dataclass(slots=True)
class UtilityInput:
    """Canonical object passed into the Utility Gate."""
    entity_id: str          # Unique identifier
    entity_type: str        # patient_alert | trial_opportunity | outbreak_signal
    mode: DeploymentMode    # Which deployment context
    title: str              # Short description
    summary: str            # Longer explanation

    # Core scores (all 0-1 scale)
    risk_probability: Optional[float] = None
    severity: Optional[float] = None
    calibration_score: Optional[float] = None
    ppv_estimate: Optional[float] = None

    # Utility-specific scores
    lead_time_hours: Optional[float] = None
    novelty_score: Optional[float] = None
    explainability_score: Optional[float] = None
    actionability_score: Optional[float] = None
    confidence_score: Optional[float] = None

    # Supporting data
    evidence: List[EvidenceItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UtilityScoreBreakdown:
    """Detailed breakdown of utility calculation."""
    rightness: float        # Is this correct? (PPV + calibration)
    novelty: float          # Is this new information?
    convincing: float       # Will user act on this?
    handler_score: float    # Combined weighted score
    net_utility: float      # Expected value calculation

    # Reasons for actions
    suppression_reasons: List[str] = field(default_factory=list)
    escalation_reasons: List[str] = field(default_factory=list)


@dataclass(slots=True)
class UtilityDecision:
    """Final decision from the Utility Gate."""
    action: DecisionAction
    should_surface: bool
    should_notify: bool
    should_escalate: bool
    snooze_hours: Optional[int]
    priority: str           # critical | high | medium | low | deferred | silent
    breakdown: UtilityScoreBreakdown
    metadata: Dict[str, Any] = field(default_factory=dict)
