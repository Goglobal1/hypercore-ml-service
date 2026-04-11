"""
Utility Engine - Handler/Feied Clinical Decision Support

This module implements the Utility Gate pattern for clinical decision support,
based on the Handler/Feied IEEE framework.

Core Components:
- UtilityGate: Final decision governance (surface/suppress/escalate)
- HandlerUtilityScorer: Three-condition test (rightness/novelty/convincing)
- Policy Registry: Mode-specific thresholds (hospital/pharma/government)

Usage:
    from app.core.utility_engine import UtilityGate, DeploymentMode, UtilityInput

    gate = UtilityGate(mode=DeploymentMode.HOSPITAL)
    decision = gate.evaluate(signal)

    if decision.should_surface:
        # Show to clinician
        pass
"""

from .schemas import (
    DeploymentMode,
    DecisionAction,
    EvidenceItem,
    UtilityInput,
    UtilityScoreBreakdown,
    UtilityDecision,
)

from .policy_registry import (
    UtilityPolicy,
    get_policy,
)

from .handler_utility import (
    HandlerUtilityScorer,
)

from .utility_gate import (
    UtilityGate,
)

__all__ = [
    # Schemas
    "DeploymentMode",
    "DecisionAction",
    "EvidenceItem",
    "UtilityInput",
    "UtilityScoreBreakdown",
    "UtilityDecision",
    # Policy
    "UtilityPolicy",
    "get_policy",
    # Handler
    "HandlerUtilityScorer",
    # Gate
    "UtilityGate",
]
