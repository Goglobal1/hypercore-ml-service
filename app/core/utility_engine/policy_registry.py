"""
Utility Policy Registry
Mode-specific thresholds and weights for the Utility Gate.

Hospital, Pharma, and Government modes have different priorities:
- Hospital: Balance alert fatigue vs missed deterioration
- Pharma: Minimize missed safety signals (high FN cost)
- Government: Maximize early outbreak detection (very high FN cost)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .schemas import DeploymentMode


@dataclass(frozen=True, slots=True)
class UtilityPolicy:
    """Mode-specific policy configuration."""

    # Handler three-condition weights (must sum to 1.0)
    rightness_weight: float
    novelty_weight: float
    convincing_weight: float

    # Surfacing thresholds
    min_handler_score_surface: float
    min_handler_score_escalate: float
    min_novelty_for_surface: float
    min_convincing_for_surface: float
    min_ppv_for_surface: float

    # Snooze/escalation config
    snooze_duration_hours: int
    max_repeat_count_before_escalation: int

    # Utility weights for expected value calculation
    utility_weights: Dict[str, float]


def get_policy(mode: DeploymentMode) -> UtilityPolicy:
    """Get the policy configuration for a deployment mode."""

    if mode == DeploymentMode.HOSPITAL:
        return UtilityPolicy(
            # Hospital: Rightness matters most (avoid false alarms)
            rightness_weight=0.40,
            novelty_weight=0.35,
            convincing_weight=0.25,

            # Moderate thresholds
            min_handler_score_surface=0.58,
            min_handler_score_escalate=0.78,
            min_novelty_for_surface=0.35,
            min_convincing_for_surface=0.45,
            min_ppv_for_surface=0.18,

            # Allow snooze to reduce alert fatigue
            snooze_duration_hours=4,
            max_repeat_count_before_escalation=3,

            # Cost matrix: Missed deterioration is costly
            utility_weights={
                "true_positive_utility": 100.0,
                "false_positive_cost": -10.0,
                "false_negative_cost": -500.0,
                "true_negative_utility": 1.0,
            },
        )

    if mode == DeploymentMode.PHARMA:
        return UtilityPolicy(
            # Pharma: Convincing matters most (regulatory evidence)
            rightness_weight=0.35,
            novelty_weight=0.25,
            convincing_weight=0.40,

            # Higher thresholds for quality
            min_handler_score_surface=0.62,
            min_handler_score_escalate=0.82,
            min_novelty_for_surface=0.20,
            min_convincing_for_surface=0.55,
            min_ppv_for_surface=0.10,

            # No snooze for safety signals
            snooze_duration_hours=0,
            max_repeat_count_before_escalation=1,

            # Cost matrix: Missed adverse event is catastrophic
            utility_weights={
                "true_positive_utility": 1000.0,
                "false_positive_cost": -5.0,
                "false_negative_cost": -10000.0,
                "true_negative_utility": 1.0,
            },
        )

    # Government mode (default)
    return UtilityPolicy(
        # Government: Novelty matters most (early detection)
        rightness_weight=0.30,
        novelty_weight=0.40,
        convincing_weight=0.30,

        # Lower thresholds for sensitivity
        min_handler_score_surface=0.55,
        min_handler_score_escalate=0.75,
        min_novelty_for_surface=0.45,
        min_convincing_for_surface=0.40,
        min_ppv_for_surface=0.05,

        # No snooze for outbreak signals
        snooze_duration_hours=0,
        max_repeat_count_before_escalation=1,

        # Cost matrix: Missed outbreak is catastrophic
        utility_weights={
            "true_positive_utility": 10000.0,
            "false_positive_cost": -100.0,
            "false_negative_cost": -100000.0,
            "true_negative_utility": 1.0,
        },
    )
