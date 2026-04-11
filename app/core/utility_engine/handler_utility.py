"""
Handler Utility Scorer
Implements the Handler/Feied IEEE framework for clinical decision support.

The three-condition test:
1. RIGHTNESS: Is this prediction correct? (PPV + calibration)
2. NOVELTY: Is this new/useful information? (lead time + non-obvious)
3. CONVINCING: Will the clinician act on this? (evidence quality)

Handler Score = weighted combination of all three.
"""

from __future__ import annotations
from typing import Dict
from .policy_registry import UtilityPolicy
from .schemas import UtilityInput


class HandlerUtilityScorer:
    """
    Implements Handler/Feied u-metrics for clinical utility.

    This is NOT just AUC - it's clinical utility that accounts for:
    - Alert fatigue (false positive cost)
    - Missed events (false negative cost)
    - Time value of predictions (lead time)
    - Evidence quality (actionability)
    """

    def __init__(self, policy: UtilityPolicy):
        """
        Initialize scorer with mode-specific policy.

        Args:
            policy: UtilityPolicy with weights and thresholds
        """
        self.policy = policy

    def score_rightness(self, signal: UtilityInput) -> float:
        """
        Score the RIGHTNESS dimension (0-1).

        Components:
        - PPV estimate: How likely is this actually true?
        - Calibration: How well does confidence match reality?
        - Risk probability: Base likelihood of condition
        """
        components = []

        # PPV is most important
        if signal.ppv_estimate is not None:
            components.append(signal.ppv_estimate * 1.5)  # Weight 1.5x

        # Calibration score
        if signal.calibration_score is not None:
            components.append(signal.calibration_score)

        # Risk probability
        if signal.risk_probability is not None:
            components.append(signal.risk_probability * 0.8)

        # Confidence as fallback
        if signal.confidence_score is not None:
            components.append(signal.confidence_score * 0.7)

        if not components:
            return 0.5  # Default neutral

        # Weighted average, clamped to [0, 1]
        return min(1.0, max(0.0, sum(components) / len(components)))

    def score_novelty(self, signal: UtilityInput) -> float:
        """
        Score the NOVELTY dimension (0-1).

        Components:
        - Lead time: How far ahead does this predict?
        - Novelty score: Is this non-obvious?
        - Severity: More severe = more valuable to know early
        """
        components = []

        # Explicit novelty score
        if signal.novelty_score is not None:
            components.append(signal.novelty_score * 1.2)

        # Lead time value (normalized to hours, max 72h = 1.0)
        if signal.lead_time_hours is not None:
            lead_time_score = min(1.0, signal.lead_time_hours / 72.0)
            components.append(lead_time_score)

        # Severity adds novelty value
        if signal.severity is not None:
            components.append(signal.severity * 0.6)

        if not components:
            return 0.5  # Default neutral

        return min(1.0, max(0.0, sum(components) / len(components)))

    def score_convincing(self, signal: UtilityInput) -> float:
        """
        Score the CONVINCING dimension (0-1).

        Components:
        - Explainability: Can we explain why?
        - Actionability: Can clinician do something?
        - Evidence quality: Is evidence strong?
        """
        components = []

        # Explainability score
        if signal.explainability_score is not None:
            components.append(signal.explainability_score)

        # Actionability score
        if signal.actionability_score is not None:
            components.append(signal.actionability_score * 1.3)

        # Evidence count and quality
        if signal.evidence:
            # More evidence = more convincing
            evidence_count_score = min(1.0, len(signal.evidence) / 5.0)
            components.append(evidence_count_score)

            # Average evidence weight
            avg_weight = sum(e.weight for e in signal.evidence) / len(signal.evidence)
            components.append(avg_weight)

        # Confidence helps convince
        if signal.confidence_score is not None:
            components.append(signal.confidence_score * 0.8)

        if not components:
            return 0.5  # Default neutral

        return min(1.0, max(0.0, sum(components) / len(components)))

    def calculate_handler_score(
        self,
        rightness: float,
        novelty: float,
        convincing: float
    ) -> float:
        """
        Calculate weighted Handler Score.

        Handler Score = R*Wr + N*Wn + C*Wc

        Where weights are mode-specific from policy.
        """
        return (
            rightness * self.policy.rightness_weight +
            novelty * self.policy.novelty_weight +
            convincing * self.policy.convincing_weight
        )

    def estimate_confusion_components(self, signal: UtilityInput) -> Dict[str, float]:
        """
        Estimate expected confusion matrix components.

        Uses PPV, sensitivity estimates to predict:
        - Expected true positives
        - Expected false positives
        - Expected false negatives
        - Expected true negatives

        Returns probabilities for single-case utility calculation.
        """
        # Get probability this is a true positive
        ppv = signal.ppv_estimate or 0.5
        risk = signal.risk_probability or 0.5

        # Estimated confusion probabilities
        # P(TP) = P(alert) * P(disease | alert) = 1 * PPV = PPV
        # P(FP) = P(alert) * P(no disease | alert) = 1 * (1 - PPV) = 1 - PPV
        # P(FN) and P(TN) depend on what we're NOT alerting
        # For a single alert, we model the expected case

        return {
            "p_tp": ppv * risk,
            "p_fp": (1 - ppv) * (1 - risk),
            "p_fn": risk * (1 - ppv),  # Actually has disease but low confidence
            "p_tn": (1 - risk) * ppv,   # Correctly identifying non-cases
        }

    def calculate_net_utility(
        self,
        p_tp: float,
        p_fp: float,
        p_fn: float,
        p_tn: float
    ) -> float:
        """
        Calculate net expected utility.

        U = P(TP)*U(TP) + P(FP)*C(FP) + P(FN)*C(FN) + P(TN)*U(TN)

        Where U() is utility and C() is cost (negative).
        """
        w = self.policy.utility_weights
        return (
            p_tp * w["true_positive_utility"] +
            p_fp * w["false_positive_cost"] +
            p_fn * w["false_negative_cost"] +
            p_tn * w["true_negative_utility"]
        )

    def evaluate(self, signal: UtilityInput) -> Dict:
        """
        Full Handler evaluation of a signal.

        Returns all scores and utility calculation.
        """
        rightness = self.score_rightness(signal)
        novelty = self.score_novelty(signal)
        convincing = self.score_convincing(signal)
        handler_score = self.calculate_handler_score(rightness, novelty, convincing)

        expected = self.estimate_confusion_components(signal)
        net_utility = self.calculate_net_utility(**expected)

        return {
            "rightness": rightness,
            "novelty": novelty,
            "convincing": convincing,
            "handler_score": handler_score,
            "net_utility": net_utility,
            "expected_confusion": expected,
            "should_surface": handler_score >= self.policy.min_handler_score_surface,
            "should_escalate": handler_score >= self.policy.min_handler_score_escalate,
        }
