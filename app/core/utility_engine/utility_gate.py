"""
Utility Gate - The Hard Decision Layer
Final decision governance: surface, snooze, escalate, or suppress.

This is THE critical layer that controls what clinicians see.

CRITICAL RULES:
1. Layer 8 (Recommendations) must NOT see suppressed candidates
2. Trial Rescue must NOT rank by p-value alone - rank by utility

Based on Handler/Feied IEEE framework.
"""

from __future__ import annotations
from typing import List
from .handler_utility import HandlerUtilityScorer
from .policy_registry import get_policy
from .schemas import (
    DecisionAction, DeploymentMode, UtilityDecision,
    UtilityInput, UtilityScoreBreakdown,
)


class UtilityGate:
    """
    Final decision layer: surface, snooze, escalate, or suppress.

    This is where clinical utility beats raw accuracy.
    A high-AUC model is useless if it causes alert fatigue.
    """

    def __init__(self, mode: DeploymentMode):
        """
        Initialize gate with deployment mode.

        Args:
            mode: DeploymentMode (HOSPITAL, PHARMA, GOVERNMENT)
        """
        self.mode = mode
        self.policy = get_policy(mode)
        self.scorer = HandlerUtilityScorer(self.policy)

    def evaluate(self, signal: UtilityInput) -> UtilityDecision:
        """
        Evaluate a signal and decide what action to take.

        Args:
            signal: UtilityInput with all relevant scores

        Returns:
            UtilityDecision with action and full breakdown
        """
        suppression_reasons: List[str] = []
        escalation_reasons: List[str] = []

        # Calculate Handler scores
        rightness = self.scorer.score_rightness(signal)
        novelty = self.scorer.score_novelty(signal)
        convincing = self.scorer.score_convincing(signal)
        handler_score = self.scorer.calculate_handler_score(rightness, novelty, convincing)

        # Calculate net utility
        expected = self.scorer.estimate_confusion_components(signal)
        net_utility = self.scorer.calculate_net_utility(**expected)

        # Get metadata flags
        repeat_count = int(signal.metadata.get("repeat_count", 0))
        manually_acknowledged = bool(signal.metadata.get("manually_acknowledged", False))
        hard_escalation_flag = bool(signal.metadata.get("hard_escalation_flag", False))

        # === CHECK SUPPRESSION CONDITIONS ===
        if (signal.ppv_estimate or 0.0) < self.policy.min_ppv_for_surface:
            suppression_reasons.append("ppv_below_policy")

        if novelty < self.policy.min_novelty_for_surface:
            suppression_reasons.append("novelty_below_policy")

        if convincing < self.policy.min_convincing_for_surface:
            suppression_reasons.append("convincing_below_policy")

        if handler_score < self.policy.min_handler_score_surface:
            suppression_reasons.append("handler_score_below_surface_threshold")

        # === CHECK ESCALATION CONDITIONS ===
        should_escalate = False

        if hard_escalation_flag:
            should_escalate = True
            escalation_reasons.append("hard_escalation_flag")

        if handler_score >= self.policy.min_handler_score_escalate:
            should_escalate = True
            escalation_reasons.append("handler_score_above_escalation_threshold")

        if repeat_count >= self.policy.max_repeat_count_before_escalation:
            should_escalate = True
            escalation_reasons.append("repeat_count_threshold_reached")

        # Manual acknowledgment blocks escalation
        if should_escalate and manually_acknowledged:
            should_escalate = False
            escalation_reasons.append("escalation_blocked_acknowledged")

        # === DETERMINE ACTION ===
        action = DecisionAction.SUPPRESS
        should_surface = False
        should_notify = False
        snooze_hours = None
        priority = "low"

        if should_escalate:
            # ESCALATE: Force immediate attention
            action = DecisionAction.ESCALATE
            should_surface = True
            should_notify = True
            priority = "critical"

        elif not suppression_reasons:
            # SURFACE: Show to user
            action = DecisionAction.SURFACE
            should_surface = True
            should_notify = True
            priority = "high" if handler_score >= 0.72 else "medium"

        else:
            # Check snooze eligibility
            snooze_eligible = (
                self.policy.snooze_duration_hours > 0
                and repeat_count > 0
                and not hard_escalation_flag
            )

            if snooze_eligible:
                # SNOOZE: Defer for later
                action = DecisionAction.SNOOZE
                snooze_hours = self.policy.snooze_duration_hours
                priority = "deferred"
            else:
                # SUPPRESS: Hide from user
                action = DecisionAction.SUPPRESS
                priority = "silent"

        # Build breakdown
        breakdown = UtilityScoreBreakdown(
            rightness=rightness,
            novelty=novelty,
            convincing=convincing,
            handler_score=handler_score,
            net_utility=net_utility,
            suppression_reasons=suppression_reasons,
            escalation_reasons=escalation_reasons,
        )

        return UtilityDecision(
            action=action,
            should_surface=should_surface,
            should_notify=should_notify,
            should_escalate=should_escalate,
            snooze_hours=snooze_hours,
            priority=priority,
            breakdown=breakdown,
            metadata={
                "mode": self.mode.value,
                "expected_confusion": expected,
            },
        )

    def evaluate_batch(self, signals: List[UtilityInput]) -> List[UtilityDecision]:
        """
        Evaluate multiple signals.

        Args:
            signals: List of UtilityInput objects

        Returns:
            List of UtilityDecision objects
        """
        return [self.evaluate(signal) for signal in signals]

    def get_surfaced(self, signals: List[UtilityInput]) -> List[dict]:
        """
        Evaluate signals and return only surfaced ones.

        Args:
            signals: List of UtilityInput objects

        Returns:
            List of dicts with signal and decision for surfaced items
        """
        results = []
        for signal in signals:
            decision = self.evaluate(signal)
            if decision.should_surface:
                results.append({
                    "signal": signal,
                    "decision": decision,
                })
        return results

    def get_suppressed(self, signals: List[UtilityInput]) -> List[dict]:
        """
        Evaluate signals and return only suppressed ones.

        Args:
            signals: List of UtilityInput objects

        Returns:
            List of dicts with signal and decision for suppressed items
        """
        results = []
        for signal in signals:
            decision = self.evaluate(signal)
            if not decision.should_surface:
                results.append({
                    "signal": signal,
                    "decision": decision,
                })
        return results
