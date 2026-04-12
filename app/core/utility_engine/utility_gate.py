"""
Utility Gate - The Hard Decision Layer
Final decision governance: surface, snooze, escalate, or suppress.

This is THE critical layer that controls what clinicians see.

CRITICAL RULES:
1. Layer 8 (Recommendations) must NOT see suppressed candidates
2. Trial Rescue must NOT rank by p-value alone - rank by utility

Based on Handler/Feied IEEE framework.

Evolution Integration:
- Emits DECISION signals for every evaluate() call
- Tracks outcomes via record_decision_outcome()
- Parameters tunable through Evolution Controller
"""

from __future__ import annotations
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .handler_utility import HandlerUtilityScorer
from .policy_registry import get_policy
from .schemas import (
    DecisionAction, DeploymentMode, UtilityDecision,
    UtilityInput, UtilityScoreBreakdown,
)

from app.core.evolution import (
    EvolutionEmitter,
    SignalType,
    ParameterUpdate,
    DeploymentDomain,
)

logger = logging.getLogger(__name__)


class UtilityGate:
    """
    Final decision layer: surface, snooze, escalate, or suppress.

    This is where clinical utility beats raw accuracy.
    A high-AUC model is useless if it causes alert fatigue.

    Evolution Integration:
    - Emits DECISION signals for tracking
    - Parameters tunable via Evolution Controller
    - Tracks decision outcomes for feedback loop
    """

    VERSION = "1.1.0"

    def __init__(self, mode: DeploymentMode):
        """
        Initialize gate with deployment mode.

        Args:
            mode: DeploymentMode (HOSPITAL, PHARMA, GOVERNMENT)
        """
        self.mode = mode
        self.policy = get_policy(mode)
        self.scorer = HandlerUtilityScorer(self.policy)

        # Evolution emitter for tracking decisions
        self._emitter = EvolutionEmitter(
            agent_id=f"utility_gate_{mode.value}",
            agent_type="utility_gate",
            version=self.VERSION,
            domain=DeploymentDomain.CLINICAL,
            configurable_parameters=self._get_configurable_parameters(),
        )

        # Pending decisions for outcome tracking
        self._pending_decisions: Dict[str, Dict[str, Any]] = {}

        # Stats
        self._decisions_made = 0
        self._decisions_surfaced = 0
        self._decisions_suppressed = 0
        self._decisions_escalated = 0

    def _get_configurable_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Define evolution-tunable parameters."""
        return {
            "surface_threshold_adjustment": {
                "type": "float",
                "min": -0.2,
                "max": 0.2,
                "default": 0.0,
                "description": "Adjustment to surface threshold (higher = more strict)",
            },
            "escalation_sensitivity": {
                "type": "float",
                "min": 0.8,
                "max": 1.2,
                "default": 1.0,
                "description": "Multiplier for escalation sensitivity",
            },
        }

    @property
    def emitter(self) -> EvolutionEmitter:
        """Get the evolution emitter."""
        return self._emitter

    def on_parameter_change(
        self,
        parameter_name: str,
        callback: Callable[[Any, Any], None],
    ) -> None:
        """Register callback for parameter changes."""
        self._emitter.on_parameter_change(parameter_name, callback)

    def evaluate(self, signal: UtilityInput, session_id: Optional[str] = None) -> UtilityDecision:
        """
        Evaluate a signal and decide what action to take.

        Args:
            signal: UtilityInput with all relevant scores
            session_id: Optional session identifier for tracking

        Returns:
            UtilityDecision with action and full breakdown
        """
        start_time = time.perf_counter()
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

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Emit decision signal for evolution tracking
        evolution_signal = self._emitter.emit(
            signal_type=SignalType.DECISION,
            payload={
                "action": action.value,
                "handler_score": handler_score,
                "net_utility": net_utility,
                "should_surface": should_surface,
                "should_escalate": should_escalate,
                "suppression_reasons": suppression_reasons,
                "escalation_reasons": escalation_reasons,
            },
            session_id=session_id,
        )

        # Update stats
        self._decisions_made += 1
        if action == DecisionAction.SURFACE:
            self._decisions_surfaced += 1
        elif action == DecisionAction.SUPPRESS:
            self._decisions_suppressed += 1
        elif action == DecisionAction.ESCALATE:
            self._decisions_escalated += 1

        # Store for outcome tracking
        self._pending_decisions[evolution_signal.request_id] = {
            "signal": signal,
            "action": action,
            "handler_score": handler_score,
        }

        # Cleanup old pending decisions (keep last 1000)
        if len(self._pending_decisions) > 1000:
            oldest_keys = list(self._pending_decisions.keys())[:-1000]
            for key in oldest_keys:
                del self._pending_decisions[key]

        decision = UtilityDecision(
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
                "evolution_request_id": evolution_signal.request_id,
                "latency_ms": latency_ms,
            },
        )

        return decision

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

    # =========================================================================
    # EVOLUTION INTEGRATION
    # =========================================================================

    def record_decision_outcome(
        self,
        request_id: str,
        outcome: Dict[str, Any],
    ) -> None:
        """
        Record the outcome of a previous decision for feedback loop.

        Args:
            request_id: The evolution request_id from the decision metadata
            outcome: Outcome data, e.g.:
                {
                    "clinician_agreed": True,
                    "patient_outcome": "improved",
                    "was_correct_decision": True,
                }
        """
        self._emitter.record_outcome(request_id, outcome)

        # Remove from pending
        self._pending_decisions.pop(request_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get utility gate statistics including evolution data."""
        return {
            "mode": self.mode.value,
            "decisions_made": self._decisions_made,
            "decisions_surfaced": self._decisions_surfaced,
            "decisions_suppressed": self._decisions_suppressed,
            "decisions_escalated": self._decisions_escalated,
            "surface_rate": (
                self._decisions_surfaced / self._decisions_made
                if self._decisions_made > 0 else 0
            ),
            "escalation_rate": (
                self._decisions_escalated / self._decisions_made
                if self._decisions_made > 0 else 0
            ),
            "pending_outcomes": len(self._pending_decisions),
            "evolution": self._emitter.get_stats(),
        }
