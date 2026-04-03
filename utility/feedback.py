"""
Alert Feedback Tracking
Logs what happens after alerts are emitted
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid


class AcknowledgmentType(Enum):
    ASSESSED_STABLE = "assessed_stable"
    INTERVENED = "intervened"
    ESCALATED = "escalated"
    MONITORING = "monitoring"
    NOT_APPLICABLE = "not_applicable"


class OutcomeType(Enum):
    STABLE = "stable"
    IMPROVED = "improved"
    WORSENED = "worsened"
    TRANSFERRED = "transferred"
    CODED = "coded"
    EXPIRED = "expired"


@dataclass
class AlertFeedback:
    id: str
    alert_id: str
    event_id: Optional[str]
    patient_id: str
    fired_at: str
    utility_score_at_fire: float
    emission_decision: str

    # View tracking
    viewed_at: Optional[str] = None
    view_duration_seconds: Optional[int] = None

    # Acknowledgment
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledgment_type: Optional[str] = None
    acknowledgment_notes: Optional[str] = None

    # Action tracking
    action_taken_within_1h: bool = False
    action_taken_within_4h: bool = False
    action_type: Optional[str] = None

    # Outcome tracking
    outcome_at_24h: Optional[str] = None
    outcome_at_72h: Optional[str] = None

    # Derived
    was_useful: Optional[bool] = None
    feedback_calculated_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class FeedbackTracker:
    """
    Tracks alert feedback for utility learning.
    """

    def __init__(self):
        self._feedback: Dict[str, AlertFeedback] = {}

    def create_feedback(
        self,
        alert_id: str,
        event_id: Optional[str],
        patient_id: str,
        utility_score: float,
        emission_decision: str
    ) -> AlertFeedback:
        """Create feedback record when alert is emitted."""
        feedback = AlertFeedback(
            id=f"FB-{uuid.uuid4().hex[:12]}",
            alert_id=alert_id,
            event_id=event_id,
            patient_id=patient_id,
            fired_at=datetime.now().isoformat(),
            utility_score_at_fire=utility_score,
            emission_decision=emission_decision
        )
        self._feedback[feedback.id] = feedback
        return feedback

    def record_view(
        self,
        alert_id: str,
        duration_seconds: int
    ) -> Optional[AlertFeedback]:
        """Record that alert was viewed."""
        feedback = self._get_by_alert_id(alert_id)
        if feedback:
            if not feedback.viewed_at:
                feedback.viewed_at = datetime.now().isoformat()
            feedback.view_duration_seconds = (feedback.view_duration_seconds or 0) + duration_seconds
        return feedback

    def record_acknowledgment(
        self,
        alert_id: str,
        user_id: str,
        ack_type: str,
        notes: Optional[str] = None
    ) -> Optional[AlertFeedback]:
        """Record alert acknowledgment."""
        feedback = self._get_by_alert_id(alert_id)
        if feedback:
            feedback.acknowledged_at = datetime.now().isoformat()
            feedback.acknowledged_by = user_id
            feedback.acknowledgment_type = ack_type
            feedback.acknowledgment_notes = notes

            # Calculate immediate usefulness
            if ack_type == AcknowledgmentType.NOT_APPLICABLE.value:
                feedback.was_useful = False
                feedback.feedback_calculated_at = datetime.now().isoformat()
            elif ack_type == AcknowledgmentType.INTERVENED.value:
                feedback.was_useful = True
                feedback.feedback_calculated_at = datetime.now().isoformat()

        return feedback

    def record_action(
        self,
        alert_id: str,
        action_type: str
    ) -> Optional[AlertFeedback]:
        """Record that action was taken post-alert."""
        feedback = self._get_by_alert_id(alert_id)
        if feedback:
            fired_at = datetime.fromisoformat(feedback.fired_at)
            hours_since = (datetime.now() - fired_at).total_seconds() / 3600

            if hours_since <= 1:
                feedback.action_taken_within_1h = True
            if hours_since <= 4:
                feedback.action_taken_within_4h = True

            feedback.action_type = action_type

            # If action taken, likely useful
            if not feedback.was_useful:
                feedback.was_useful = True
                feedback.feedback_calculated_at = datetime.now().isoformat()

        return feedback

    def record_outcome(
        self,
        alert_id: str,
        outcome: str,
        hours: int = 24
    ) -> Optional[AlertFeedback]:
        """Record patient outcome at 24h or 72h."""
        feedback = self._get_by_alert_id(alert_id)
        if feedback:
            if hours <= 24:
                feedback.outcome_at_24h = outcome
            else:
                feedback.outcome_at_72h = outcome

            # Update usefulness based on outcome
            self._calculate_usefulness(feedback)

        return feedback

    def _calculate_usefulness(self, feedback: AlertFeedback) -> None:
        """Calculate whether alert was useful based on all data."""
        if feedback.was_useful is not None:
            return  # Already determined

        # Not applicable = not useful
        if feedback.acknowledgment_type == AcknowledgmentType.NOT_APPLICABLE.value:
            feedback.was_useful = False
            feedback.feedback_calculated_at = datetime.now().isoformat()
            return

        # Action taken = useful
        if feedback.action_taken_within_4h:
            feedback.was_useful = True
            feedback.feedback_calculated_at = datetime.now().isoformat()
            return

        # Intervened = useful
        if feedback.acknowledgment_type == AcknowledgmentType.INTERVENED.value:
            feedback.was_useful = True
            feedback.feedback_calculated_at = datetime.now().isoformat()
            return

        # Monitoring + improved = useful
        if (feedback.acknowledgment_type == AcknowledgmentType.MONITORING.value and
            feedback.outcome_at_24h in [OutcomeType.IMPROVED.value, OutcomeType.STABLE.value]):
            feedback.was_useful = True
            feedback.feedback_calculated_at = datetime.now().isoformat()
            return

        # Not acknowledged = not useful
        if not feedback.acknowledged_at:
            fired_at = datetime.fromisoformat(feedback.fired_at)
            if (datetime.now() - fired_at).total_seconds() > 4 * 3600:  # 4+ hours
                feedback.was_useful = False
                feedback.feedback_calculated_at = datetime.now().isoformat()
            return

    def _get_by_alert_id(self, alert_id: str) -> Optional[AlertFeedback]:
        for fb in self._feedback.values():
            if fb.alert_id == alert_id:
                return fb
        return None

    def get_feedback(self, feedback_id: str) -> Optional[AlertFeedback]:
        return self._feedback.get(feedback_id)

    def get_patient_feedback(
        self,
        patient_id: str,
        since_hours: int = 24
    ) -> List[AlertFeedback]:
        cutoff = datetime.now() - timedelta(hours=since_hours)
        return [
            fb for fb in self._feedback.values()
            if fb.patient_id == patient_id and
            datetime.fromisoformat(fb.fired_at) > cutoff
        ]

    def get_utility_metrics(
        self,
        since_hours: int = 168  # 1 week default
    ) -> Dict:
        """Calculate utility KPIs."""
        cutoff = datetime.now() - timedelta(hours=since_hours)
        recent = [
            fb for fb in self._feedback.values()
            if datetime.fromisoformat(fb.fired_at) > cutoff
        ]

        if not recent:
            return {
                'total_alerts': 0,
                'actionable_rate': 0,
                'conversion_rate': 0,
                'redundant_rate': 0,
                'ignored_rate': 0
            }

        total = len(recent)
        fired = [fb for fb in recent if fb.emission_decision == 'fire']
        suppressed = [fb for fb in recent if fb.emission_decision == 'suppress']

        acknowledged = [fb for fb in fired if fb.acknowledged_at]
        action_taken = [fb for fb in fired if fb.action_taken_within_4h]
        not_applicable = [fb for fb in fired if fb.acknowledgment_type == AcknowledgmentType.NOT_APPLICABLE.value]
        ignored = [fb for fb in fired if not fb.acknowledged_at]
        useful = [fb for fb in recent if fb.was_useful == True]

        return {
            'total_alerts': total,
            'fired': len(fired),
            'suppressed': len(suppressed),
            'suppression_rate': len(suppressed) / total if total else 0,
            'acknowledged': len(acknowledged),
            'acknowledgment_rate': len(acknowledged) / len(fired) if fired else 0,
            'action_taken': len(action_taken),
            'actionable_rate': len(action_taken) / len(fired) if fired else 0,
            'conversion_rate': len(useful) / len(fired) if fired else 0,
            'redundant_rate': len(not_applicable) / len(fired) if fired else 0,
            'ignored_rate': len(ignored) / len(fired) if fired else 0,
        }


# Singleton
_tracker = None


def get_feedback_tracker() -> FeedbackTracker:
    global _tracker
    if _tracker is None:
        _tracker = FeedbackTracker()
    return _tracker
