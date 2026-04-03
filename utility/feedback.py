"""
Alert Feedback Tracking
Logs what happens after alerts are emitted
Uses Redis for persistence
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from .redis_store import get_redis_store


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

    @classmethod
    def from_dict(cls, data: Dict) -> 'AlertFeedback':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class FeedbackTracker:
    """
    Tracks alert feedback for utility learning.
    Uses Redis for persistence, falls back to in-memory.
    """

    def __init__(self):
        self._store = get_redis_store()

    def create_feedback(
        self,
        alert_id: str,
        event_id: Optional[str],
        patient_id: str,
        utility_score: float,
        emission_decision: str
    ) -> AlertFeedback:
        """Create feedback record when alert is emitted."""
        feedback = {
            'id': f"FB-{uuid.uuid4().hex[:12]}",
            'alert_id': alert_id,
            'event_id': event_id,
            'patient_id': patient_id,
            'fired_at': datetime.now().isoformat(),
            'utility_score_at_fire': utility_score,
            'emission_decision': emission_decision,
            'viewed_at': None,
            'view_duration_seconds': None,
            'acknowledged_at': None,
            'acknowledged_by': None,
            'acknowledgment_type': None,
            'acknowledgment_notes': None,
            'action_taken_within_1h': False,
            'action_taken_within_4h': False,
            'action_type': None,
            'outcome_at_24h': None,
            'outcome_at_72h': None,
            'was_useful': None,
            'feedback_calculated_at': None
        }

        self._store.save_feedback(feedback)
        return AlertFeedback.from_dict(feedback)

    def record_view(
        self,
        alert_id: str,
        duration_seconds: int
    ) -> Optional[AlertFeedback]:
        """Record that alert was viewed."""
        feedback = self._store.get_feedback_by_alert(alert_id)
        if feedback:
            if not feedback.get('viewed_at'):
                feedback['viewed_at'] = datetime.now().isoformat()
            feedback['view_duration_seconds'] = (feedback.get('view_duration_seconds') or 0) + duration_seconds
            self._store.save_feedback(feedback)
            return AlertFeedback.from_dict(feedback)
        return None

    def record_acknowledgment(
        self,
        alert_id: str,
        user_id: str,
        ack_type: str,
        notes: Optional[str] = None
    ) -> Optional[AlertFeedback]:
        """Record alert acknowledgment."""
        feedback = self._store.get_feedback_by_alert(alert_id)
        if feedback:
            feedback['acknowledged_at'] = datetime.now().isoformat()
            feedback['acknowledged_by'] = user_id
            feedback['acknowledgment_type'] = ack_type
            feedback['acknowledgment_notes'] = notes

            # Calculate immediate usefulness
            if ack_type == AcknowledgmentType.NOT_APPLICABLE.value:
                feedback['was_useful'] = False
                feedback['feedback_calculated_at'] = datetime.now().isoformat()
            elif ack_type == AcknowledgmentType.INTERVENED.value:
                feedback['was_useful'] = True
                feedback['feedback_calculated_at'] = datetime.now().isoformat()

            self._store.save_feedback(feedback)
            self._store.record_acknowledgment(feedback['id'])
            return AlertFeedback.from_dict(feedback)
        return None

    def record_action(
        self,
        alert_id: str,
        action_type: str
    ) -> Optional[AlertFeedback]:
        """Record that action was taken post-alert."""
        feedback = self._store.get_feedback_by_alert(alert_id)
        if feedback:
            fired_at = datetime.fromisoformat(feedback['fired_at'])
            hours_since = (datetime.now() - fired_at).total_seconds() / 3600

            if hours_since <= 1:
                feedback['action_taken_within_1h'] = True
            if hours_since <= 4:
                feedback['action_taken_within_4h'] = True

            feedback['action_type'] = action_type
            feedback['was_useful'] = True
            feedback['feedback_calculated_at'] = datetime.now().isoformat()

            self._store.save_feedback(feedback)
            self._store.record_action(feedback['id'])
            return AlertFeedback.from_dict(feedback)
        return None

    def record_outcome(
        self,
        alert_id: str,
        outcome: str,
        hours: int = 24
    ) -> Optional[AlertFeedback]:
        """Record patient outcome at 24h or 72h."""
        feedback = self._store.get_feedback_by_alert(alert_id)
        if feedback:
            if hours <= 24:
                feedback['outcome_at_24h'] = outcome
            else:
                feedback['outcome_at_72h'] = outcome

            # Update usefulness based on outcome
            self._calculate_usefulness(feedback)
            self._store.save_feedback(feedback)
            return AlertFeedback.from_dict(feedback)
        return None

    def _calculate_usefulness(self, feedback: Dict) -> None:
        """Calculate whether alert was useful based on all data."""
        if feedback.get('was_useful') is not None:
            return  # Already determined

        # Not applicable = not useful
        if feedback.get('acknowledgment_type') == AcknowledgmentType.NOT_APPLICABLE.value:
            feedback['was_useful'] = False
            feedback['feedback_calculated_at'] = datetime.now().isoformat()
            return

        # Action taken = useful
        if feedback.get('action_taken_within_4h'):
            feedback['was_useful'] = True
            feedback['feedback_calculated_at'] = datetime.now().isoformat()
            return

        # Intervened = useful
        if feedback.get('acknowledgment_type') == AcknowledgmentType.INTERVENED.value:
            feedback['was_useful'] = True
            feedback['feedback_calculated_at'] = datetime.now().isoformat()
            return

        # Monitoring + improved = useful
        if (feedback.get('acknowledgment_type') == AcknowledgmentType.MONITORING.value and
            feedback.get('outcome_at_24h') in [OutcomeType.IMPROVED.value, OutcomeType.STABLE.value]):
            feedback['was_useful'] = True
            feedback['feedback_calculated_at'] = datetime.now().isoformat()
            return

        # Not acknowledged = not useful (after 4 hours)
        if not feedback.get('acknowledged_at'):
            fired_at = datetime.fromisoformat(feedback['fired_at'])
            if (datetime.now() - fired_at).total_seconds() > 4 * 3600:
                feedback['was_useful'] = False
                feedback['feedback_calculated_at'] = datetime.now().isoformat()

    def get_feedback(self, feedback_id: str) -> Optional[AlertFeedback]:
        feedback = self._store.get_feedback(feedback_id)
        if feedback:
            return AlertFeedback.from_dict(feedback)
        return None

    def get_patient_feedback(
        self,
        patient_id: str,
        since_hours: int = 24
    ) -> List[AlertFeedback]:
        feedback_dicts = self._store.get_patient_feedback(patient_id, since_hours)
        return [AlertFeedback.from_dict(fb) for fb in feedback_dicts]

    def get_utility_metrics(
        self,
        since_hours: int = 168  # 1 week default
    ) -> Dict:
        """Calculate utility KPIs from Redis."""
        return self._store.get_metrics(since_hours)


# Singleton
_tracker = None


def get_feedback_tracker() -> FeedbackTracker:
    global _tracker
    if _tracker is None:
        _tracker = FeedbackTracker()
    return _tracker
