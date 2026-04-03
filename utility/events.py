"""
Clinical Event Detection and Management
Links alerts to underlying clinical issues
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid


class EventStatus(Enum):
    ACTIVE = "active"
    MONITORING = "monitoring"
    RESOLVED = "resolved"


class EventTrajectory(Enum):
    WORSENING = "worsening"
    STABLE = "stable"
    IMPROVING = "improving"
    RESOLVED = "resolved"


@dataclass
class ClinicalEvent:
    id: str
    patient_id: str
    event_type: str
    primary_endpoint: str
    first_detected_at: str
    first_alert_id: Optional[str]
    alert_count: int
    current_severity: str
    trajectory: str
    last_alert_at: str
    status: str
    resolved_at: Optional[str]
    resolution_type: Optional[str]

    def to_dict(self) -> Dict:
        return asdict(self)


class EventManager:
    """
    Manages clinical events and their lifecycle.
    In production, this would interact with a database.
    For now, we use in-memory storage with optional persistence.
    """

    def __init__(self):
        # In-memory event storage (replace with DB in production)
        self._events: Dict[str, ClinicalEvent] = {}

    def detect_or_link_event(
        self,
        patient_id: str,
        alert_type: str,
        endpoint: str,
        analysis: Dict,
        alert_id: Optional[str] = None
    ) -> Tuple[ClinicalEvent, bool]:
        """
        Find existing event or create new one.
        Returns: (event, is_new)
        """
        # Look for active event matching this alert
        for event in self._events.values():
            if event.patient_id != patient_id:
                continue
            if event.status != EventStatus.ACTIVE.value:
                continue

            # Match by type or endpoint
            if event.event_type == alert_type or event.primary_endpoint == endpoint:
                # Update existing event
                event.alert_count += 1
                event.last_alert_at = datetime.now().isoformat()
                event.current_severity = analysis.get('clinical_state', event.current_severity)

                # Update trajectory based on score change
                velocity = analysis.get('velocity', 'stable')
                if velocity in ['rapid_worsening', 'worsening']:
                    event.trajectory = EventTrajectory.WORSENING.value
                elif velocity in ['rapid_improving', 'improving']:
                    event.trajectory = EventTrajectory.IMPROVING.value
                else:
                    event.trajectory = EventTrajectory.STABLE.value

                return event, False

        # Create new event
        new_event = ClinicalEvent(
            id=f"EVT-{uuid.uuid4().hex[:12]}",
            patient_id=patient_id,
            event_type=alert_type,
            primary_endpoint=endpoint,
            first_detected_at=datetime.now().isoformat(),
            first_alert_id=alert_id,
            alert_count=1,
            current_severity=analysis.get('clinical_state', 'warning'),
            trajectory=EventTrajectory.WORSENING.value,
            last_alert_at=datetime.now().isoformat(),
            status=EventStatus.ACTIVE.value,
            resolved_at=None,
            resolution_type=None
        )

        self._events[new_event.id] = new_event
        return new_event, True

    def get_event(self, event_id: str) -> Optional[ClinicalEvent]:
        return self._events.get(event_id)

    def get_patient_events(
        self,
        patient_id: str,
        status: Optional[str] = None
    ) -> List[ClinicalEvent]:
        events = [e for e in self._events.values() if e.patient_id == patient_id]
        if status:
            events = [e for e in events if e.status == status]
        return events

    def resolve_event(
        self,
        event_id: str,
        resolution_type: str
    ) -> Optional[ClinicalEvent]:
        event = self._events.get(event_id)
        if event:
            event.status = EventStatus.RESOLVED.value
            event.resolved_at = datetime.now().isoformat()
            event.resolution_type = resolution_type
            event.trajectory = EventTrajectory.RESOLVED.value
        return event

    def get_active_events_count(self, patient_id: str) -> int:
        return len([
            e for e in self._events.values()
            if e.patient_id == patient_id and e.status == EventStatus.ACTIVE.value
        ])


# Singleton instance
_event_manager = None


def get_event_manager() -> EventManager:
    global _event_manager
    if _event_manager is None:
        _event_manager = EventManager()
    return _event_manager
