"""
Clinical Event Detection and Management
Links alerts to underlying clinical issues
Uses Redis for persistence
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from .redis_store import get_redis_store


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

    @classmethod
    def from_dict(cls, data: Dict) -> 'ClinicalEvent':
        return cls(**data)


class EventManager:
    """
    Manages clinical events and their lifecycle.
    Uses Redis for persistence, falls back to in-memory.
    """

    def __init__(self):
        self._store = get_redis_store()

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
        # Check Redis for existing event
        existing = self._store.find_matching_event(patient_id, alert_type, endpoint)

        if existing:
            # Update existing event
            existing['alert_count'] = existing.get('alert_count', 1) + 1
            existing['last_alert_at'] = datetime.now().isoformat()
            existing['current_severity'] = analysis.get('clinical_state', existing.get('current_severity'))

            # Update trajectory based on velocity
            velocity = analysis.get('velocity', 'stable')
            if velocity in ['rapid_worsening', 'worsening']:
                existing['trajectory'] = EventTrajectory.WORSENING.value
            elif velocity in ['rapid_improving', 'improving']:
                existing['trajectory'] = EventTrajectory.IMPROVING.value
            else:
                existing['trajectory'] = EventTrajectory.STABLE.value

            self._store.save_event(existing)
            return ClinicalEvent.from_dict(existing), False

        # Create new event
        new_event = {
            'id': f"EVT-{uuid.uuid4().hex[:12]}",
            'patient_id': patient_id,
            'event_type': alert_type,
            'primary_endpoint': endpoint,
            'first_detected_at': datetime.now().isoformat(),
            'first_alert_id': alert_id,
            'alert_count': 1,
            'current_severity': analysis.get('clinical_state', 'warning'),
            'trajectory': EventTrajectory.WORSENING.value,
            'last_alert_at': datetime.now().isoformat(),
            'status': EventStatus.ACTIVE.value,
            'resolved_at': None,
            'resolution_type': None
        }

        self._store.save_event(new_event)
        return ClinicalEvent.from_dict(new_event), True

    def get_event(self, event_id: str) -> Optional[ClinicalEvent]:
        event_dict = self._store.get_event(event_id)
        if event_dict:
            return ClinicalEvent.from_dict(event_dict)
        return None

    def get_patient_events(
        self,
        patient_id: str,
        status: Optional[str] = None
    ) -> List[ClinicalEvent]:
        event_dicts = self._store.get_patient_events(patient_id, status)
        return [ClinicalEvent.from_dict(e) for e in event_dicts]

    def resolve_event(
        self,
        event_id: str,
        resolution_type: str
    ) -> Optional[ClinicalEvent]:
        event = self._store.get_event(event_id)
        if event:
            event['status'] = EventStatus.RESOLVED.value
            event['resolved_at'] = datetime.now().isoformat()
            event['resolution_type'] = resolution_type
            event['trajectory'] = EventTrajectory.RESOLVED.value
            self._store.save_event(event)
            return ClinicalEvent.from_dict(event)
        return None

    def get_active_events_count(self, patient_id: str) -> int:
        events = self._store.get_patient_events(patient_id, status=EventStatus.ACTIVE.value)
        return len(events)


# Singleton instance
_event_manager = None


def get_event_manager() -> EventManager:
    global _event_manager
    if _event_manager is None:
        _event_manager = EventManager()
    return _event_manager
