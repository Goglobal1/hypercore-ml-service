"""
API Routes for Clinical Event Management (FastAPI)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

from utility.events import get_event_manager, EventStatus

router = APIRouter(prefix="/events", tags=["events"])


class DetectEventRequest(BaseModel):
    patient_id: str
    alert_type: str
    endpoint: str
    analysis: Dict[str, Any]
    alert_id: Optional[str] = None


class ResolveEventRequest(BaseModel):
    resolution_type: str


@router.post("/detect")
async def detect_or_link_event(request: DetectEventRequest):
    """
    Detect or link to existing clinical event.

    Request body:
    {
        "patient_id": "P001",
        "alert_type": "renal_deterioration",
        "endpoint": "renal",
        "analysis": {
            "clinical_state": "warning",
            "velocity": "worsening"
        },
        "alert_id": "ALT-123"
    }
    """
    try:
        manager = get_event_manager()
        event, is_new = manager.detect_or_link_event(
            patient_id=request.patient_id,
            alert_type=request.alert_type,
            endpoint=request.endpoint,
            analysis=request.analysis,
            alert_id=request.alert_id
        )

        return {
            'success': True,
            'event': event.to_dict(),
            'is_new_event': is_new
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{event_id}")
async def get_event(event_id: str):
    """Get event by ID."""
    try:
        manager = get_event_manager()
        event = manager.get_event(event_id)

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        return {
            'success': True,
            'event': event.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_id}")
async def get_patient_events(patient_id: str, status: Optional[str] = None):
    """
    Get events for a patient.

    Query params:
    - status: Filter by status (active, monitoring, resolved)
    """
    try:
        manager = get_event_manager()
        events = manager.get_patient_events(patient_id, status)

        return {
            'success': True,
            'patient_id': patient_id,
            'events': [e.to_dict() for e in events],
            'count': len(events)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{event_id}/resolve")
async def resolve_event(event_id: str, request: ResolveEventRequest):
    """
    Resolve a clinical event.

    Request body:
    {
        "resolution_type": "improved"  # improved, transferred, expired, other
    }
    """
    try:
        manager = get_event_manager()
        event = manager.resolve_event(event_id, request.resolution_type)

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        return {
            'success': True,
            'event': event.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_id}/active-count")
async def get_active_events_count(patient_id: str):
    """Get count of active events for a patient."""
    try:
        manager = get_event_manager()
        count = manager.get_active_events_count(patient_id)

        return {
            'success': True,
            'patient_id': patient_id,
            'active_events': count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
