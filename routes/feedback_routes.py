"""
API Routes for Alert Feedback Tracking (FastAPI)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any

from utility.feedback import get_feedback_tracker, AcknowledgmentType

router = APIRouter(prefix="/feedback", tags=["feedback"])


class CreateFeedbackRequest(BaseModel):
    alert_id: str
    event_id: Optional[str] = None
    patient_id: str
    utility_score: float
    emission_decision: str


class RecordViewRequest(BaseModel):
    duration_seconds: int


class RecordAcknowledgmentRequest(BaseModel):
    user_id: str
    acknowledgment_type: str
    notes: Optional[str] = None


class RecordActionRequest(BaseModel):
    action_type: str


class RecordOutcomeRequest(BaseModel):
    outcome: str
    hours: int = 24


@router.post("/create")
async def create_feedback(request: CreateFeedbackRequest):
    """
    Create feedback record when alert is emitted.

    Request body:
    {
        "alert_id": "ALT-123",
        "event_id": "EVT-456",
        "patient_id": "P001",
        "utility_score": 0.85,
        "emission_decision": "fire"
    }
    """
    try:
        tracker = get_feedback_tracker()
        feedback = tracker.create_feedback(
            alert_id=request.alert_id,
            event_id=request.event_id,
            patient_id=request.patient_id,
            utility_score=request.utility_score,
            emission_decision=request.emission_decision
        )

        return {
            'success': True,
            'feedback': feedback.to_dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{feedback_id}")
async def get_feedback(feedback_id: str):
    """Get feedback by ID."""
    try:
        tracker = get_feedback_tracker()
        feedback = tracker.get_feedback(feedback_id)

        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")

        return {
            'success': True,
            'feedback': feedback.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alert/{alert_id}/view")
async def record_view(alert_id: str, request: RecordViewRequest):
    """
    Record that alert was viewed.

    Request body:
    {
        "duration_seconds": 15
    }
    """
    try:
        tracker = get_feedback_tracker()
        feedback = tracker.record_view(alert_id, request.duration_seconds)

        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found for alert")

        return {
            'success': True,
            'feedback': feedback.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alert/{alert_id}/acknowledge")
async def record_acknowledgment(alert_id: str, request: RecordAcknowledgmentRequest):
    """
    Record alert acknowledgment.

    Request body:
    {
        "user_id": "U123",
        "acknowledgment_type": "intervened",  # assessed_stable, intervened, escalated, monitoring, not_applicable
        "notes": "Started IV fluids"
    }
    """
    try:
        tracker = get_feedback_tracker()
        feedback = tracker.record_acknowledgment(
            alert_id=alert_id,
            user_id=request.user_id,
            ack_type=request.acknowledgment_type,
            notes=request.notes
        )

        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found for alert")

        return {
            'success': True,
            'feedback': feedback.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alert/{alert_id}/action")
async def record_action(alert_id: str, request: RecordActionRequest):
    """
    Record that action was taken post-alert.

    Request body:
    {
        "action_type": "medication_change"
    }
    """
    try:
        tracker = get_feedback_tracker()
        feedback = tracker.record_action(alert_id, request.action_type)

        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found for alert")

        return {
            'success': True,
            'feedback': feedback.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alert/{alert_id}/outcome")
async def record_outcome(alert_id: str, request: RecordOutcomeRequest):
    """
    Record patient outcome at 24h or 72h.

    Request body:
    {
        "outcome": "improved",  # stable, improved, worsened, transferred, coded, expired
        "hours": 24
    }
    """
    try:
        tracker = get_feedback_tracker()
        feedback = tracker.record_outcome(
            alert_id=alert_id,
            outcome=request.outcome,
            hours=request.hours
        )

        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found for alert")

        return {
            'success': True,
            'feedback': feedback.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_id}")
async def get_patient_feedback(patient_id: str, hours: int = 24):
    """
    Get feedback for a patient.

    Query params:
    - hours: Feedback since N hours ago (default 24)
    """
    try:
        tracker = get_feedback_tracker()
        feedback_list = tracker.get_patient_feedback(patient_id, hours)

        return {
            'success': True,
            'patient_id': patient_id,
            'since_hours': hours,
            'feedback': [fb.to_dict() for fb in feedback_list],
            'count': len(feedback_list)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
