"""
API Routes for Utility Engine (FastAPI)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

from utility.engine import get_utility_engine, EmissionDecision
from utility.events import get_event_manager
from utility.feedback import get_feedback_tracker

router = APIRouter(prefix="/utility", tags=["utility"])


class ProposedAlert(BaseModel):
    tier: str = "warning"
    type: str = ""
    endpoint: str = ""
    score: float = 0
    message: str = ""


class AnalysisData(BaseModel):
    hypercore_score: float = 50
    clinical_state: str = "stable"
    time_to_harm_min: float = 48
    time_to_harm_max: float = 72
    top_endpoints: List[str] = []
    velocity: str = "stable"


class UtilityCalculateRequest(BaseModel):
    patient_id: str
    proposed_alert: ProposedAlert
    analysis: AnalysisData
    user_id: str
    user_alert_load: int = 0


class UtilityDecideRequest(BaseModel):
    patient_id: str
    proposed_alert: ProposedAlert
    analysis: AnalysisData
    user_id: str
    user_alert_load: int = 0


@router.post("/calculate")
async def calculate_utility(request: UtilityCalculateRequest):
    """
    Calculate utility score for a proposed alert.

    Request body:
    {
        "patient_id": "P001",
        "proposed_alert": {
            "tier": "warning",
            "type": "renal_deterioration",
            "endpoint": "renal",
            "score": 72,
            "message": "Renal function declining"
        },
        "analysis": {
            "hypercore_score": 72,
            "clinical_state": "warning",
            "time_to_harm_min": 12,
            "time_to_harm_max": 24,
            "top_endpoints": ["renal", "metabolic"],
            "velocity": "worsening"
        },
        "user_id": "U123",
        "user_alert_load": 3
    }
    """
    try:
        # Get alert and event history
        event_manager = get_event_manager()
        feedback_tracker = get_feedback_tracker()

        event_history = [e.to_dict() for e in event_manager.get_patient_events(request.patient_id, 'active')]
        alert_history = [fb.to_dict() for fb in feedback_tracker.get_patient_feedback(request.patient_id, 48)]

        # Calculate utility
        engine = get_utility_engine()
        result = engine.calculate_utility(
            patient_id=request.patient_id,
            proposed_alert=request.proposed_alert.model_dump(),
            analysis=request.analysis.model_dump(),
            user_id=request.user_id,
            user_alert_load=request.user_alert_load,
            alert_history=alert_history,
            event_history=event_history
        )

        return {
            'success': True,
            'utility_score': result.utility_score,
            'decision': result.decision.value,
            'components': {
                'information_gain': result.components.information_gain,
                'urgency_factor': result.components.urgency_factor,
                'actionability': result.components.actionability,
                'redundancy_penalty': result.components.redundancy_penalty,
                'interruption_cost': result.components.interruption_cost
            },
            'event': {
                'event_id': result.event_id,
                'is_new_event': result.is_new_event,
                'alert_sequence': result.alert_sequence
            },
            'explanation': result.explanation,
            'recommended_action': result.recommended_action,
            'delay_minutes': result.delay_minutes,
            'downgrade_to_tier': result.downgrade_to_tier
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/decide")
async def decide(request: UtilityDecideRequest):
    """
    Simplified endpoint - just returns decision.
    Same input as /calculate.
    """
    try:
        event_manager = get_event_manager()
        feedback_tracker = get_feedback_tracker()

        event_history = [e.to_dict() for e in event_manager.get_patient_events(request.patient_id, 'active')]
        alert_history = [fb.to_dict() for fb in feedback_tracker.get_patient_feedback(request.patient_id, 48)]

        engine = get_utility_engine()
        result = engine.calculate_utility(
            patient_id=request.patient_id,
            proposed_alert=request.proposed_alert.model_dump(),
            analysis=request.analysis.model_dump(),
            user_id=request.user_id,
            user_alert_load=request.user_alert_load,
            alert_history=alert_history,
            event_history=event_history
        )

        return {
            'success': True,
            'decision': result.decision.value,
            'utility_score': result.utility_score,
            'should_fire': result.decision == EmissionDecision.FIRE,
            'delay_minutes': result.delay_minutes
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(hours: int = 168):
    """Get utility KPIs."""
    try:
        tracker = get_feedback_tracker()
        metrics = tracker.get_utility_metrics(hours)

        return {
            'success': True,
            'period_hours': hours,
            'metrics': metrics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
