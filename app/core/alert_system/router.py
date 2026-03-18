"""
Alert System FastAPI Router - All Endpoints
Provides the unified /patient/intake endpoint and all alert-related APIs.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import logging

from fastapi import APIRouter, HTTPException, Query, WebSocket, Depends
from fastapi.responses import StreamingResponse

from .models import (
    ClinicalState,
    AlertType,
    AlertSeverity,
    EventType,
    generate_ack_id,
    AcknowledgmentRecord,
)
from .engine import get_engine, evaluate_patient
from .storage import get_storage
from .routing import get_router, get_escalation_manager
from .pipeline import get_pipeline, process_patient_intake
from .realtime import get_hub, websocket_handler, sse_generator
from .config import DOMAIN_CONFIGS, get_domain_config, BIOMARKER_THRESHOLDS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alert System"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PatientIntakeRequest(BaseModel):
    """Request body for unified patient intake."""
    patient_id: str = Field(..., description="Unique patient identifier")
    risk_domain: str = Field(..., description="Risk domain (e.g., sepsis, cardiac)")
    lab_data: Optional[Dict[str, float]] = Field(None, description="Lab values by biomarker name")
    vitals_data: Optional[Dict[str, float]] = Field(None, description="Vital signs")
    clinical_data: Optional[Dict[str, Any]] = Field(None, description="Additional clinical data")
    risk_score: Optional[float] = Field(None, ge=0, le=1, description="Pre-calculated risk score (0-1)")
    location: Optional[str] = Field(None, description="Patient location (ward, unit)")
    encounter_id: Optional[str] = Field(None, description="Encounter/visit ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "P12345",
                "risk_domain": "sepsis",
                "lab_data": {
                    "lactate": 4.5,
                    "procalcitonin": 2.1,
                    "wbc": 18.5,
                    "creatinine": 1.8
                },
                "vitals_data": {
                    "heart_rate": 115,
                    "respiratory_rate": 24,
                    "temperature": 38.9,
                    "blood_pressure_systolic": 88
                },
                "location": "ICU-2A"
            }
        }


class EvaluateRequest(BaseModel):
    """Request body for direct evaluation."""
    patient_id: str
    risk_domain: str
    risk_score: float = Field(..., ge=0, le=1)
    contributing_biomarkers: List[str] = Field(default_factory=list)
    tth_hours: Optional[float] = Field(None, ge=0)


class AcknowledgeRequest(BaseModel):
    """Request body for acknowledging an alert."""
    alert_id: str = Field(..., description="Event ID of the alert to acknowledge")
    acknowledged_by: str = Field(..., description="Clinician ID or name")
    action_taken: Optional[str] = Field(None, description="Action taken in response")
    notes: Optional[str] = Field(None, description="Additional notes")
    close_episode: bool = Field(False, description="Whether to close the episode")


class SubscriptionRequest(BaseModel):
    """Request body for updating subscriptions."""
    patient_ids: Optional[List[str]] = None
    risk_domains: Optional[List[str]] = None
    roles: Optional[List[str]] = None


# =============================================================================
# UNIFIED PATIENT INTAKE ENDPOINT
# =============================================================================

@router.post("/patient/intake")
async def patient_intake(request: PatientIntakeRequest):
    """
    Unified patient intake endpoint.

    Processes patient data through the complete 11-step pipeline:
    1. Data received
    2. Risk score calculated (if not provided)
    3. CSE evaluates state transition
    4. Agents invoked (on state change or high severity)
    5. Cross-validation
    6. TTH prediction
    7. Confidence scoring
    8. Alert decision
    9. Alert routing (if fired)
    10. Audit logging
    11. Dashboard notification

    Returns complete pipeline result with evaluation, routing, and timing.
    """
    try:
        result = await process_patient_intake(
            patient_id=request.patient_id,
            risk_domain=request.risk_domain,
            lab_data=request.lab_data,
            vitals_data=request.vitals_data,
            clinical_data=request.clinical_data,
            risk_score=request.risk_score,
            location=request.location,
            encounter_id=request.encounter_id,
            metadata=request.metadata,
        )
        return result.to_dict()

    except Exception as e:
        logger.exception(f"Patient intake failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DIRECT EVALUATION ENDPOINT
# =============================================================================

@router.post("/evaluate")
async def evaluate_alert(request: EvaluateRequest):
    """
    Direct CSE evaluation without full pipeline.

    Use this for testing or when risk score is pre-calculated
    and full pipeline is not needed.
    """
    try:
        result = evaluate_patient(
            patient_id=request.patient_id,
            risk_domain=request.risk_domain,
            risk_score=request.risk_score,
            contributing_biomarkers=request.contributing_biomarkers,
            tth_hours=request.tth_hours,
        )
        return result.to_dict()

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ACKNOWLEDGMENT ENDPOINT
# =============================================================================

@router.post("/acknowledge")
async def acknowledge_alert(request: AcknowledgeRequest):
    """
    Acknowledge an alert.

    This stops escalation timers and optionally closes the episode.
    """
    storage = get_storage()
    escalation_manager = get_escalation_manager()
    hub = get_hub()

    # Get the alert event
    event = storage.get_event_by_id(request.alert_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Alert {request.alert_id} not found")

    # Create acknowledgment record
    ack = AcknowledgmentRecord(
        ack_id=generate_ack_id(),
        alert_id=request.alert_id,
        patient_id=event.patient_id,
        episode_id=event.episode_id or "",
        acknowledged_by=request.acknowledged_by,
        acknowledged_at=datetime.now(timezone.utc),
        action_taken=request.action_taken,
        notes=request.notes,
        close_episode=request.close_episode,
    )

    # Save acknowledgment
    storage.save_acknowledgment(ack)

    # Cancel escalation timer
    escalation_manager.cancel_escalation(request.alert_id)

    # Notify dashboards
    await hub.notify_acknowledgment(
        event.patient_id,
        event.risk_domain,
        ack.to_dict(),
    )

    return {
        "success": True,
        "ack_id": ack.ack_id,
        "alert_id": request.alert_id,
        "escalation_cancelled": True,
        "episode_closed": request.close_episode,
    }


# =============================================================================
# PATIENT STATE ENDPOINTS
# =============================================================================

@router.get("/patient/{patient_id}/state")
async def get_patient_states(patient_id: str):
    """Get current state for a patient across all risk domains."""
    storage = get_storage()
    states = {}

    for domain in DOMAIN_CONFIGS.keys():
        state = storage.get_patient_state(patient_id, domain)
        if state:
            states[domain] = state.to_dict()

    if not states:
        raise HTTPException(status_code=404, detail=f"No state found for patient {patient_id}")

    return {
        "patient_id": patient_id,
        "states": states,
        "domain_count": len(states),
    }


@router.get("/patient/{patient_id}/state/{risk_domain}")
async def get_patient_domain_state(patient_id: str, risk_domain: str):
    """Get current state for a specific patient and risk domain."""
    storage = get_storage()
    state = storage.get_patient_state(patient_id, risk_domain)

    if not state:
        raise HTTPException(
            status_code=404,
            detail=f"No state found for patient {patient_id} in domain {risk_domain}"
        )

    return state.to_dict()


# =============================================================================
# EPISODE ENDPOINTS
# =============================================================================

@router.get("/episodes")
async def get_episodes(
    patient_id: Optional[str] = Query(None),
    open_only: bool = Query(True),
    limit: int = Query(100, le=1000),
):
    """Get episodes, optionally filtered by patient."""
    storage = get_storage()

    if open_only:
        episodes = storage.get_open_episodes(patient_id)
    else:
        # For closed episodes, we'd need to query events
        episodes = storage.get_open_episodes(patient_id)

    return {
        "episodes": [e.to_dict() for e in episodes[:limit]],
        "count": len(episodes),
        "open_only": open_only,
    }


@router.get("/episodes/{episode_id}")
async def get_episode(episode_id: str):
    """Get a specific episode by ID."""
    storage = get_storage()
    episode = storage.get_episode(episode_id)

    if not episode:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")

    return episode.to_dict()


# =============================================================================
# EVENT/AUDIT LOG ENDPOINTS
# =============================================================================

@router.get("/events")
async def get_events(
    patient_id: Optional[str] = Query(None),
    risk_domain: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    since: Optional[datetime] = Query(None),
    until: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
):
    """Query the audit log with filters."""
    storage = get_storage()

    event_type_enum = None
    if event_type:
        try:
            event_type_enum = EventType(event_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event_type: {event_type}")

    events = storage.get_events(
        patient_id=patient_id,
        risk_domain=risk_domain,
        event_type=event_type_enum,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )

    return {
        "events": [e.to_dict() for e in events],
        "count": len(events),
        "filters": {
            "patient_id": patient_id,
            "risk_domain": risk_domain,
            "event_type": event_type,
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
        },
    }


@router.get("/events/{event_id}")
async def get_event(event_id: str):
    """Get a specific event by ID."""
    storage = get_storage()
    event = storage.get_event_by_id(event_id)

    if not event:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    return event.to_dict()


# =============================================================================
# ACKNOWLEDGMENT HISTORY
# =============================================================================

@router.get("/acknowledgments")
async def get_acknowledgments(
    patient_id: Optional[str] = Query(None),
    episode_id: Optional[str] = Query(None),
    since: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
):
    """Get acknowledgment records."""
    storage = get_storage()

    acks = storage.get_acknowledgments(
        patient_id=patient_id,
        episode_id=episode_id,
        since=since,
        limit=limit,
    )

    return {
        "acknowledgments": [a.to_dict() for a in acks],
        "count": len(acks),
    }


# =============================================================================
# ROUTING RULES ENDPOINTS
# =============================================================================

@router.get("/routing/rules")
async def get_routing_rules():
    """Get all configured routing rules."""
    alert_router = get_router()
    rules = alert_router.get_rules()

    return {
        "rules": [r.to_dict() for r in rules],
        "count": len(rules),
    }


@router.get("/routing/escalations")
async def get_pending_escalations():
    """Get all pending escalations."""
    manager = get_escalation_manager()

    return {
        "pending": [p.to_dict() for p in manager.pending.values()],
        "count": manager.get_pending_count(),
    }


# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@router.get("/config/domains")
async def get_domain_configs():
    """Get all domain configurations."""
    return {
        "domains": {name: cfg.to_dict() for name, cfg in DOMAIN_CONFIGS.items()},
        "count": len(DOMAIN_CONFIGS),
    }


@router.get("/config/domains/{risk_domain}")
async def get_domain_config_endpoint(risk_domain: str):
    """Get configuration for a specific domain."""
    config = get_domain_config(risk_domain)
    if not config:
        raise HTTPException(status_code=404, detail=f"Domain {risk_domain} not found")

    return {
        "domain": risk_domain,
        "config": config.to_dict(),
    }


@router.get("/config/biomarkers")
async def get_biomarker_thresholds():
    """Get all biomarker thresholds."""
    result = {}
    for domain, biomarkers in BIOMARKER_THRESHOLDS.items():
        result[domain] = {
            name: {
                "warning": t.warning,
                "critical": t.critical,
                "unit": t.unit,
                "direction": t.direction,
                "weight": t.weight,
            }
            for name, t in biomarkers.items()
        }

    return {
        "domains": result,
        "domain_count": len(result),
        "total_biomarkers": sum(len(b) for b in BIOMARKER_THRESHOLDS.values()),
    }


@router.get("/config/biomarkers/{risk_domain}")
async def get_domain_biomarkers(risk_domain: str):
    """Get biomarker thresholds for a specific domain."""
    if risk_domain not in BIOMARKER_THRESHOLDS:
        raise HTTPException(status_code=404, detail=f"No biomarkers for domain {risk_domain}")

    biomarkers = BIOMARKER_THRESHOLDS[risk_domain]
    return {
        "domain": risk_domain,
        "biomarkers": {
            name: {
                "warning": t.warning,
                "critical": t.critical,
                "unit": t.unit,
                "direction": t.direction,
                "weight": t.weight,
            }
            for name, t in biomarkers.items()
        },
        "count": len(biomarkers),
    }


# =============================================================================
# STATISTICS ENDPOINT
# =============================================================================

@router.get("/stats")
async def get_stats():
    """Get alert system statistics."""
    storage = get_storage()
    hub = get_hub()
    escalation_manager = get_escalation_manager()

    return {
        "storage": storage.get_stats(),
        "realtime": hub.connections.get_stats(),
        "escalations": {
            "pending": escalation_manager.get_pending_count(),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# REAL-TIME ENDPOINTS
# =============================================================================

@router.websocket("/ws")
async def alerts_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alert updates.

    After connecting, send subscription messages to filter alerts:
    ```json
    {
        "type": "subscribe",
        "patient_ids": ["P12345"],
        "risk_domains": ["sepsis", "cardiac"],
        "roles": ["attending"]
    }
    ```
    """
    hub = get_hub()
    await websocket_handler(websocket, hub)


@router.get("/sse")
async def alerts_sse(
    patient_ids: List[str] = Query(None),
    risk_domains: List[str] = Query(None),
    roles: List[str] = Query(None),
):
    """
    Server-Sent Events endpoint for real-time alert updates.

    Use query parameters to filter alerts:
    - patient_ids: List of patient IDs to subscribe to
    - risk_domains: List of risk domains to subscribe to
    - roles: List of clinician roles to receive alerts for
    """
    hub = get_hub()
    return StreamingResponse(
        sse_generator(hub, patient_ids, risk_domains, roles),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get("/health")
async def health_check():
    """Health check for the alert system."""
    try:
        storage = get_storage()
        stats = storage.get_stats()

        return {
            "status": "healthy",
            "storage_type": stats.get("storage_type", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
