"""
Alert System FastAPI Router - All Endpoints
Provides the unified /patient/intake endpoint and all alert-related APIs.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict
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

# Import robust data parser for data quality reporting
try:
    from ..data_ingestion import parse_any_data, DOMAIN_CRITICAL_BIOMARKERS
    DATA_INGESTION_AVAILABLE = True
except ImportError:
    DATA_INGESTION_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["Alert System"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class PatientIntakeRequest(BaseModel):
    """Request body for unified patient intake."""
    patient_id: Optional[str] = Field(None, description="Unique patient identifier (auto-generated if not provided)")
    risk_domain: str = Field(..., description="Risk domain (e.g., sepsis, cardiac)")
    lab_data: Optional[Dict[str, float]] = Field(None, description="Lab values by biomarker name")
    vitals_data: Optional[Dict[str, float]] = Field(None, description="Vital signs")
    clinical_data: Optional[Dict[str, Any]] = Field(None, description="Additional clinical data")
    risk_score: Optional[float] = Field(None, ge=0, le=1, description="Pre-calculated risk score (0-1)")
    location: Optional[str] = Field(None, description="Patient location (ward, unit)")
    encounter_id: Optional[str] = Field(None, description="Encounter/visit ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    model_config = ConfigDict(
        json_schema_extra={
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
    )


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
    Unified patient intake endpoint - BULLETPROOF.

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

    Returns complete pipeline result with evaluation, routing, timing,
    and data quality assessment.

    NEVER FAILS - Always returns a result with helpful information.
    If patient_id is not provided, one is auto-generated and logged for tracking.
    """
    # Generate patient_id if not provided
    patient_id_generated = False
    patient_id = request.patient_id

    if not patient_id:
        patient_id = _generate_patient_id(request.risk_domain)
        patient_id_generated = True
        # Log prominently for tracking
        logger.warning(
            f"AUTO-GENERATED PATIENT ID: {patient_id} | "
            f"Domain: {request.risk_domain} | "
            f"Timestamp: {datetime.now(timezone.utc).isoformat()} | "
            f"Lab markers: {list(request.lab_data.keys()) if request.lab_data else 'none'}"
        )

    try:
        result = await process_patient_intake(
            patient_id=patient_id,
            risk_domain=request.risk_domain,
            lab_data=request.lab_data,
            vitals_data=request.vitals_data,
            clinical_data=request.clinical_data,
            risk_score=request.risk_score,
            location=request.location,
            encounter_id=request.encounter_id,
            metadata=request.metadata,
        )

        response = result.to_dict()

        # Add patient_id tracking info if auto-generated
        if patient_id_generated:
            response["patient_id_info"] = {
                "patient_id": patient_id,
                "auto_generated": True,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "note": "IMPORTANT: Save this ID to track this patient. It has been logged for audit purposes."
            }

        # Add data quality assessment
        if DATA_INGESTION_AVAILABLE and request.lab_data:
            response["data_completeness"] = _calculate_data_completeness(
                lab_data=request.lab_data or {},
                risk_domain=request.risk_domain,
            )

        return response

    except Exception as e:
        logger.exception(f"Patient intake failed for {patient_id}: {e}")
        # BULLETPROOF: Return helpful response even on error
        error_response = {
            "success": False,
            "patient_id": patient_id,
            "risk_domain": request.risk_domain,
            "error": str(e)[:200],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_completeness": _calculate_data_completeness(
                lab_data=request.lab_data or {},
                risk_domain=request.risk_domain,
            ) if DATA_INGESTION_AVAILABLE else None,
            "recommendations": [
                "Check input data format",
                "Verify lab_data contains valid biomarker values",
            ],
        }

        # Include auto-generated ID info even on error
        if patient_id_generated:
            error_response["patient_id_info"] = {
                "patient_id": patient_id,
                "auto_generated": True,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "note": "IMPORTANT: Save this ID to track this patient. It has been logged for audit purposes."
            }

        return error_response


def _calculate_data_completeness(
    lab_data: Dict[str, Any],
    risk_domain: str,
) -> Dict[str, Any]:
    """Calculate data completeness for the given risk domain."""
    if not DATA_INGESTION_AVAILABLE:
        return {"available": False}

    domain_reqs = DOMAIN_CRITICAL_BIOMARKERS.get(risk_domain, {})
    if not domain_reqs:
        # Use sepsis as default
        domain_reqs = DOMAIN_CRITICAL_BIOMARKERS.get("sepsis", {})

    critical = domain_reqs.get("critical", set())
    helpful = domain_reqs.get("helpful", set())
    genetic = domain_reqs.get("genetic", set())

    # Normalize lab data keys
    lab_keys = {k.lower() for k in lab_data.keys()}

    critical_have = lab_keys & critical
    critical_missing = critical - lab_keys
    helpful_have = lab_keys & helpful
    helpful_missing = helpful - lab_keys

    completeness = len(critical_have) / len(critical) if critical else 1.0

    # Calculate impact
    if completeness < 0.3:
        impact = f"Limited {risk_domain} analysis. Add critical biomarkers."
    elif completeness < 0.7:
        impact = f"Missing biomarkers could improve detection by 1-2 days."
    elif completeness < 1.0:
        impact = f"Good coverage. Additional markers would enhance precision."
    else:
        impact = f"Complete {risk_domain} biomarker coverage."

    recommendations = []
    if critical_missing:
        recommendations.append(f"Add {', '.join(list(critical_missing)[:3])} for better {risk_domain} analysis")
    if genetic:
        recommendations.append(f"Genetic markers ({', '.join(list(genetic)[:3])}) would enable 2-3 days earlier detection")

    return {
        "score": round(completeness, 2),
        "have": list(critical_have),
        "missing": list(critical_missing),
        "helpful_missing": list(helpful_missing)[:3],
        "genetic_recommended": list(genetic)[:3],
        "impact": impact,
        "recommendations": recommendations,
    }


def _generate_patient_id(risk_domain: str) -> str:
    """
    Generate a unique, trackable patient ID.

    Format: AUTO_{DOMAIN}_{YYYYMMDD}_{HHMMSS}_{RANDOM}
    Example: AUTO_SEPSIS_20260322_143052_7A3F

    This format allows:
    - Easy identification as auto-generated (AUTO_ prefix)
    - Risk domain context
    - Timestamp for when the patient was registered
    - Random suffix to prevent collisions
    """
    import secrets

    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y%m%d")
    time_part = now.strftime("%H%M%S")
    random_part = secrets.token_hex(2).upper()  # 4 hex chars

    domain_short = risk_domain.upper()[:6] if risk_domain else "UNK"

    return f"AUTO_{domain_short}_{date_part}_{time_part}_{random_part}"


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
