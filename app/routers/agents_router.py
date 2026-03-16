"""
Diagnostic Agents API Router for HyperCore

Provides endpoints for:
- Agent health and status
- Individual agent analysis
- Multi-agent orchestration
- Inter-agent communication
"""

import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
from datetime import datetime, timezone

# Import agents
from app.agents.base_agent import AgentRegistry, AgentType, AgentFinding
from app.agents.biomarker_agent import get_biomarker_agent, BiomarkerAgent
from app.agents.diagnostic_agent import get_diagnostic_agent, DiagnosticAgent
from app.agents.trial_rescue_agent import get_trial_rescue_agent, TrialRescueAgent
from app.agents.surveillance_agent import get_surveillance_agent, SurveillanceAgent

router = APIRouter(prefix="/agents", tags=["diagnostic-agents"])


# Request/Response Models
class BiomarkerAnalysisRequest(BaseModel):
    """Request for biomarker analysis."""
    biomarkers: Dict[str, float] = Field(..., description="Biomarker name-value pairs")
    genes: Optional[List[str]] = Field(None, description="Genes to analyze")
    patient_context: Optional[Dict[str, Any]] = Field(None, description="Patient context")
    include_genomics: bool = Field(True, description="Include genomics analysis")
    correlation_id: Optional[str] = Field(None, description="Session correlation ID")


class DiagnosticAnalysisRequest(BaseModel):
    """Request for diagnostic analysis."""
    clinical_findings: List[str] = Field(..., description="Clinical findings/symptoms")
    biomarkers: Optional[Dict[str, float]] = Field(None, description="Biomarker values")
    vital_signs: Optional[Dict[str, float]] = Field(None, description="Vital signs")
    history: Optional[Dict[str, Any]] = Field(None, description="Patient history")
    correlation_id: Optional[str] = Field(None, description="Session correlation ID")


class TrialAnalysisRequest(BaseModel):
    """Request for trial/rescue analysis."""
    diagnosis: Optional[str] = Field(None, description="Primary diagnosis")
    current_medications: Optional[List[str]] = Field(None, description="Current medications")
    failed_therapies: Optional[List[str]] = Field(None, description="Failed therapies")
    patient_profile: Optional[Dict[str, Any]] = Field(None, description="Patient profile")
    treatment_response: Optional[str] = Field("unknown", description="Treatment response status")
    correlation_id: Optional[str] = Field(None, description="Session correlation ID")


class SurveillanceAnalysisRequest(BaseModel):
    """Request for surveillance analysis."""
    region: Optional[str] = Field(None, description="Geographic region")
    pathogens: Optional[List[str]] = Field(None, description="Pathogens to analyze")
    time_period: Optional[Dict[str, int]] = Field(None, description="Time period")
    include_amr: bool = Field(True, description="Include AMR analysis")
    include_vaccination: bool = Field(True, description="Include vaccination data")
    patient_location: Optional[str] = Field(None, description="Patient location/country")
    correlation_id: Optional[str] = Field(None, description="Session correlation ID")


class MultiAgentRequest(BaseModel):
    """Request for multi-agent orchestrated analysis."""
    patient_data: Dict[str, Any] = Field(..., description="Complete patient data")
    agents: Optional[List[str]] = Field(None, description="Specific agents to use (default: all)")
    correlation_id: Optional[str] = Field(None, description="Session correlation ID")


# Health and Status Endpoints
@router.get("/health")
async def agents_health():
    """Check health of all diagnostic agents."""
    agents = {
        "biomarker": get_biomarker_agent(),
        "diagnostic": get_diagnostic_agent(),
        "trial_rescue": get_trial_rescue_agent(),
        "surveillance": get_surveillance_agent(),
    }

    statuses = {}
    for name, agent in agents.items():
        try:
            statuses[name] = agent.get_status()
        except Exception as e:
            statuses[name] = {"status": "error", "error": str(e)}

    return {
        "status": "healthy",
        "module": "diagnostic_agents",
        "version": "1.0.0",
        "agents": statuses,
        "registered_agents": [t.value for t in AgentRegistry.list_agents()],
    }


@router.get("/list")
async def list_agents():
    """List all available diagnostic agents."""
    agents = [
        {
            "type": "biomarker",
            "name": "Biomarker Signal Interpreter",
            "description": "Interprets biomarker signals across multi-omic data sources",
            "capabilities": ["biomarker_interpretation", "expression_pattern_detection", "variant_phenotype_correlation"],
        },
        {
            "type": "diagnostic",
            "name": "Differential Reasoning Engine",
            "description": "Generates and ranks differential diagnoses based on multi-source evidence",
            "capabilities": ["differential_diagnosis", "hypothesis_ranking", "test_recommendation"],
        },
        {
            "type": "trial_rescue",
            "name": "Trial Rescue Intelligence",
            "description": "Matches patients to trials and identifies rescue therapies",
            "capabilities": ["trial_matching", "rescue_therapy_identification", "pharmacogenomic_assessment"],
        },
        {
            "type": "surveillance",
            "name": "Population Surveillance Intelligence",
            "description": "Detects population-level anomalies and outbreak patterns",
            "capabilities": ["outbreak_detection", "amr_trend_analysis", "regional_risk_assessment"],
        },
    ]

    return {
        "total_agents": len(agents),
        "agents": agents,
    }


@router.get("/status/{agent_type}")
async def get_agent_status(agent_type: str):
    """Get status of a specific agent."""
    agent_map = {
        "biomarker": get_biomarker_agent,
        "diagnostic": get_diagnostic_agent,
        "trial_rescue": get_trial_rescue_agent,
        "surveillance": get_surveillance_agent,
    }

    if agent_type not in agent_map:
        raise HTTPException(status_code=404, detail=f"Unknown agent type: {agent_type}")

    try:
        agent = agent_map[agent_type]()
        return agent.get_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent status: {str(e)}")


# Individual Agent Analysis Endpoints
@router.post("/biomarker/analyze")
async def analyze_biomarkers(request: BiomarkerAnalysisRequest):
    """
    Run biomarker signal interpretation analysis.

    Analyzes biomarker levels, gene expression, and variants
    to generate clinical findings.
    """
    try:
        agent = get_biomarker_agent()

        # Generate correlation ID if not provided
        correlation_id = request.correlation_id or f"bio_{uuid.uuid4().hex[:8]}"

        result = await agent.analyze({
            "biomarkers": request.biomarkers,
            "genes": request.genes or [],
            "patient_context": request.patient_context or {},
            "include_genomics": request.include_genomics,
            "correlation_id": correlation_id,
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Biomarker analysis error: {str(e)}")


@router.post("/diagnostic/analyze")
async def analyze_diagnostic(request: DiagnosticAnalysisRequest):
    """
    Run differential diagnosis analysis.

    Generates ranked differential diagnoses based on
    clinical findings, biomarkers, and vital signs.
    """
    try:
        agent = get_diagnostic_agent()

        correlation_id = request.correlation_id or f"diag_{uuid.uuid4().hex[:8]}"

        result = await agent.analyze({
            "clinical_findings": request.clinical_findings,
            "biomarkers": request.biomarkers or {},
            "vital_signs": request.vital_signs or {},
            "history": request.history or {},
            "correlation_id": correlation_id,
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic analysis error: {str(e)}")


@router.post("/trial-rescue/analyze")
async def analyze_trials(request: TrialAnalysisRequest):
    """
    Run trial matching and rescue therapy analysis.

    Identifies matching clinical trials and rescue options
    for treatment failures.
    """
    try:
        agent = get_trial_rescue_agent()

        correlation_id = request.correlation_id or f"trial_{uuid.uuid4().hex[:8]}"

        result = await agent.analyze({
            "diagnosis": request.diagnosis,
            "current_medications": request.current_medications or [],
            "failed_therapies": request.failed_therapies or [],
            "patient_profile": request.patient_profile or {},
            "treatment_response": request.treatment_response,
            "correlation_id": correlation_id,
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trial analysis error: {str(e)}")


@router.post("/surveillance/analyze")
async def analyze_surveillance(request: SurveillanceAnalysisRequest):
    """
    Run population surveillance analysis.

    Detects outbreaks, AMR trends, and population health anomalies.
    """
    try:
        agent = get_surveillance_agent()

        correlation_id = request.correlation_id or f"surv_{uuid.uuid4().hex[:8]}"

        result = await agent.analyze({
            "region": request.region,
            "pathogens": request.pathogens or [],
            "time_period": request.time_period or {"years": 5},
            "include_amr": request.include_amr,
            "include_vaccination": request.include_vaccination,
            "patient_location": request.patient_location,
            "correlation_id": correlation_id,
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Surveillance analysis error: {str(e)}")


# Multi-Agent Orchestration
@router.post("/orchestrate")
async def orchestrate_analysis(request: MultiAgentRequest):
    """
    Run orchestrated multi-agent analysis.

    Coordinates all agents to analyze patient data and
    share findings between agents.
    """
    try:
        correlation_id = request.correlation_id or f"orch_{uuid.uuid4().hex[:8]}"
        patient_data = request.patient_data

        # Clear previous session findings
        AgentRegistry.clear_findings()

        results = {
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agents_executed": [],
            "combined_findings": [],
            "consensus": None,
        }

        # Determine which agents to run
        agents_to_run = request.agents or ["biomarker", "diagnostic", "trial_rescue", "surveillance"]

        # 1. Run Biomarker Agent first (provides evidence for others)
        if "biomarker" in agents_to_run:
            biomarker_agent = get_biomarker_agent()
            biomarker_result = await biomarker_agent.analyze({
                "biomarkers": patient_data.get("biomarkers", {}),
                "genes": patient_data.get("genes", []),
                "patient_context": patient_data.get("patient_context", {}),
                "include_genomics": True,
                "correlation_id": correlation_id,
            })
            results["biomarker_analysis"] = biomarker_result
            results["agents_executed"].append("biomarker")

        # 2. Run Diagnostic Agent (uses biomarker findings)
        if "diagnostic" in agents_to_run:
            diagnostic_agent = get_diagnostic_agent()
            diagnostic_result = await diagnostic_agent.analyze({
                "clinical_findings": patient_data.get("clinical_findings", []),
                "biomarkers": patient_data.get("biomarkers", {}),
                "vital_signs": patient_data.get("vital_signs", {}),
                "history": patient_data.get("history", {}),
                "correlation_id": correlation_id,
            })
            results["diagnostic_analysis"] = diagnostic_result
            results["agents_executed"].append("diagnostic")

        # 3. Run Trial Rescue Agent (uses diagnostic findings)
        if "trial_rescue" in agents_to_run:
            trial_agent = get_trial_rescue_agent()
            trial_result = await trial_agent.analyze({
                "diagnosis": patient_data.get("diagnosis"),
                "current_medications": patient_data.get("medications", []),
                "failed_therapies": patient_data.get("failed_therapies", []),
                "patient_profile": patient_data.get("patient_profile", {}),
                "treatment_response": patient_data.get("treatment_response", "unknown"),
                "correlation_id": correlation_id,
            })
            results["trial_analysis"] = trial_result
            results["agents_executed"].append("trial_rescue")

        # 4. Run Surveillance Agent (provides population context)
        if "surveillance" in agents_to_run:
            surveillance_agent = get_surveillance_agent()
            surveillance_result = await surveillance_agent.analyze({
                "region": patient_data.get("region"),
                "pathogens": patient_data.get("pathogens", []),
                "time_period": {"years": 5},
                "include_amr": True,
                "include_vaccination": True,
                "patient_location": patient_data.get("location"),
                "correlation_id": correlation_id,
            })
            results["surveillance_analysis"] = surveillance_result
            results["agents_executed"].append("surveillance")

        # 5. Collect all shared findings
        all_findings = AgentRegistry.get_shared_findings()
        results["combined_findings"] = [f.to_dict() for f in all_findings]
        results["total_findings"] = len(all_findings)

        # 6. Generate consensus summary
        high_confidence = [f for f in all_findings if f.confidence >= 0.7]
        critical = [f for f in all_findings if "critical" in f.category.lower()]

        results["consensus"] = {
            "high_confidence_findings": len(high_confidence),
            "critical_findings": len(critical),
            "top_findings": [f.to_dict() for f in sorted(all_findings, key=lambda x: x.confidence, reverse=True)[:5]],
            "agents_agreed": len(set(f.agent_type.value for f in high_confidence)),
        }

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration error: {str(e)}")


# Utility Endpoints
@router.get("/findings")
async def get_shared_findings(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results")
):
    """Get shared findings across agents."""
    agent_type_enum = None
    if agent_type:
        try:
            agent_type_enum = AgentType(agent_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown agent type: {agent_type}")

    findings = AgentRegistry.get_shared_findings(
        agent_type=agent_type_enum,
        min_confidence=min_confidence
    )

    return {
        "findings": [f.to_dict() for f in findings[:limit]],
        "total": len(findings),
        "filter": {
            "agent_type": agent_type,
            "min_confidence": min_confidence,
        }
    }


@router.delete("/findings")
async def clear_findings():
    """Clear all shared findings (start new session)."""
    AgentRegistry.clear_findings()
    return {"status": "cleared", "message": "All shared findings cleared"}


@router.get("/diagnostic/diagnoses")
async def list_diagnoses():
    """List all available diagnostic patterns."""
    agent = get_diagnostic_agent()
    return {
        "diagnoses": agent.list_diagnoses(),
        "total": len(agent.list_diagnoses())
    }


@router.get("/diagnostic/diagnoses/{diagnosis}")
async def get_diagnosis_info(diagnosis: str):
    """Get information about a specific diagnosis."""
    agent = get_diagnostic_agent()
    return agent.get_diagnosis_info(diagnosis)


@router.get("/biomarker/panels")
async def list_biomarker_panels():
    """List available biomarker panels."""
    from app.agents.biomarker_agent import BIOMARKER_REFERENCE
    return {
        "panels": list(BIOMARKER_REFERENCE.keys()),
        "total": len(BIOMARKER_REFERENCE)
    }


@router.get("/biomarker/panels/{category}")
async def get_biomarker_panel(category: str):
    """Get biomarker reference panel for a category."""
    agent = get_biomarker_agent()
    return agent.get_biomarker_panel(category)


@router.get("/trial-rescue/categories")
async def list_rescue_categories():
    """List available rescue therapy categories."""
    agent = get_trial_rescue_agent()
    return {
        "categories": agent.list_rescue_categories(),
        "total": len(agent.list_rescue_categories())
    }


@router.get("/surveillance/regions")
async def list_regions():
    """List available regional profiles."""
    agent = get_surveillance_agent()
    return {
        "regions": agent.list_regions(),
        "total": len(agent.list_regions())
    }


@router.get("/surveillance/regions/{region}")
async def get_regional_profile(region: str):
    """Get risk profile for a region."""
    agent = get_surveillance_agent()
    return agent.get_regional_profile(region)
