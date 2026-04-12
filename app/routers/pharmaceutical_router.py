"""
Pharmaceutical API Router for HyperCore

Provides endpoints for:
- Drug profiles and adverse events (FDA FAERS)
- Clinical trials search (AACT)
- Drug response prediction (pharmacogenomics)
- Drug-drug interaction checking
- Trial Rescue Engine (Pharma Mode)
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import io
import logging
import math

logger = logging.getLogger(__name__)


def _sanitize_value(v):
    """Sanitize a value for JSON serialization (handle NaN/Inf and numpy types)."""
    if v is None:
        return None
    # Handle numpy arrays
    if isinstance(v, np.ndarray):
        return [_sanitize_value(item) for item in v.tolist()]
    # Handle numpy scalar types
    if isinstance(v, (np.floating, np.integer)):
        val = float(v)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(v, np.bool_):
        return bool(v)
    # Handle Python float
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    # Handle dict
    if isinstance(v, dict):
        return {str(k): _sanitize_value(val) for k, val in v.items()}
    # Handle list/tuple
    if isinstance(v, (list, tuple)):
        return [_sanitize_value(item) for item in v]
    # Handle pandas types
    if hasattr(v, 'item'):  # numpy scalar method
        try:
            return _sanitize_value(v.item())
        except (ValueError, AttributeError):
            pass
    return v

from app.models.pharmaceutical_models import (
    DrugProfile,
    AdverseEvent,
    AdverseEventResponse,
    ClinicalTrial,
    DrugResponseRequest,
    DrugResponsePrediction,
    InteractionCheckRequest,
    InteractionCheckResponse,
    DrugInteraction,
    TrialSearchRequest,
    TrialSearchResponse,
    MetabolizerStatus,
)
from app.core.drug_response_predictor import (
    get_drug_profile,
    predict_drug_response,
    check_drug_interactions,
    get_adverse_events,
    search_trials,
    get_data_status,
    get_predictor,
    PHARMACOGENOMIC_MAP,
)

router = APIRouter(prefix="/pharma", tags=["pharmaceutical"])


@router.get("/health")
async def pharma_health():
    """Check pharmaceutical module health and data availability."""
    status = get_data_status()

    return {
        "status": "healthy",
        "module": "drug_response_predictor",
        "version": "1.0.0",
        **status
    }


@router.get("/drug/{drug_name}")
async def get_drug(drug_name: str):
    """
    Get comprehensive drug profile.

    Returns:
    - Pharmacogenomic gene associations
    - Common adverse events from FAERS
    - Clinical trial information
    - Known drug interactions
    """
    try:
        profile = get_drug_profile(drug_name)
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching drug profile: {str(e)}")


@router.get("/adverse-events/{drug_name}")
async def get_drug_adverse_events(
    drug_name: str,
    max_quarters: int = Query(4, ge=1, le=10, description="Number of recent quarters to search")
):
    """
    Get adverse events for a drug from FDA FAERS database.

    Returns adverse event reports including:
    - Top reactions by frequency
    - Total report count
    - Quarters searched
    """
    try:
        predictor = get_predictor()
        result = predictor.faers.search_drug(drug_name, max_quarters)

        return AdverseEventResponse(
            drug_name=drug_name,
            total_reports=result.get("total_reports", 0),
            adverse_events=[
                AdverseEvent(
                    primary_id=e.get("primaryid", ""),
                    case_id=e.get("primaryid", ""),
                    drug_name=drug_name,
                    reaction=e.get("reaction", ""),
                )
                for e in result.get("all_events", [])[:100]
            ],
            event_statistics={
                e["reaction"]: e["count"]
                for e in result.get("top_adverse_events", [])
            },
            serious_event_percentage=0.0,  # Would need outcome data
            common_outcomes={}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching adverse events: {str(e)}")


@router.get("/trials/{condition}")
async def search_trials_by_condition(
    condition: str,
    limit: int = Query(50, ge=1, le=200)
):
    """
    Search clinical trials by condition/disease.

    Returns trials from ClinicalTrials.gov AACT database.
    """
    try:
        result = search_trials(condition=condition, limit=limit)
        return TrialSearchResponse(
            query={"condition": condition},
            total_trials=result.get("count", 0),
            trials=[
                ClinicalTrial(
                    nct_id=t.get("nct_id", ""),
                    conditions=[t.get("condition", "")],
                    interventions=t.get("interventions", [])
                )
                for t in result.get("trials", [])
            ],
            phase_distribution={},
            status_distribution={}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching trials: {str(e)}")


@router.post("/predict-response")
async def predict_response(request: DrugResponseRequest):
    """
    Predict drug response based on pharmacogenomics.

    Takes into account:
    - Patient metabolizer status (CYP2D6, CYP2C19, etc.)
    - Genetic variants affecting drug response
    - Concurrent medications (interaction check)

    Returns efficacy and toxicity predictions with recommendations.
    """
    try:
        # Convert metabolizer status to dict of strings
        metabolizer_dict = None
        if request.metabolizer_status:
            metabolizer_dict = {
                gene: status.value
                for gene, status in request.metabolizer_status.items()
            }

        result = predict_drug_response(
            drug_name=request.drug_name,
            metabolizer_status=metabolizer_dict,
            patient_genes=request.patient_genes,
            concurrent_medications=request.concurrent_medications
        )

        return DrugResponsePrediction(
            drug_name=result["drug_name"],
            efficacy_prediction=result["efficacy_prediction"],
            efficacy_confidence=result["efficacy_confidence"],
            toxicity_risk=result["toxicity_risk"],
            toxicity_confidence=result["toxicity_confidence"],
            dose_adjustment=result.get("dose_adjustment"),
            pharmacogenomic_factors=result.get("pharmacogenomic_factors", []),
            interaction_warnings=result.get("interaction_warnings", []),
            recommendations=result.get("recommendations", []),
            evidence_sources=result.get("evidence_sources", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting response: {str(e)}")


@router.post("/interaction-check")
async def check_interactions(request: InteractionCheckRequest):
    """
    Check for drug-drug interactions.

    Analyzes all pairwise combinations of provided drugs
    and returns any known interactions with severity levels.
    """
    if len(request.drugs) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 drugs required for interaction check"
        )

    if len(request.drugs) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 drugs per interaction check"
        )

    try:
        result = check_drug_interactions(request.drugs)

        interactions = [
            DrugInteraction(
                drug_a=i["drug_a"],
                drug_b=i["drug_b"],
                interaction_type=i.get("type", "unknown"),
                severity=i.get("severity", "unknown"),
                mechanism=None,
                clinical_effect=i.get("effect"),
                management=i.get("management")
            )
            for i in result.get("interactions", [])
        ]

        return InteractionCheckResponse(
            drugs_checked=result["drugs_checked"],
            interactions_found=interactions,
            interaction_count=result["interaction_count"],
            max_severity=result["max_severity"],
            recommendations=result.get("recommendations", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking interactions: {str(e)}")


@router.get("/pharmacogenomics/{gene}")
async def get_pharmacogenomics(gene: str):
    """
    Get pharmacogenomic information for a gene.

    Returns drugs affected by variants in this gene
    and the expected impact on drug response.
    """
    gene_upper = gene.upper()

    if gene_upper in PHARMACOGENOMIC_MAP:
        info = PHARMACOGENOMIC_MAP[gene_upper]
        return {
            "gene": gene_upper,
            "affected_drugs": info["drugs"],
            "effect_type": info["effect"],
            "poor_metabolizer_impact": info.get("poor_metabolizer_impact"),
            "variant_impact": info.get("variant_impact"),
            "clinical_recommendation": _get_gene_recommendation(gene_upper)
        }
    else:
        return {
            "gene": gene_upper,
            "affected_drugs": [],
            "note": f"No pharmacogenomic data available for {gene_upper}"
        }


def _get_gene_recommendation(gene: str) -> str:
    """Get clinical recommendation for a pharmacogene."""
    recommendations = {
        "CYP2D6": "Test before prescribing codeine, tamoxifen, or tricyclic antidepressants",
        "CYP2C19": "Test before prescribing clopidogrel or PPIs long-term",
        "CYP2C9": "Test before prescribing warfarin, consider VKORC1 as well",
        "VKORC1": "Test together with CYP2C9 before warfarin therapy",
        "SLCO1B1": "Test before high-dose statin therapy, especially simvastatin",
        "HLA-B*57:01": "Mandatory testing before abacavir therapy",
        "HLA-B*15:02": "Test in Asian populations before carbamazepine",
        "TPMT": "Test before thiopurine therapy (azathioprine, mercaptopurine)",
        "UGT1A1": "Test before irinotecan therapy",
        "DPYD": "Test before fluoropyrimidine therapy",
        "G6PD": "Test before oxidative stress-inducing drugs",
    }
    return recommendations.get(gene, "Consult pharmacogenomics guidelines")


@router.get("/genes")
async def list_pharmacogenes():
    """List all pharmacogenomic genes in the database."""
    genes = []
    for gene, info in PHARMACOGENOMIC_MAP.items():
        genes.append({
            "gene": gene,
            "drug_count": len(info["drugs"]),
            "effect_type": info["effect"],
            "sample_drugs": info["drugs"][:3]
        })

    return {
        "total_genes": len(genes),
        "genes": genes
    }


@router.get("/trials-by-drug/{drug_name}")
async def search_trials_by_drug(
    drug_name: str,
    limit: int = Query(50, ge=1, le=200)
):
    """
    Search clinical trials by drug/intervention name.

    Returns trials from ClinicalTrials.gov AACT database.
    """
    try:
        result = search_trials(drug=drug_name, limit=limit)
        return {
            "query": drug_name,
            "total_trials": result.get("count", 0),
            "trials": result.get("trials", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching trials: {str(e)}")


@router.get("/faers-quarters")
async def list_faers_quarters():
    """List available FAERS data quarters."""
    predictor = get_predictor()
    quarters = predictor.faers.list_available_quarters()

    return {
        "total_quarters": len(quarters),
        "quarters": quarters,
        "date_range": f"{quarters[0]} to {quarters[-1]}" if quarters else "No data"
    }

# ============================================================================
# TRIAL RESCUE ENGINE
# ============================================================================

# Import Trial Rescue Engine
try:
    from app.modes.pharma import (
        get_trial_rescue_engine,
        TrialRescueInput,
    )
    TRIAL_RESCUE_AVAILABLE = True
except ImportError:
    TRIAL_RESCUE_AVAILABLE = False
    logger.warning("Trial Rescue Engine not available")


class TrialRescueRequest(BaseModel):
    """Request model for trial rescue analysis."""
    trial_data: str = Field(..., description="CSV string with trial data")
    treatment_column: str = Field(default="treatment", description="Column name for treatment assignment (0/1)")
    outcome_column: str = Field(default="outcome", description="Column name for outcome variable")
    patient_id_column: str = Field(default="patient_id", description="Column name for patient IDs")
    trial_name: Optional[str] = Field(default=None, description="Name of the trial")
    sponsor: Optional[str] = Field(default=None, description="Trial sponsor")
    phase: Optional[str] = Field(default=None, description="Trial phase (1, 2, 3)")
    indication: Optional[str] = Field(default=None, description="Target indication")
    base_asset_value_usd: float = Field(default=100_000_000, description="Base asset value in USD")


@router.post("/trial-rescue")
async def analyze_trial_rescue(request: TrialRescueRequest):
    """
    Analyze a clinical trial for rescue opportunities.

    The Trial Rescue Engine finds hidden value in "failed" trials through:
    1. Subgroup Discovery - Find hidden responder populations
    2. Confounder Detection - Identify masking variables
    3. Endpoint Reinterpretation - Find alternative endpoints with effect
    4. Rescue Strategy Generation - Rank by UTILITY (not p-value!)
    5. Asset Valuation - Calculate potential value with hard escalation flags

    CRITICAL: Opportunities are ranked by handler_score + net_utility + asset_value,
    NOT by p-value alone.

    Hard escalation is triggered when estimated_asset_value_usd >= $1,000,000,000.
    """
    import traceback

    if not TRIAL_RESCUE_AVAILABLE:
        return {"success": False, "error": "Trial Rescue Engine not available"}

    try:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(request.trial_data))

        # Validate required columns
        required_cols = [request.treatment_column, request.outcome_column]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}"
            )

        # Create input
        rescue_input = TrialRescueInput(
            trial_data=df,
            treatment_column=request.treatment_column,
            outcome_column=request.outcome_column,
            patient_id_column=request.patient_id_column,
            trial_name=request.trial_name,
            sponsor=request.sponsor,
            phase=request.phase,
            indication=request.indication,
            base_asset_value_usd=request.base_asset_value_usd,
        )

        # Run analysis
        engine = get_trial_rescue_engine()
        result = engine.analyze(rescue_input)

        # DEBUG: Return minimal response first
        return {
            "success": result.success,
            "trial_name": result.trial_name,
            "engine_version": result.engine_version,
            "subgroups_found": result.subgroups_found,
            "confounders_found": result.confounders_found,
            "alternative_endpoints_found": result.alternative_endpoints_found,
            "debug": "minimal_response",
        }

        # Convert to dict for response (sanitize NaN/Inf values)
        response = {
            "success": result.success,
            "trial_name": result.trial_name,
            "timestamp": result.timestamp,
            "engine_version": result.engine_version,
            "utility_gate_mode": result.utility_gate_mode,
            "subgroups_found": result.subgroups_found,
            "confounders_found": result.confounders_found,
            "alternative_endpoints_found": result.alternative_endpoints_found,
            "surfaced_opportunities": len(result.surfaced_opportunities),
            "suppressed_opportunities": len(result.suppressed_opportunities),
            "hard_escalation_triggered": result.hard_escalation_triggered,
            "estimated_total_asset_value_usd": _sanitize_value(result.estimated_total_asset_value_usd),
            "top_opportunities": [
                {
                    "opportunity_id": o.opportunity_id,
                    "opportunity_type": o.opportunity_type,
                    "title": o.title,
                    "description": o.description,
                    "confidence": _sanitize_value(o.confidence),
                    "effect_size": _sanitize_value(o.effect_size),
                    "estimated_asset_value_usd": _sanitize_value(o.estimated_asset_value_usd),
                    "hard_escalation_flag": o.hard_escalation_flag,
                    "utility_decision": o.utility_decision,
                    "utility_breakdown": _sanitize_value(o.utility_breakdown),
                    "recommended_actions": o.recommended_actions,
                    "regulatory_pathway": o.regulatory_pathway,
                    "evidence": o.evidence,
                    "pvalue": _sanitize_value(o.pvalue),
                }
                for o in result.surfaced_opportunities[:10]
            ],
            "subgroups": [
                {
                    "subgroup_id": s.subgroup_id,
                    "subgroup_name": s.subgroup_name,
                    "n_patients": s.n_patients,
                    "percentage_of_total": _sanitize_value(s.percentage_of_total),
                    "response_rate": _sanitize_value(s.response_rate),
                    "relative_improvement": _sanitize_value(s.relative_response_improvement),
                    "effect_size": _sanitize_value(s.effect_size),
                    "pvalue": _sanitize_value(s.pvalue),
                    "biological_plausibility": _sanitize_value(s.biological_plausibility),
                    "defining_features": _sanitize_value(s.defining_features),
                }
                for s in result.subgroups[:10]
            ],
            "confounders": [
                {
                    "variable_name": c.variable_name,
                    "confounder_type": c.confounder_type,
                    "impact_score": _sanitize_value(c.impact_score),
                    "unadjusted_effect": _sanitize_value(c.unadjusted_effect),
                    "adjusted_effect": _sanitize_value(c.adjusted_effect),
                    "effect_change_percentage": _sanitize_value(c.effect_change_percentage),
                    "adjustment_method": c.adjustment_method,
                    "recommendation": c.recommendation,
                }
                for c in result.confounders[:10]
            ],
            "alternative_endpoints": [
                {
                    "endpoint_name": e.endpoint_name,
                    "endpoint_type": e.endpoint_type,
                    "effect_size": _sanitize_value(e.effect_size),
                    "pvalue": _sanitize_value(e.pvalue),
                    "improvement_over_primary": _sanitize_value(e.improvement_over_primary),
                    "clinical_relevance": _sanitize_value(e.clinical_relevance),
                    "regulatory_acceptability": _sanitize_value(e.regulatory_acceptability),
                }
                for e in result.alternative_endpoints[:10]
            ],
            "report": {
                "executive_summary": result.report.executive_summary if result.report else None,
                "primary_recommendation": result.report.primary_recommendation if result.report else None,
                "secondary_recommendations": result.report.secondary_recommendations if result.report else [],
                "regulatory_considerations": result.report.regulatory_considerations if result.report else [],
            } if result.report else None,
            "metadata": _sanitize_value(result.metadata),
        }
        return response

    except HTTPException:
        raise
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Trial rescue analysis failed: {e}\n{error_traceback}")
        # Return error details for debugging
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_traceback.split('\n')[-5:],  # Last 5 lines of traceback
            "trial_name": request.trial_name or "Unknown",
            "engine_version": "1.0.4",
        }


