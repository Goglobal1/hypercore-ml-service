"""
Pharmaceutical API Router for HyperCore

Provides endpoints for:
- Drug profiles and adverse events (FDA FAERS)
- Clinical trials search (AACT)
- Drug response prediction (pharmacogenomics)
- Drug-drug interaction checking
"""

from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, Query

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
