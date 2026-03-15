"""
Pathogen Detection API Router for HyperCore

Provides endpoints for:
- Pathogen information lookup
- WHO surveillance data search
- Outbreak detection and alerts
- Antimicrobial resistance (AMR) analysis
- Vaccination coverage monitoring
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from app.models.pathogen_models import (
    PathogenType,
    AlertLevel,
    WHOIndicator,
    OutbreakAlert,
    PathogenSearchRequest,
    PathogenSearchResponse,
    OutbreakDetectionRequest,
    OutbreakDetectionResponse,
    AMRAnalysisRequest,
    AMRAnalysisResponse,
    AMRProfile,
)
from app.core.pathogen_detection import (
    get_pathogen_info,
    get_disease_pathogens,
    detect_outbreaks,
    analyze_amr,
    get_vaccination_coverage,
    search_surveillance,
    get_data_status,
    PATHOGEN_DATABASE,
    DISEASE_PATHOGEN_MAP,
)

router = APIRouter(prefix="/pathogen", tags=["pathogen-detection"])


@router.get("/health")
async def pathogen_health():
    """Check pathogen detection module health and data availability."""
    status = get_data_status()

    return {
        "status": "healthy",
        "module": "pathogen_detection",
        "version": "1.0.0",
        **status
    }


@router.get("/info/{pathogen}")
async def get_pathogen(pathogen: str):
    """
    Get detailed information about a pathogen.

    Returns:
    - Pathogen type (bacterial, viral, fungal, parasitic)
    - Associated diseases
    - AMR concerns and key resistances
    - ICD-10 codes
    """
    try:
        info = get_pathogen_info(pathogen)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching pathogen info: {str(e)}")


@router.get("/disease/{disease}")
async def get_pathogens_by_disease(disease: str):
    """
    Get pathogens associated with a disease.

    Returns list of pathogens that commonly cause the specified disease.
    """
    try:
        pathogens = get_disease_pathogens(disease)
        return {
            "disease": disease,
            "pathogens": pathogens,
            "count": len(pathogens)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching disease pathogens: {str(e)}")


@router.get("/list")
async def list_pathogens():
    """List all pathogens in the database."""
    pathogens = []
    for key, info in PATHOGEN_DATABASE.items():
        pathogens.append({
            "pathogen_id": key,
            "common_name": info["common_name"],
            "type": info["type"],
            "amr_concern": info.get("amr_concern", False),
            "disease_count": len(info.get("diseases", []))
        })

    return {
        "total_pathogens": len(pathogens),
        "pathogens": pathogens
    }


@router.get("/diseases")
async def list_diseases():
    """List all disease-pathogen mappings."""
    diseases = []
    for disease, pathogen_list in DISEASE_PATHOGEN_MAP.items():
        diseases.append({
            "disease": disease,
            "pathogen_count": len(pathogen_list),
            "pathogens": pathogen_list
        })

    return {
        "total_diseases": len(diseases),
        "diseases": diseases
    }


@router.post("/outbreak-detection")
async def outbreak_detection(request: OutbreakDetectionRequest):
    """
    Detect potential outbreaks based on surveillance data.

    Uses statistical analysis to identify unusual increases
    in disease indicators compared to historical baselines.

    Returns alerts with severity levels:
    - critical: >3 standard deviations above mean
    - high: >2 standard deviations
    - moderate: >1.5 standard deviations
    - low: above threshold but <1.5 std
    """
    try:
        result = detect_outbreaks(
            regions=request.regions,
            threshold_multiplier=request.threshold_multiplier,
            lookback_years=request.lookback_years
        )

        # Convert to response model
        alerts = []
        for a in result.get("alerts", []):
            alerts.append(OutbreakAlert(
                pathogen="unknown",  # From surveillance, pathogen not always specified
                pathogen_type=PathogenType.UNKNOWN,
                alert_level=AlertLevel(a["alert_level"]),
                affected_regions=[a["country"]],
                case_count=0,  # Not available from indicators
                trend=a["trend"],
                detection_date=None,  # Would need current timestamp
                confidence=a["confidence"],
                evidence=[f"Indicator: {a['indicator']}", f"Value: {a['current_value']}"],
                recommendations=["Investigate local data", "Confirm with laboratory testing"]
            ))

        return OutbreakDetectionResponse(
            alerts=alerts,
            total_alerts=result["total_alerts"],
            critical_alerts=result["critical_alerts"],
            regions_analyzed=result["regions_analyzed"],
            analysis_period=result["analysis_period"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in outbreak detection: {str(e)}")


@router.get("/outbreak-quick")
async def quick_outbreak_check(
    region: Optional[str] = Query(None, description="Region to analyze"),
    threshold: float = Query(1.5, ge=1.0, le=5.0, description="Alert threshold (std devs)"),
    years: int = Query(5, ge=1, le=20, description="Lookback years")
):
    """Quick outbreak detection check with simple parameters."""
    try:
        regions = [region] if region else None
        result = detect_outbreaks(
            regions=regions,
            threshold_multiplier=threshold,
            lookback_years=years
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in outbreak check: {str(e)}")


@router.post("/amr-analysis")
async def amr_analysis(request: AMRAnalysisRequest):
    """
    Analyze antimicrobial resistance patterns.

    Returns:
    - AMR surveillance data from WHO
    - High-risk pathogen-antibiotic combinations
    - Clinical recommendations
    """
    try:
        result = analyze_amr(
            pathogen=request.pathogen,
            antibiotic=request.antibiotic,
            country=request.country
        )

        # Convert to response model
        profiles = []
        for p in result.get("high_risk_pathogens", []):
            for resistance in p.get("key_resistances", []):
                profiles.append(AMRProfile(
                    pathogen=p["pathogen"],
                    antibiotic=resistance,
                    resistance_rate=0.0,  # Would need actual data
                    sample_size=0,
                    country=request.country or "global",
                    year=2024,
                    trend="monitoring",
                    alert_level=AlertLevel.HIGH if p.get("amr_concern") else AlertLevel.LOW
                ))

        return AMRAnalysisResponse(
            profiles=profiles,
            high_risk_combinations=[
                {"pathogen": p["pathogen"], "resistances": p["key_resistances"]}
                for p in result.get("high_risk_pathogens", [])
            ],
            trend_summary={"overall": "requires_monitoring"},
            recommendations=result.get("recommendations", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in AMR analysis: {str(e)}")


@router.get("/amr/{pathogen}")
async def get_amr_for_pathogen(
    pathogen: str,
    country: Optional[str] = Query(None, description="Filter by country")
):
    """Get AMR information for a specific pathogen."""
    try:
        result = analyze_amr(pathogen=pathogen, country=country)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching AMR data: {str(e)}")


@router.get("/vaccination")
async def get_vaccination(
    disease: Optional[str] = Query(None, description="Filter by disease"),
    country: Optional[str] = Query(None, description="Filter by country")
):
    """
    Get vaccination coverage data.

    Currently supports COVID-19 vaccination data from WHO.
    """
    try:
        result = get_vaccination_coverage(disease=disease, country=country)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching vaccination data: {str(e)}")


@router.post("/search")
async def search_pathogens(request: PathogenSearchRequest):
    """
    Search WHO surveillance indicators.

    Supports filtering by:
    - Pathogen/disease name
    - Country
    - Year range
    - Indicator type
    """
    try:
        result = search_surveillance(
            indicator_name=request.pathogen_name or request.disease,
            country=request.country,
            year_from=request.year_from,
            year_to=request.year_to,
            limit=request.limit
        )

        # Convert to response model
        indicators = []
        for r in result.get("results", []):
            indicators.append(WHOIndicator(
                indicator_id=r.get("indicator_id", ""),
                indicator_code=r.get("indicator_code", ""),
                indicator_name=r.get("indicator_name", ""),
                country=r.get("country", ""),
                country_code=r.get("country_code"),
                year=r.get("year", 0),
                value=r.get("value", 0.0),
                lower_bound=r.get("lower_bound"),
                upper_bound=r.get("upper_bound"),
                category=r.get("category")
            ))

        # Get unique countries and years
        countries = list(set(i.country for i in indicators if i.country))
        years = sorted(set(i.year for i in indicators if i.year))

        return PathogenSearchResponse(
            query=result.get("query", {}),
            indicators=indicators,
            total_results=result.get("count", 0),
            countries_affected=countries,
            year_range=years
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching surveillance data: {str(e)}")


@router.get("/surveillance")
async def search_surveillance_get(
    indicator: Optional[str] = Query(None, description="Indicator name filter"),
    country: Optional[str] = Query(None, description="Country filter"),
    year_from: Optional[int] = Query(None, ge=1990, le=2030, description="Start year"),
    year_to: Optional[int] = Query(None, ge=1990, le=2030, description="End year"),
    limit: int = Query(100, ge=1, le=1000, description="Max results")
):
    """Search WHO surveillance indicators (GET version)."""
    try:
        result = search_surveillance(
            indicator_name=indicator,
            country=country,
            year_from=year_from,
            year_to=year_to,
            limit=limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching surveillance: {str(e)}")


@router.get("/correlation/{pathogen}")
async def get_clinical_correlation(pathogen: str):
    """
    Get clinical-pathogen correlation analysis.

    Returns:
    - Associated diseases
    - ICD-10 code mappings
    - AMR concerns
    """
    try:
        from app.core.pathogen_detection import get_engine
        engine = get_engine()
        result = engine.get_clinical_pathogen_correlation(pathogen)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching correlation: {str(e)}")
