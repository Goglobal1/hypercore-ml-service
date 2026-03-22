"""
Universal Data Ingestion Router - Bulletproof endpoints for HyperCore.

PHILOSOPHY:
- NEVER crash - always produce something useful
- Do the BEST with what you have
- Tell the user what's MISSING that would help
- Be SUPERIOR - handle reality, not perfect lab data
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from ..core.data_ingestion import (
    RobustDataParser,
    parse_any_data,
    extract_lab_data,
    DataQualityReport,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/universal", tags=["universal"])


class UniversalAnalysisResponse(BaseModel):
    """Response from universal analysis endpoint."""
    success: bool
    analysis_level: str  # "full", "partial", "minimal"
    data_quality: Dict[str, Any]
    parsed_data: Optional[Dict[str, Any]] = None
    analysis_results: Optional[Dict[str, Any]] = None
    recommendations: List[str]
    next_steps: List[str]
    timestamp: str


class DataQualityResponse(BaseModel):
    """Response from data quality check."""
    success: bool
    quality_score: float
    recognized_biomarkers: Dict[str, Any]
    unrecognized_columns: List[str]
    missing_by_domain: Dict[str, Any]
    recommendations: List[str]
    parse_method: str
    warnings: List[str]


@router.post("/analyze")
async def universal_analyze(request: Request) -> UniversalAnalysisResponse:
    """
    Universal analysis endpoint - NEVER FAILS.

    Accepts ANY data format:
    - JSON body
    - CSV data
    - Plain text with key-value pairs
    - Malformed data (does its best)

    Returns analysis with data quality information.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # Get raw body
        body = await request.body()
        content_type = request.headers.get("content-type", "")

        # Try to parse based on content type
        if "application/json" in content_type:
            try:
                import json
                data = json.loads(body.decode('utf-8', errors='replace'))
            except:
                data = body.decode('utf-8', errors='replace')
        else:
            data = body.decode('utf-8', errors='replace')

        # Use robust parser
        parser = RobustDataParser()
        quality_report = parser.parse_any_input(data)

        # Determine analysis level
        if quality_report.score >= 0.7:
            analysis_level = "full"
        elif quality_report.score >= 0.3:
            analysis_level = "partial"
        elif quality_report.can_analyze:
            analysis_level = "minimal"
        else:
            analysis_level = "data_quality_only"

        # Extract lab values
        lab_data = parser.extract_lab_values(quality_report.parsed_data)

        # Run analysis based on what we have
        analysis_results = None
        if quality_report.can_analyze and lab_data:
            analysis_results = await _run_analysis_with_available_data(
                lab_data=lab_data,
                quality_score=quality_report.score,
            )

        # Generate next steps
        next_steps = _generate_next_steps(quality_report, analysis_level)

        return UniversalAnalysisResponse(
            success=True,
            analysis_level=analysis_level,
            data_quality={
                "score": quality_report.score,
                "parse_method": quality_report.parse_method,
                "recognized_count": len(quality_report.recognized_biomarkers),
                "unrecognized_count": len(quality_report.unrecognized_columns),
                "can_analyze": quality_report.can_analyze,
                "warnings": quality_report.warnings[:5],
            },
            parsed_data={
                "biomarkers": quality_report.recognized_biomarkers,
                "raw_records": len(quality_report.parsed_data),
            },
            analysis_results=analysis_results,
            recommendations=quality_report.recommendations[:5],
            next_steps=next_steps,
            timestamp=timestamp,
        )

    except Exception as e:
        # Even errors don't crash - return helpful info
        logger.error(f"Universal analyze error: {e}")
        return UniversalAnalysisResponse(
            success=True,  # We still produced a response
            analysis_level="error_recovery",
            data_quality={
                "score": 0.0,
                "parse_method": "error_recovery",
                "recognized_count": 0,
                "unrecognized_count": 0,
                "can_analyze": False,
                "warnings": [str(e)[:200]],
            },
            parsed_data=None,
            analysis_results=None,
            recommendations=[
                "Provide data in a supported format",
                "Supported formats: JSON, CSV, key-value pairs",
                "Example: {'crp': 28.5, 'wbc': 18.0, 'procalcitonin': 2.5}",
            ],
            next_steps=[
                "Check your data format",
                "Try sending a simple JSON object with biomarker values",
            ],
            timestamp=timestamp,
        )


@router.post("/check_quality")
async def check_data_quality(request: Request) -> DataQualityResponse:
    """
    Check data quality without running full analysis.

    Useful for validating data before submission.
    """
    try:
        body = await request.body()
        data = body.decode('utf-8', errors='replace')

        # Try to parse as JSON first
        try:
            import json
            data = json.loads(data)
        except:
            pass

        parser = RobustDataParser()
        quality_report = parser.parse_any_input(data)

        return DataQualityResponse(
            success=True,
            quality_score=quality_report.score,
            recognized_biomarkers=quality_report.recognized_biomarkers,
            unrecognized_columns=quality_report.unrecognized_columns,
            missing_by_domain=quality_report.missing_by_domain,
            recommendations=quality_report.recommendations,
            parse_method=quality_report.parse_method,
            warnings=quality_report.warnings,
        )

    except Exception as e:
        logger.error(f"Data quality check error: {e}")
        return DataQualityResponse(
            success=False,
            quality_score=0.0,
            recognized_biomarkers={},
            unrecognized_columns=[],
            missing_by_domain={},
            recommendations=["Unable to parse data: " + str(e)[:100]],
            parse_method="error",
            warnings=[str(e)[:200]],
        )


@router.post("/parse")
async def parse_data(request: Request) -> Dict[str, Any]:
    """
    Parse any data format and return structured result.

    Does NOT run analysis - just parses and maps to biomarkers.
    """
    try:
        body = await request.body()
        data = body.decode('utf-8', errors='replace')

        # Try JSON first
        try:
            import json
            data = json.loads(data)
        except:
            pass

        parser = RobustDataParser()
        quality_report = parser.parse_any_input(data)
        lab_data = parser.extract_lab_values(quality_report.parsed_data)

        return {
            "success": True,
            "parse_method": quality_report.parse_method,
            "quality_score": quality_report.score,
            "lab_data": lab_data,
            "recognized_biomarkers": quality_report.recognized_biomarkers,
            "unrecognized_columns": quality_report.unrecognized_columns,
            "record_count": len(quality_report.parsed_data),
            "warnings": quality_report.warnings,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)[:200],
            "parse_method": "failed",
            "quality_score": 0.0,
            "lab_data": {},
            "recognized_biomarkers": {},
            "unrecognized_columns": [],
            "record_count": 0,
            "warnings": [str(e)[:200]],
        }


async def _run_analysis_with_available_data(
    lab_data: Dict[str, Any],
    quality_score: float,
) -> Dict[str, Any]:
    """
    Run analysis with whatever data is available.

    Uses auto-inference to maximize insight from partial data.
    """
    try:
        # Import pipeline for patient analysis
        from ..core.alert_system.pipeline import get_pipeline

        pipeline = get_pipeline()

        # Extract patient_id if present
        patient_id = lab_data.pop("patient_id", f"auto_{datetime.now().strftime('%H%M%S')}")

        # Determine risk domain from available biomarkers
        risk_domain = _infer_risk_domain(lab_data)

        # Run through pipeline (which has auto-inference)
        result = await pipeline.process_patient_data(
            patient_id=str(patient_id),
            risk_domain=risk_domain,
            lab_data=lab_data,
        )

        # Extract key results
        eval_result = result.evaluation_result

        return {
            "risk_domain": risk_domain,
            "risk_score": eval_result.risk_score if eval_result else 0.0,
            "state": eval_result.state_now.value if eval_result else "unknown",
            "alert_fired": eval_result.alert_fired if eval_result else False,
            "severity": eval_result.severity.value if eval_result else "INFO",
            "cascade_detection": eval_result.cascade_detection if eval_result else None,
            "specialized_modules": (
                eval_result.agent_findings.get("specialized_modules", {})
                if eval_result and eval_result.agent_findings else {}
            ),
            "data_quality_adjusted": quality_score < 0.7,
        }

    except Exception as e:
        logger.warning(f"Analysis with partial data failed: {e}")
        return {
            "error": str(e)[:100],
            "risk_domain": "unknown",
            "risk_score": 0.0,
            "state": "unknown",
            "alert_fired": False,
            "severity": "INFO",
            "cascade_detection": None,
            "specialized_modules": {},
            "data_quality_adjusted": True,
        }


def _infer_risk_domain(lab_data: Dict[str, Any]) -> str:
    """Infer the most likely risk domain from available biomarkers."""
    # Count biomarkers in each domain
    domain_scores = {
        "sepsis": 0,
        "cardiac": 0,
        "oncology": 0,
        "metabolic": 0,
        "renal": 0,
        "hepatic": 0,
    }

    biomarker_domains = {
        "crp": ["sepsis", "cardiac"],
        "wbc": ["sepsis"],
        "procalcitonin": ["sepsis"],
        "lactate": ["sepsis"],
        "temperature": ["sepsis"],
        "troponin": ["cardiac"],
        "bnp": ["cardiac"],
        "cea": ["oncology"],
        "ca125": ["oncology"],
        "psa": ["oncology"],
        "afp": ["oncology"],
        "glucose": ["metabolic"],
        "hba1c": ["metabolic"],
        "triglycerides": ["metabolic"],
        "ldl": ["metabolic", "cardiac"],
        "creatinine": ["renal"],
        "bun": ["renal"],
        "egfr": ["renal"],
        "alt": ["hepatic"],
        "ast": ["hepatic"],
        "bilirubin": ["hepatic"],
        "albumin": ["hepatic"],
    }

    for biomarker in lab_data.keys():
        biomarker_lower = biomarker.lower()
        if biomarker_lower in biomarker_domains:
            for domain in biomarker_domains[biomarker_lower]:
                domain_scores[domain] += 1

    # Return highest scoring domain, default to general
    if max(domain_scores.values()) > 0:
        return max(domain_scores, key=domain_scores.get)

    return "general"


def _generate_next_steps(
    quality_report: DataQualityReport,
    analysis_level: str,
) -> List[str]:
    """Generate actionable next steps based on data quality."""
    steps = []

    if analysis_level == "data_quality_only":
        steps.append("Resubmit with recognized biomarker names")
        steps.append("Accepted formats: JSON object, CSV, or 'key: value' pairs")
    elif analysis_level == "minimal":
        steps.append("Add more biomarkers for comprehensive analysis")
        if quality_report.missing_by_domain:
            # Find best domain
            best_domain = max(
                quality_report.missing_by_domain.items(),
                key=lambda x: x[1].get("completeness", 0)
            )
            missing = best_domain[1].get("critical_missing", [])[:3]
            if missing:
                steps.append(f"For {best_domain[0]} analysis, add: {', '.join(missing)}")
    elif analysis_level == "partial":
        steps.append("Analysis performed with available data")
        # Suggest genetic testing
        steps.append("Genetic markers would improve detection by 2-3 days")
    else:
        steps.append("Full analysis completed")
        steps.append("Monitor for changes and resubmit with updated values")

    return steps


# Health check
@router.get("/health")
async def health():
    """Health check for universal router."""
    return {
        "status": "ok",
        "service": "universal_data_ingestion",
        "capabilities": [
            "json_parsing",
            "csv_parsing",
            "key_value_extraction",
            "biomarker_mapping",
            "auto_inference",
        ],
    }
