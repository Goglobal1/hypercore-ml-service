"""
Risk Calculator - Auto-calculate risk scores from raw biomarker/vital values.

This module provides automatic risk scoring based on the biomarker thresholds
defined in config.py. It handles:
- Direction-aware scoring (rising vs falling thresholds)
- Weight-based composite scoring
- Case-insensitive biomarker matching
- Domain-specific threshold lookup
"""

from typing import Dict, Any, Optional, List, Tuple
import logging

from .config import BIOMARKER_THRESHOLDS, BiomarkerThreshold

logger = logging.getLogger(__name__)


# Biomarker name normalization mapping
BIOMARKER_ALIASES: Dict[str, str] = {
    # Sepsis markers
    "lactate": "lactate",
    "lactic_acid": "lactate",
    "wbc": "wbc",
    "white_blood_cell": "wbc",
    "white_blood_cells": "wbc",
    "leukocytes": "wbc",
    "crp": "crp",
    "c_reactive_protein": "crp",
    "c-reactive_protein": "crp",
    "procalcitonin": "procalcitonin",
    "pct": "procalcitonin",
    "temperature": "temperature",
    "temp": "temperature",
    "heart_rate": "heart_rate",
    "hr": "heart_rate",
    "pulse": "heart_rate",
    "respiratory_rate": "respiratory_rate",
    "rr": "respiratory_rate",
    "resp_rate": "respiratory_rate",
    "map": "map",
    "mean_arterial_pressure": "map",

    # Cardiac markers
    "troponin": "troponin",
    "troponin_i": "troponin_i",
    "tropi": "troponin_i",
    "troponin_t": "troponin_t",
    "tropt": "troponin_t",
    "bnp": "bnp",
    "nt_probnp": "nt_probnp",
    "ntprobnp": "nt_probnp",
    "ck_mb": "ck_mb",
    "ckmb": "ck_mb",
    "systolic_bp": "systolic_bp",
    "bp_systolic": "systolic_bp",
    "blood_pressure_systolic": "systolic_bp",
    "sbp": "systolic_bp",
    "diastolic_bp": "diastolic_bp",
    "bp_diastolic": "diastolic_bp",
    "blood_pressure_diastolic": "diastolic_bp",
    "dbp": "diastolic_bp",

    # Kidney markers
    "creatinine": "creatinine",
    "creat": "creatinine",
    "bun": "bun",
    "blood_urea_nitrogen": "bun",
    "potassium": "potassium",
    "k": "potassium",
    "gfr": "gfr",
    "egfr": "gfr",
    "urine_output": "urine_output",
    "uo": "urine_output",
    "sodium": "sodium",
    "na": "sodium",

    # Respiratory markers
    "spo2": "spo2",
    "oxygen_saturation": "spo2",
    "o2_sat": "spo2",
    "pao2": "pao2",
    "fio2": "fio2",
    "pao2_fio2": "pao2_fio2",
    "pf_ratio": "pao2_fio2",
    "paco2": "paco2",

    # Hepatic markers
    "alt": "alt",
    "sgpt": "alt",
    "ast": "ast",
    "sgot": "ast",
    "bilirubin": "bilirubin",
    "bili": "bilirubin",
    "total_bilirubin": "bilirubin",
    "inr": "inr",
    "albumin": "albumin",
    "alb": "albumin",
    "ammonia": "ammonia",

    # Neurological markers
    "gcs": "gcs",
    "glasgow_coma_scale": "gcs",
    "icp": "icp",
    "intracranial_pressure": "icp",
    "cpp": "cpp",
    "cerebral_perfusion_pressure": "cpp",

    # Hematologic markers
    "hemoglobin": "hemoglobin",
    "hgb": "hemoglobin",
    "hb": "hemoglobin",
    "platelets": "platelets",
    "plt": "platelets",
    "platelet_count": "platelets",
    "fibrinogen": "fibrinogen",
    "fib": "fibrinogen",

    # Metabolic markers
    "glucose": "glucose",
    "blood_glucose": "glucose",
    "bg": "glucose",
    "ph": "ph",
    "anion_gap": "anion_gap",
    "ag": "anion_gap",
}


def normalize_biomarker_name(name: str) -> str:
    """Normalize biomarker name to canonical form."""
    # Lowercase and replace common separators
    normalized = name.lower().strip().replace("-", "_").replace(" ", "_")

    # Check aliases
    return BIOMARKER_ALIASES.get(normalized, normalized)


def get_domain_thresholds(risk_domain: str) -> Dict[str, BiomarkerThreshold]:
    """Get biomarker thresholds for a domain with fallback logic."""
    domain_lower = risk_domain.lower().replace("-", "_").replace(" ", "_")

    # Direct match
    if domain_lower in BIOMARKER_THRESHOLDS:
        return BIOMARKER_THRESHOLDS[domain_lower]

    # Try partial match
    for key in BIOMARKER_THRESHOLDS:
        if key in domain_lower or domain_lower in key:
            return BIOMARKER_THRESHOLDS[key]

    # Default to sepsis (most comprehensive)
    return BIOMARKER_THRESHOLDS.get("sepsis", {})


def calculate_biomarker_score(
    value: float,
    threshold: BiomarkerThreshold,
) -> Tuple[float, str]:
    """
    Calculate score for a single biomarker value.

    Returns:
        Tuple of (score, level) where:
        - score: 0.0-1.0 normalized severity
        - level: "normal", "warning", or "critical"
    """
    direction = threshold.direction.lower()

    if direction == "rising":
        # Bad when high (lactate, WBC, HR, etc.)
        if value >= threshold.critical:
            return 1.0, "critical"
        elif value >= threshold.warning:
            # Interpolate between warning (0.5) and critical (1.0)
            range_size = threshold.critical - threshold.warning
            if range_size > 0:
                progress = (value - threshold.warning) / range_size
                return 0.5 + 0.5 * progress, "warning"
            return 0.5, "warning"
        else:
            # Below warning - could still contribute slightly if close
            if threshold.warning > 0:
                ratio = value / threshold.warning
                if ratio > 0.8:
                    return 0.2 * ratio, "normal"
            return 0.0, "normal"

    elif direction == "falling":
        # Bad when low (BP, SpO2, GFR, etc.)
        if value <= threshold.critical:
            return 1.0, "critical"
        elif value <= threshold.warning:
            # Interpolate between warning (0.5) and critical (1.0)
            range_size = threshold.warning - threshold.critical
            if range_size > 0:
                progress = (threshold.warning - value) / range_size
                return 0.5 + 0.5 * progress, "warning"
            return 0.5, "warning"
        else:
            # Above warning (good direction) - could still contribute if close
            if threshold.warning > 0:
                ratio = threshold.warning / value if value > 0 else 0
                if ratio > 0.8:
                    return 0.2 * ratio, "normal"
            return 0.0, "normal"

    return 0.0, "normal"


def calculate_risk_score(
    risk_domain: str,
    lab_data: Optional[Dict[str, Any]] = None,
    vital_signs: Optional[Dict[str, Any]] = None,
    clinical_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate composite risk score (0.0 - 1.0) from raw biomarker values.

    Logic:
    1. For each biomarker in the data:
       - If value exceeds CRITICAL threshold: score contribution ~0.8-1.0
       - If value exceeds WARNING threshold: score contribution ~0.4-0.7
       - If value is normal: score contribution ~0.0-0.2
    2. Weight by biomarker importance (from config)
    3. Normalize by total possible weight
    4. Apply domain-specific adjustments
    5. Cap at 1.0

    Returns:
        Dict containing:
        - risk_score: float (0.0-1.0)
        - contributing_biomarkers: List of biomarkers that contributed
        - critical_biomarkers: List of biomarkers at critical level
        - warning_biomarkers: List of biomarkers at warning level
        - details: Dict of per-biomarker scoring details
    """
    # Get domain thresholds
    thresholds = get_domain_thresholds(risk_domain)

    # Combine all data
    all_data: Dict[str, Any] = {}
    if lab_data:
        all_data.update(lab_data)
    if vital_signs:
        all_data.update(vital_signs)

    if not all_data:
        return {
            "risk_score": 0.0,
            "contributing_biomarkers": [],
            "critical_biomarkers": [],
            "warning_biomarkers": [],
            "details": {},
            "calculation_method": "no_data",
        }

    # Calculate scores
    total_weighted_score = 0.0
    total_weight = 0.0
    contributing_biomarkers: List[str] = []
    critical_biomarkers: List[str] = []
    warning_biomarkers: List[str] = []
    details: Dict[str, Any] = {}

    matched_biomarkers = 0

    for raw_name, value in all_data.items():
        # Skip non-numeric values
        if not isinstance(value, (int, float)):
            continue

        # Normalize name
        normalized_name = normalize_biomarker_name(raw_name)

        # Find threshold
        threshold = thresholds.get(normalized_name)
        if threshold is None:
            # Try to find in any domain for common biomarkers
            for domain_thresholds in BIOMARKER_THRESHOLDS.values():
                if normalized_name in domain_thresholds:
                    threshold = domain_thresholds[normalized_name]
                    break

        if threshold is None:
            # Unknown biomarker - skip but log
            logger.debug(f"No threshold found for biomarker: {raw_name} (normalized: {normalized_name})")
            continue

        matched_biomarkers += 1

        # Calculate score
        score, level = calculate_biomarker_score(value, threshold)
        weight = threshold.weight

        weighted_contribution = score * weight
        total_weighted_score += weighted_contribution
        total_weight += weight

        # Track contributions
        if score > 0.1:
            contributing_biomarkers.append(raw_name)

        if level == "critical":
            critical_biomarkers.append(raw_name)
        elif level == "warning":
            warning_biomarkers.append(raw_name)

        details[raw_name] = {
            "normalized_name": normalized_name,
            "value": value,
            "threshold_warning": threshold.warning,
            "threshold_critical": threshold.critical,
            "direction": threshold.direction,
            "weight": weight,
            "score": round(score, 3),
            "level": level,
            "weighted_contribution": round(weighted_contribution, 3),
        }

    # Calculate final score
    if total_weight == 0:
        risk_score = 0.0
        calculation_method = "no_matching_biomarkers"
    else:
        # Base score from weighted average
        risk_score = total_weighted_score / total_weight

        # Boost for multiple critical/warning markers
        critical_count = len(critical_biomarkers)
        warning_count = len(warning_biomarkers)

        if critical_count >= 3:
            risk_score = min(risk_score * 1.15, 1.0)  # 15% boost
        elif critical_count >= 2:
            risk_score = min(risk_score * 1.10, 1.0)  # 10% boost

        if warning_count >= 4:
            risk_score = min(risk_score * 1.05, 1.0)  # 5% boost for many warnings

        calculation_method = "weighted_threshold_analysis"

    # Cap at 1.0
    risk_score = min(risk_score, 1.0)

    return {
        "risk_score": round(risk_score, 4),
        "contributing_biomarkers": contributing_biomarkers,
        "critical_biomarkers": critical_biomarkers,
        "warning_biomarkers": warning_biomarkers,
        "matched_biomarkers": matched_biomarkers,
        "total_inputs": len(all_data),
        "total_weight": round(total_weight, 2),
        "details": details,
        "calculation_method": calculation_method,
    }


def quick_risk_score(
    risk_domain: str,
    lab_data: Optional[Dict[str, Any]] = None,
    vital_signs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Quick helper to get just the risk score float value.

    Use this for simple integrations where you just need the score.
    """
    result = calculate_risk_score(risk_domain, lab_data, vital_signs)
    return result["risk_score"]


# Convenience exports
__all__ = [
    "calculate_risk_score",
    "quick_risk_score",
    "calculate_biomarker_score",
    "normalize_biomarker_name",
    "get_domain_thresholds",
    "BIOMARKER_ALIASES",
]
