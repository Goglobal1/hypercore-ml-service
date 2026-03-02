# domain_classifier.py
"""
Domain Discovery Engine for Clinical State Engine (CSE)

Classifies biomarkers and features into clinical domains for domain-specific
alerting configurations. Part of Alert Trigger Contract (ATC) v1.

Usage:
    from app.core.domain_classifier import classify_domains, get_primary_domain

    domains = classify_domains(
        top_features=["lactate", "CRP", "WBC"],
        feature_values={"lactate": 4.2, "CRP": 150, "WBC": 18000},
        feature_changes={"lactate": 0.8, "CRP": 50}
    )
    # Returns: [{"domain": "sepsis", "confidence": 0.85, "primary_drivers": ["lactate", "CRP", "WBC"]}]
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class ClinicalDomain(str, Enum):
    """Supported clinical domains for risk classification."""
    SEPSIS = "sepsis"
    CARDIAC = "deterioration_cardiac"
    KIDNEY = "kidney_injury"
    RESPIRATORY = "respiratory_failure"
    HEPATIC = "hepatic_dysfunction"
    NEUROLOGICAL = "neurological"
    METABOLIC = "metabolic"
    HEMATOLOGIC = "hematologic"
    ONCOLOGY = "oncology_inception"
    MULTI_SYSTEM = "multi_system"
    UNKNOWN = "unknown"


@dataclass
class DomainSignature:
    """Defines biomarker signatures for a clinical domain."""
    domain: ClinicalDomain
    primary_markers: Set[str]      # High-weight markers
    secondary_markers: Set[str]    # Supporting markers
    min_primary_match: int         # Minimum primary markers to consider
    base_confidence: float         # Base confidence when matched


# Domain signature definitions
DOMAIN_SIGNATURES: Dict[ClinicalDomain, DomainSignature] = {
    ClinicalDomain.SEPSIS: DomainSignature(
        domain=ClinicalDomain.SEPSIS,
        primary_markers={
            "lactate", "procalcitonin", "pct", "crp", "c_reactive_protein",
            "wbc", "white_blood_cell", "leukocytes", "il6", "interleukin_6",
            "tnf_alpha", "presepsin", "sofa_score", "qsofa"
        },
        secondary_markers={
            "temperature", "temp", "fever", "heart_rate", "hr", "tachycardia",
            "respiratory_rate", "rr", "blood_pressure", "bp", "hypotension",
            "bandemia", "bands", "immature_granulocytes", "ig"
        },
        min_primary_match=1,
        base_confidence=0.60
    ),

    ClinicalDomain.CARDIAC: DomainSignature(
        domain=ClinicalDomain.CARDIAC,
        primary_markers={
            "troponin", "troponin_i", "troponin_t", "hs_troponin", "tni", "tnt",
            "bnp", "nt_probnp", "nt_pro_bnp", "pro_bnp", "ck_mb", "ckmb",
            "myoglobin", "ldh", "lactate_dehydrogenase", "ecg_st_elevation",
            "lvef", "ejection_fraction", "ef"
        },
        secondary_markers={
            "chest_pain", "dyspnea", "shortness_of_breath", "sob",
            "heart_rate", "hr", "arrhythmia", "af", "afib", "blood_pressure",
            "bp", "hypertension", "hypotension", "edema", "jugular_venous_pressure"
        },
        min_primary_match=1,
        base_confidence=0.65
    ),

    ClinicalDomain.KIDNEY: DomainSignature(
        domain=ClinicalDomain.KIDNEY,
        primary_markers={
            "creatinine", "cr", "scr", "serum_creatinine", "bun", "blood_urea_nitrogen",
            "urea", "egfr", "gfr", "glomerular_filtration_rate", "cystatin_c",
            "ngal", "kim1", "kim_1", "urine_output", "oliguria", "anuria"
        },
        secondary_markers={
            "potassium", "k", "hyperkalemia", "phosphate", "phosphorus",
            "bicarbonate", "hco3", "acidosis", "proteinuria", "albuminuria",
            "urine_protein", "fluid_overload", "edema"
        },
        min_primary_match=1,
        base_confidence=0.65
    ),

    ClinicalDomain.RESPIRATORY: DomainSignature(
        domain=ClinicalDomain.RESPIRATORY,
        primary_markers={
            "pao2", "po2", "oxygen_saturation", "spo2", "sao2", "fio2",
            "pao2_fio2", "pf_ratio", "p_f_ratio", "paco2", "pco2",
            "respiratory_rate", "rr", "peep", "tidal_volume", "minute_ventilation",
            "ards", "acute_respiratory_distress"
        },
        secondary_markers={
            "dyspnea", "shortness_of_breath", "sob", "wheezing", "crackles",
            "cyanosis", "accessory_muscle_use", "chest_xray_infiltrates",
            "intubation", "mechanical_ventilation", "oxygen_therapy"
        },
        min_primary_match=1,
        base_confidence=0.60
    ),

    ClinicalDomain.HEPATIC: DomainSignature(
        domain=ClinicalDomain.HEPATIC,
        primary_markers={
            "alt", "alanine_aminotransferase", "sgpt", "ast", "aspartate_aminotransferase",
            "sgot", "bilirubin", "total_bilirubin", "direct_bilirubin", "tbili",
            "alp", "alkaline_phosphatase", "ggt", "gamma_gt", "inr", "pt",
            "prothrombin_time", "albumin", "ammonia", "nh3"
        },
        secondary_markers={
            "jaundice", "icterus", "ascites", "encephalopathy", "hepatomegaly",
            "coagulopathy", "varices", "portal_hypertension", "spider_angioma"
        },
        min_primary_match=1,
        base_confidence=0.60
    ),

    ClinicalDomain.NEUROLOGICAL: DomainSignature(
        domain=ClinicalDomain.NEUROLOGICAL,
        primary_markers={
            "gcs", "glasgow_coma_scale", "consciousness", "loc", "level_of_consciousness",
            "pupils", "pupil_reactivity", "nihss", "nih_stroke_scale",
            "icp", "intracranial_pressure", "cerebral_perfusion", "cpp",
            "eeg_pattern", "seizure_activity"
        },
        secondary_markers={
            "confusion", "delirium", "altered_mental_status", "ams",
            "headache", "focal_deficit", "weakness", "numbness", "vision_change",
            "speech_change", "aphasia", "ataxia"
        },
        min_primary_match=1,
        base_confidence=0.55
    ),

    ClinicalDomain.METABOLIC: DomainSignature(
        domain=ClinicalDomain.METABOLIC,
        primary_markers={
            "glucose", "blood_glucose", "bg", "hba1c", "hemoglobin_a1c",
            "sodium", "na", "hyponatremia", "hypernatremia", "potassium", "k",
            "hypokalemia", "hyperkalemia", "calcium", "ca", "magnesium", "mg",
            "ph", "blood_ph", "lactate", "anion_gap", "osmolality"
        },
        secondary_markers={
            "polyuria", "polydipsia", "ketones", "ketonuria", "dka",
            "hhs", "metabolic_acidosis", "metabolic_alkalosis", "electrolyte_imbalance"
        },
        min_primary_match=2,
        base_confidence=0.50
    ),

    ClinicalDomain.HEMATOLOGIC: DomainSignature(
        domain=ClinicalDomain.HEMATOLOGIC,
        primary_markers={
            "hemoglobin", "hgb", "hb", "hematocrit", "hct", "rbc", "red_blood_cells",
            "platelets", "plt", "thrombocytopenia", "wbc", "white_blood_cells",
            "neutrophils", "anc", "lymphocytes", "inr", "pt", "ptt", "aptt",
            "fibrinogen", "d_dimer", "ddimer"
        },
        secondary_markers={
            "anemia", "bleeding", "petechiae", "ecchymosis", "pallor",
            "fatigue", "weakness", "transfusion", "coagulopathy", "dic"
        },
        min_primary_match=2,
        base_confidence=0.50
    ),

    ClinicalDomain.ONCOLOGY: DomainSignature(
        domain=ClinicalDomain.ONCOLOGY,
        primary_markers={
            "tumor_marker", "cea", "ca125", "ca_125", "ca199", "ca_19_9",
            "psa", "afp", "alpha_fetoprotein", "ldh", "b2_microglobulin",
            "ctdna", "circulating_tumor_dna", "ctc", "circulating_tumor_cells",
            "mutation_burden", "tmb", "msi", "microsatellite_instability"
        },
        secondary_markers={
            "weight_loss", "cachexia", "night_sweats", "lymphadenopathy",
            "mass", "tumor", "malignancy", "metastasis", "recurrence"
        },
        min_primary_match=1,
        base_confidence=0.50
    )
}


def _normalize_feature_name(feature: str) -> str:
    """Normalize feature name for matching."""
    return feature.lower().strip().replace("-", "_").replace(" ", "_")


def _calculate_domain_confidence(
    domain_sig: DomainSignature,
    matched_primary: List[str],
    matched_secondary: List[str],
    feature_values: Dict[str, float],
    feature_changes: Dict[str, float]
) -> float:
    """
    Calculate confidence score for a domain match.

    Factors:
    - Number of primary markers matched
    - Number of secondary markers matched
    - Abnormal values (if thresholds known)
    - Rate of change in key markers
    """
    confidence = domain_sig.base_confidence

    # Boost for multiple primary markers
    primary_boost = min(0.25, len(matched_primary) * 0.08)
    confidence += primary_boost

    # Smaller boost for secondary markers
    secondary_boost = min(0.10, len(matched_secondary) * 0.03)
    confidence += secondary_boost

    # Boost for rapidly changing markers
    normalized_changes = {_normalize_feature_name(k): v for k, v in feature_changes.items()}
    for marker in matched_primary:
        if marker in normalized_changes:
            change = abs(normalized_changes[marker])
            if change > 0.5:  # Significant change
                confidence += 0.05

    return min(0.99, confidence)


def classify_domains(
    top_features: List[str],
    feature_values: Optional[Dict[str, float]] = None,
    feature_changes: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Classify features into clinical domains with confidence scores.

    Args:
        top_features: List of top contributing features/biomarkers
        feature_values: Optional dict of feature name -> current value
        feature_changes: Optional dict of feature name -> rate of change

    Returns:
        List of detected domains sorted by confidence (highest first):
        [
            {
                "domain": "sepsis",
                "confidence": 0.85,
                "primary_drivers": ["lactate", "CRP", "WBC"],
                "secondary_drivers": ["temperature", "heart_rate"]
            },
            ...
        ]
    """
    feature_values = feature_values or {}
    feature_changes = feature_changes or {}

    # Normalize all feature names
    normalized_features = [_normalize_feature_name(f) for f in top_features]
    normalized_values = {_normalize_feature_name(k): v for k, v in feature_values.items()}

    results = []

    for domain, signature in DOMAIN_SIGNATURES.items():
        # Find matching markers
        matched_primary = [
            f for f in normalized_features
            if f in signature.primary_markers
        ]
        matched_secondary = [
            f for f in normalized_features
            if f in signature.secondary_markers
        ]

        # Check minimum match threshold
        if len(matched_primary) < signature.min_primary_match:
            continue

        # Calculate confidence
        confidence = _calculate_domain_confidence(
            signature,
            matched_primary,
            matched_secondary,
            normalized_values,
            feature_changes
        )

        results.append({
            "domain": domain.value,
            "confidence": round(confidence, 3),
            "primary_drivers": matched_primary,
            "secondary_drivers": matched_secondary
        })

    # Check for multi-system involvement
    if len(results) >= 2:
        # Multiple domains detected - add multi-system marker
        max_confidence = max(r["confidence"] for r in results)
        all_primary_drivers = []
        for r in results:
            all_primary_drivers.extend(r["primary_drivers"])

        results.append({
            "domain": ClinicalDomain.MULTI_SYSTEM.value,
            "confidence": round(min(0.95, max_confidence + 0.05), 3),
            "primary_drivers": list(set(all_primary_drivers))[:5],
            "secondary_drivers": [],
            "component_domains": [r["domain"] for r in results if r["domain"] != "multi_system"]
        })

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # If no domains matched, return unknown
    if not results:
        results.append({
            "domain": ClinicalDomain.UNKNOWN.value,
            "confidence": 0.0,
            "primary_drivers": top_features[:3] if top_features else [],
            "secondary_drivers": []
        })

    return results


def get_primary_domain(
    top_features: List[str],
    feature_values: Optional[Dict[str, float]] = None,
    feature_changes: Optional[Dict[str, float]] = None
) -> Tuple[str, float, List[str]]:
    """
    Get the single most likely clinical domain.

    Args:
        top_features: List of top contributing features/biomarkers
        feature_values: Optional dict of feature name -> current value
        feature_changes: Optional dict of feature name -> rate of change

    Returns:
        Tuple of (domain_name, confidence, primary_drivers)
    """
    domains = classify_domains(top_features, feature_values, feature_changes)

    if not domains:
        return (ClinicalDomain.UNKNOWN.value, 0.0, [])

    primary = domains[0]
    return (
        primary["domain"],
        primary["confidence"],
        primary.get("primary_drivers", [])
    )


def get_domain_from_endpoint(
    endpoint_name: str,
    fallback_features: Optional[List[str]] = None
) -> str:
    """
    Map endpoint name to default domain when no features available.

    This provides sensible defaults for endpoints that don't return
    individual biomarkers but still need domain classification.
    """
    endpoint_domain_map = {
        "analyze": "cohort_analysis",
        "early_risk_discovery": "early_risk",
        "multi_omic_fusion": "multi_omic",
        "responder_prediction": "trial_response",
        "digital_twin_simulation": "digital_twin",
        "population_risk": "population_risk",
        "forecast_timeline": "forecast_risk",
        "outbreak_detection": "outbreak",
        "medication_interaction": "medication_interaction",
        "trial_rescue": "trial_rescue",
        "lead_time_analysis": "lead_time",
        "change_point_detect": "change_point",
        "surveillance": "surveillance",
        "predict": "synthetic_cohort",
        "confounder_detection": "confounder_analysis"
    }

    # Try direct match
    if endpoint_name in endpoint_domain_map:
        return endpoint_domain_map[endpoint_name]

    # Try partial match
    for key, domain in endpoint_domain_map.items():
        if key in endpoint_name.lower():
            return domain

    # If features provided, try to classify
    if fallback_features:
        domain, confidence, _ = get_primary_domain(fallback_features)
        if confidence > 0.3:
            return domain

    return "unknown"


# Convenience exports
__all__ = [
    "ClinicalDomain",
    "classify_domains",
    "get_primary_domain",
    "get_domain_from_endpoint",
    "DOMAIN_SIGNATURES"
]
