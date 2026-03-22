"""
Clinical Domain Classifier - Evidence-Based Auto-Inference.

Uses validated clinical criteria (SOFA, qSOFA, HEART, MELD, AKI, etc.)
to automatically determine risk domain from biomarker data.

REQUIREMENTS:
- Weighted scoring with primary/secondary markers
- Minimum 60% confidence threshold
- Must have at least 1 PRIMARY marker to consider domain
- Returns confidence scores for ALL domains
- Supports multi-system involvement
- Flags "indeterminate" if no domain reaches threshold
"""

import logging
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

# Import BIOMARKER_MAPPINGS for normalization
from .robust_parser import BIOMARKER_MAPPINGS

logger = logging.getLogger(__name__)

# Common unit suffixes to strip when normalizing biomarker names
_UNIT_SUFFIXES = [
    "_ng_l", "_ng_ml", "_pg_ml", "_ug_ml", "_mg_l", "_mg_dl", "_g_dl",
    "_mmol_l", "_umol_l", "_meq_l", "_u_l", "_iu_l", "_mu_l",
    "_10e9_l", "_10e12_l", "_percent", "_pct", "_ratio",
    "_mm_hr", "_seconds", "_sec", "_ms",
]


def _normalize_biomarker(name: str) -> str:
    """
    Normalize a biomarker name to standard form.

    Handles units in name, prefixes, case variations.
    """
    normalized = name.lower().strip().replace("-", "_").replace(" ", "_")

    # Direct lookup
    if normalized in BIOMARKER_MAPPINGS:
        return BIOMARKER_MAPPINGS[normalized]

    # Try stripping unit suffixes
    for suffix in _UNIT_SUFFIXES:
        if normalized.endswith(suffix):
            stripped = normalized[:-len(suffix)]
            if stripped in BIOMARKER_MAPPINGS:
                return BIOMARKER_MAPPINGS[stripped]

    # Try regex stripping
    stripped = re.sub(r"_(?:ng|pg|ug|mg|g|mmol|umol|meq|u|iu|mu|10e\d+)_?(?:l|dl|ml)?$", "", normalized)
    if stripped in BIOMARKER_MAPPINGS:
        return BIOMARKER_MAPPINGS[stripped]

    return normalized


# =============================================================================
# CLINICAL DOMAIN DEFINITIONS (Evidence-Based)
# =============================================================================

@dataclass
class MarkerCriteria:
    """Definition of a biomarker threshold criterion."""
    marker: str
    threshold: float
    operator: str  # ">" or "<"
    weight: float
    is_primary: bool
    clinical_basis: str


@dataclass
class DomainDefinition:
    """Complete definition of a clinical domain."""
    name: str
    display_name: str
    clinical_basis: str
    markers: List[MarkerCriteria]
    min_confidence: float = 0.60


# SEPSIS - Based on Sepsis-3/SOFA criteria
SEPSIS_DOMAIN = DomainDefinition(
    name="sepsis",
    display_name="Sepsis/Infection",
    clinical_basis="Sepsis-3/SOFA criteria",
    markers=[
        # PRIMARY markers (must have 1+)
        MarkerCriteria("procalcitonin", 0.5, ">", 0.35, True, "Sepsis-3: PCT >0.5 suggests bacterial infection"),
        MarkerCriteria("pct", 0.5, ">", 0.35, True, "Sepsis-3: PCT >0.5 suggests bacterial infection"),
        MarkerCriteria("lactate", 2.0, ">", 0.30, True, "SOFA: Lactate >2 indicates tissue hypoperfusion"),
        # SECONDARY markers
        MarkerCriteria("wbc", 12.0, ">", 0.15, False, "SIRS: WBC >12,000"),
        MarkerCriteria("wbc_low", 4.0, "<", 0.15, False, "SIRS: WBC <4,000"),
        MarkerCriteria("crp", 10.0, ">", 0.10, False, "Inflammatory marker"),
        MarkerCriteria("temperature", 38.3, ">", 0.10, False, "SIRS: Temp >38.3°C"),
        MarkerCriteria("temperature_low", 36.0, "<", 0.10, False, "SIRS: Temp <36°C"),
    ],
)

# CARDIAC - Based on ACS/Heart Failure criteria
CARDIAC_DOMAIN = DomainDefinition(
    name="cardiac",
    display_name="Cardiac/Cardiovascular",
    clinical_basis="ACS/HEART Score/Heart Failure criteria",
    markers=[
        # PRIMARY markers
        MarkerCriteria("troponin", 0.04, ">", 0.40, True, "ACS: Troponin >0.04 indicates myocardial injury"),
        MarkerCriteria("troponin_i", 0.04, ">", 0.40, True, "ACS: Troponin I >0.04"),
        MarkerCriteria("troponin_t", 0.1, ">", 0.40, True, "ACS: Troponin T >0.1"),
        MarkerCriteria("bnp", 100.0, ">", 0.35, True, "Heart Failure: BNP >100 pg/mL"),
        MarkerCriteria("ntprobnp", 300.0, ">", 0.35, True, "Heart Failure: NT-proBNP >300"),
        MarkerCriteria("nt_probnp", 300.0, ">", 0.35, True, "Heart Failure: NT-proBNP >300"),
        # SECONDARY markers
        MarkerCriteria("crp", 3.0, ">", 0.10, False, "Cardiovascular inflammation marker"),
        MarkerCriteria("ldh", 250.0, ">", 0.10, False, "Tissue damage marker"),
        MarkerCriteria("d_dimer", 500.0, ">", 0.05, False, "Thrombosis marker"),
        MarkerCriteria("ddimer", 500.0, ">", 0.05, False, "Thrombosis marker"),
    ],
)

# ONCOLOGY - Based on Tumor Marker criteria
ONCOLOGY_DOMAIN = DomainDefinition(
    name="oncology",
    display_name="Oncology",
    clinical_basis="Standard tumor marker thresholds",
    markers=[
        # PRIMARY markers
        MarkerCriteria("cea", 5.0, ">", 0.25, True, "CEA >5: Colorectal, lung, breast cancer"),
        MarkerCriteria("ca125", 35.0, ">", 0.25, True, "CA-125 >35: Ovarian cancer"),
        MarkerCriteria("ca_125", 35.0, ">", 0.25, True, "CA-125 >35: Ovarian cancer"),
        MarkerCriteria("ca199", 37.0, ">", 0.25, True, "CA 19-9 >37: Pancreatic cancer"),
        MarkerCriteria("ca_199", 37.0, ">", 0.25, True, "CA 19-9 >37: Pancreatic cancer"),
        MarkerCriteria("ca19_9", 37.0, ">", 0.25, True, "CA 19-9 >37: Pancreatic cancer"),
        MarkerCriteria("psa", 4.0, ">", 0.25, True, "PSA >4: Prostate cancer screening"),
        MarkerCriteria("afp", 10.0, ">", 0.25, True, "AFP >10: Hepatocellular/germ cell"),
        # SECONDARY markers
        MarkerCriteria("ldh", 300.0, ">", 0.15, False, "Elevated in many malignancies"),
        MarkerCriteria("ferritin", 500.0, ">", 0.10, False, "Elevated in malignancy"),
    ],
)

# HEPATIC - Based on Liver Function criteria
HEPATIC_DOMAIN = DomainDefinition(
    name="hepatic",
    display_name="Hepatic/Liver",
    clinical_basis="Liver function test criteria",
    markers=[
        # PRIMARY markers
        MarkerCriteria("alt", 40.0, ">", 0.30, True, "ALT >40: Hepatocellular injury"),
        MarkerCriteria("sgpt", 40.0, ">", 0.30, True, "SGPT/ALT >40: Hepatocellular injury"),
        MarkerCriteria("ast", 40.0, ">", 0.30, True, "AST >40: Hepatocellular injury"),
        MarkerCriteria("sgot", 40.0, ">", 0.30, True, "SGOT/AST >40: Hepatocellular injury"),
        # SECONDARY markers
        MarkerCriteria("bilirubin", 1.2, ">", 0.20, False, "Hyperbilirubinemia"),
        MarkerCriteria("total_bilirubin", 1.2, ">", 0.20, False, "Hyperbilirubinemia"),
        MarkerCriteria("albumin", 3.5, "<", 0.10, False, "Hypoalbuminemia - synthetic dysfunction"),
        MarkerCriteria("inr", 1.2, ">", 0.10, False, "Coagulopathy - synthetic dysfunction"),
    ],
)

# RENAL - Based on AKI/CKD criteria (KDIGO)
RENAL_DOMAIN = DomainDefinition(
    name="renal",
    display_name="Renal/Kidney",
    clinical_basis="KDIGO AKI/CKD criteria",
    markers=[
        # PRIMARY markers
        MarkerCriteria("creatinine", 1.3, ">", 0.35, True, "Creatinine >1.3: Renal dysfunction"),
        MarkerCriteria("creat", 1.3, ">", 0.35, True, "Creatinine >1.3: Renal dysfunction"),
        MarkerCriteria("egfr", 60.0, "<", 0.35, True, "eGFR <60: CKD Stage 3+"),
        MarkerCriteria("gfr", 60.0, "<", 0.35, True, "GFR <60: CKD Stage 3+"),
        # SECONDARY markers
        MarkerCriteria("bun", 20.0, ">", 0.15, False, "Elevated BUN"),
        MarkerCriteria("potassium", 5.0, ">", 0.10, False, "Hyperkalemia"),
        MarkerCriteria("k", 5.0, ">", 0.10, False, "Hyperkalemia"),
        MarkerCriteria("phosphorus", 4.5, ">", 0.05, False, "Hyperphosphatemia"),
        MarkerCriteria("phosphate", 4.5, ">", 0.05, False, "Hyperphosphatemia"),
    ],
)

# METABOLIC - Based on Diabetes/Metabolic Syndrome criteria
METABOLIC_DOMAIN = DomainDefinition(
    name="metabolic",
    display_name="Metabolic/Endocrine",
    clinical_basis="ADA Diabetes/Metabolic Syndrome criteria",
    markers=[
        # PRIMARY markers
        MarkerCriteria("glucose", 126.0, ">", 0.35, True, "Fasting glucose >126: Diabetes"),
        MarkerCriteria("blood_sugar", 126.0, ">", 0.35, True, "Fasting glucose >126: Diabetes"),
        MarkerCriteria("hba1c", 6.5, ">", 0.35, True, "HbA1c >6.5%: Diabetes"),
        MarkerCriteria("a1c", 6.5, ">", 0.35, True, "HbA1c >6.5%: Diabetes"),
        # SECONDARY markers
        MarkerCriteria("triglycerides", 150.0, ">", 0.15, False, "Hypertriglyceridemia"),
        MarkerCriteria("tg", 150.0, ">", 0.15, False, "Hypertriglyceridemia"),
        MarkerCriteria("ldl", 130.0, ">", 0.10, False, "Elevated LDL"),
        MarkerCriteria("hdl", 40.0, "<", 0.05, False, "Low HDL"),
    ],
)

# HEMATOLOGIC - Based on CBC abnormalities
HEMATOLOGIC_DOMAIN = DomainDefinition(
    name="hematologic",
    display_name="Hematologic",
    clinical_basis="CBC/Coagulation criteria",
    markers=[
        # PRIMARY markers
        MarkerCriteria("hemoglobin", 10.0, "<", 0.30, True, "Anemia: Hgb <10"),
        MarkerCriteria("hgb", 10.0, "<", 0.30, True, "Anemia: Hgb <10"),
        MarkerCriteria("hb", 10.0, "<", 0.30, True, "Anemia: Hgb <10"),
        MarkerCriteria("platelets", 100.0, "<", 0.30, True, "Thrombocytopenia: Plt <100k"),
        MarkerCriteria("plt", 100.0, "<", 0.30, True, "Thrombocytopenia: Plt <100k"),
        MarkerCriteria("platelets_high", 400.0, ">", 0.20, True, "Thrombocytosis: Plt >400k"),
        # SECONDARY markers
        MarkerCriteria("inr", 1.5, ">", 0.15, False, "Coagulopathy"),
        MarkerCriteria("d_dimer", 500.0, ">", 0.10, False, "Fibrinolysis/DIC"),
        MarkerCriteria("fibrinogen", 150.0, "<", 0.10, False, "Hypofibrinogenemia"),
    ],
)

# All domains for evaluation
ALL_DOMAINS = [
    SEPSIS_DOMAIN,
    CARDIAC_DOMAIN,
    ONCOLOGY_DOMAIN,
    HEPATIC_DOMAIN,
    RENAL_DOMAIN,
    METABOLIC_DOMAIN,
    HEMATOLOGIC_DOMAIN,
]


# =============================================================================
# CLINICAL DOMAIN CLASSIFIER
# =============================================================================

@dataclass
class DomainScore:
    """Score for a single domain."""
    domain: str
    display_name: str
    confidence: float
    has_primary_marker: bool
    primary_markers_found: List[str]
    secondary_markers_found: List[str]
    clinical_basis: str
    meets_threshold: bool


@dataclass
class DomainClassificationResult:
    """Complete result of domain classification."""
    primary_domain: Optional[str]
    primary_confidence: float
    all_domains: List[DomainScore]
    multi_system: bool
    involved_systems: List[str]
    indeterminate: bool
    classification_notes: List[str]


class ClinicalDomainClassifier:
    """
    Evidence-based clinical domain classifier.

    Uses validated clinical criteria to determine risk domain
    from biomarker data with confidence scoring.
    """

    def __init__(self, min_confidence: float = 0.60):
        self.min_confidence = min_confidence
        self.domains = ALL_DOMAINS

    def classify(self, lab_data: Dict[str, Any]) -> DomainClassificationResult:
        """
        Classify patient into clinical domain(s) based on lab data.

        Returns comprehensive classification with confidence scores.
        """
        # Normalize lab data keys using BIOMARKER_MAPPINGS
        normalized = {_normalize_biomarker(k): v for k, v in lab_data.items()}

        # Score each domain
        domain_scores = []
        for domain_def in self.domains:
            score = self._score_domain(domain_def, normalized)
            domain_scores.append(score)

        # Sort by confidence (descending)
        domain_scores.sort(key=lambda x: x.confidence, reverse=True)

        # Find domains meeting threshold with primary markers
        qualifying_domains = [
            ds for ds in domain_scores
            if ds.meets_threshold and ds.has_primary_marker
        ]

        # Determine primary domain
        if qualifying_domains:
            primary = qualifying_domains[0]
            primary_domain = primary.domain
            primary_confidence = primary.confidence
            indeterminate = False
        else:
            primary_domain = None
            primary_confidence = 0.0
            indeterminate = True

        # Check for multi-system involvement
        multi_system = len(qualifying_domains) > 1
        involved_systems = [ds.domain for ds in qualifying_domains]

        # Generate classification notes
        notes = self._generate_notes(
            domain_scores, qualifying_domains, normalized, indeterminate
        )

        return DomainClassificationResult(
            primary_domain=primary_domain,
            primary_confidence=primary_confidence,
            all_domains=domain_scores,
            multi_system=multi_system,
            involved_systems=involved_systems,
            indeterminate=indeterminate,
            classification_notes=notes,
        )

    def _score_domain(
        self,
        domain_def: DomainDefinition,
        lab_data: Dict[str, Any]
    ) -> DomainScore:
        """Score a single domain against lab data."""
        total_weight = 0.0
        earned_weight = 0.0
        has_primary = False
        primary_found = []
        secondary_found = []

        for marker in domain_def.markers:
            marker_name = marker.marker.lower()

            # Check if marker exists in lab data
            if marker_name not in lab_data:
                continue

            value = lab_data[marker_name]
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue

            # Check threshold
            threshold_met = False
            if marker.operator == ">":
                threshold_met = value > marker.threshold
            elif marker.operator == "<":
                threshold_met = value < marker.threshold

            if threshold_met:
                earned_weight += marker.weight
                if marker.is_primary:
                    has_primary = True
                    primary_found.append(f"{marker_name}={value}")
                else:
                    secondary_found.append(f"{marker_name}={value}")

            # Track total possible weight for markers we have data for
            total_weight += marker.weight

        # Calculate confidence as earned/possible weight
        # If no markers found, confidence is 0
        if total_weight > 0:
            confidence = earned_weight / total_weight
        else:
            confidence = 0.0

        # Boost confidence if has primary marker(s)
        if has_primary:
            # Having primary markers validates the domain assignment
            confidence = min(confidence + 0.15, 1.0)

        meets_threshold = (
            confidence >= self.min_confidence and
            has_primary
        )

        return DomainScore(
            domain=domain_def.name,
            display_name=domain_def.display_name,
            confidence=round(confidence, 3),
            has_primary_marker=has_primary,
            primary_markers_found=primary_found,
            secondary_markers_found=secondary_found,
            clinical_basis=domain_def.clinical_basis,
            meets_threshold=meets_threshold,
        )

    def _generate_notes(
        self,
        all_scores: List[DomainScore],
        qualifying: List[DomainScore],
        lab_data: Dict[str, Any],
        indeterminate: bool,
    ) -> List[str]:
        """Generate clinical notes about the classification."""
        notes = []

        if indeterminate:
            notes.append(
                "INDETERMINATE: No domain reached 60% confidence with primary marker. "
                "Consider additional biomarkers."
            )
            # Suggest what markers would help
            best_without_primary = [
                s for s in all_scores
                if not s.has_primary_marker and s.confidence > 0
            ]
            if best_without_primary:
                best = best_without_primary[0]
                notes.append(
                    f"Closest domain: {best.display_name} ({best.confidence:.0%}) - "
                    f"needs primary marker"
                )

        if len(qualifying) > 1:
            domains_str = ", ".join([q.display_name for q in qualifying])
            notes.append(f"MULTI-SYSTEM INVOLVEMENT: {domains_str}")

        if qualifying:
            primary = qualifying[0]
            notes.append(
                f"Primary domain: {primary.display_name} based on {primary.clinical_basis}"
            )
            if primary.primary_markers_found:
                notes.append(
                    f"Key findings: {', '.join(primary.primary_markers_found)}"
                )

        return notes


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_classifier: Optional[ClinicalDomainClassifier] = None


def get_classifier() -> ClinicalDomainClassifier:
    """Get global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = ClinicalDomainClassifier()
    return _classifier


def classify_domain(lab_data: Dict[str, Any]) -> DomainClassificationResult:
    """Classify risk domain from lab data."""
    return get_classifier().classify(lab_data)


def infer_risk_domain(lab_data: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
    """
    Infer risk domain from lab data.

    Returns:
        Tuple of (domain_name, confidence, full_classification_info)
    """
    result = classify_domain(lab_data)

    if result.primary_domain:
        domain = result.primary_domain
        confidence = result.primary_confidence
    else:
        # Default to "general" if indeterminate
        domain = "general"
        confidence = 0.0

    # Build classification info dict
    info = {
        "inferred_domain": domain,
        "confidence": confidence,
        "auto_inferred": True,
        "multi_system": result.multi_system,
        "involved_systems": result.involved_systems,
        "indeterminate": result.indeterminate,
        "all_domain_scores": [
            {
                "domain": ds.domain,
                "confidence": ds.confidence,
                "has_primary": ds.has_primary_marker,
                "meets_threshold": ds.meets_threshold,
                "primary_markers": ds.primary_markers_found,
                "secondary_markers": ds.secondary_markers_found,
            }
            for ds in result.all_domains
        ],
        "classification_notes": result.classification_notes,
    }

    return domain, confidence, info
