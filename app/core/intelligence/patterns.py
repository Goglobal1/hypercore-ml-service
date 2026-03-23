"""
Universal Pattern Data Structures

Every module reports patterns using these structures.
This enables cross-domain correlation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class PatternType(Enum):
    # Trajectory patterns
    RATE_OF_CHANGE = "rate_of_change"
    INFLECTION_POINT = "inflection_point"
    TRAJECTORY_SHAPE = "trajectory_shape"
    FORECAST = "forecast"

    # Clinical domain patterns
    SEPSIS = "sepsis"
    CARDIAC = "cardiac"
    RENAL = "renal"
    HEPATIC = "hepatic"
    RESPIRATORY = "respiratory"
    METABOLIC = "metabolic"
    MULTI_ORGAN = "multi_organ"

    # Genomic patterns
    PATHOGENIC_VARIANT = "pathogenic_variant"
    PHARMACOGENOMIC = "pharmacogenomic"
    RISK_ALLELE = "risk_allele"

    # Pharma patterns
    DRUG_INTERACTION = "drug_interaction"
    METABOLISM_VARIANT = "metabolism_variant"
    DOSING_ALERT = "dosing_alert"

    # Pathogen patterns
    INFECTION_SIGNAL = "infection_signal"
    RESISTANCE_PATTERN = "resistance_pattern"
    OUTBREAK_CLUSTER = "outbreak_cluster"

    # Alert patterns
    STATE_TRANSITION = "state_transition"
    ESCALATION = "escalation"

    # Surveillance patterns
    POPULATION_TREND = "population_trend"
    ANOMALY_CLUSTER = "anomaly_cluster"


class PatternSource(Enum):
    TRAJECTORY = "trajectory"
    EARLY_RISK = "early_risk"
    ANALYSIS = "analysis"
    GENOMICS = "genomics"
    PHARMA = "pharma"
    PATHOGEN = "pathogen"
    MULTIOMIC = "multiomic"
    ALERT_SYSTEM = "alert_system"
    SURVEILLANCE = "surveillance"
    TRIAL_RESCUE = "trial_rescue"
    DOMAIN_CLASSIFIER = "domain_classifier"


@dataclass
class Pattern:
    """Base pattern structure - ALL patterns inherit from this."""
    patient_id: str
    pattern_type: PatternType
    source: PatternSource
    confidence: float
    severity: float
    id: str = ""
    detected_at: datetime = field(default_factory=datetime.now)
    onset_days_ago: Optional[float] = None
    predicted_days_to_event: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    contributing_factors: List[str] = field(default_factory=list)
    related_biomarkers: List[str] = field(default_factory=list)
    related_genes: List[str] = field(default_factory=list)
    related_drugs: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.source.value}_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "pattern_type": self.pattern_type.value,
            "source": self.source.value,
            "confidence": self.confidence,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
            "onset_days_ago": self.onset_days_ago,
            "predicted_days_to_event": self.predicted_days_to_event,
            "evidence": self.evidence,
            "contributing_factors": self.contributing_factors,
            "related_biomarkers": self.related_biomarkers,
            "related_genes": self.related_genes,
            "related_drugs": self.related_drugs,
            "recommendations": self.recommendations
        }


@dataclass
class TrajectoryPattern(Pattern):
    """Pattern from trajectory analysis."""
    biomarker: str = ""
    rate_of_change: float = 0.0
    acceleration: float = 0.0
    z_score: float = 0.0
    days_of_trend: int = 0
    trajectory_phase: str = "unknown"
    inflection_day: Optional[int] = None
    forecast_threshold_crossing: Optional[float] = None

    def __post_init__(self):
        if not self.source:
            self.source = PatternSource.TRAJECTORY
        super().__post_init__()


@dataclass
class GenomicPattern(Pattern):
    """Pattern from genomics module."""
    gene: str = ""
    variant: str = ""
    classification: str = ""
    clinical_significance: str = ""
    associated_conditions: List[str] = field(default_factory=list)
    drug_implications: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.source:
            self.source = PatternSource.GENOMICS
        super().__post_init__()


@dataclass
class PharmaPattern(Pattern):
    """Pattern from pharma module."""
    drug_a: str = ""
    drug_b: Optional[str] = None
    interaction_type: str = ""
    effect: str = ""
    mechanism: str = ""
    management: str = ""
    affected_gene: Optional[str] = None
    metabolism_status: Optional[str] = None

    def __post_init__(self):
        if not self.source:
            self.source = PatternSource.PHARMA
        super().__post_init__()


@dataclass
class PathogenPattern(Pattern):
    """Pattern from pathogen module."""
    pathogen: Optional[str] = None
    infection_type: str = ""
    resistance_genes: List[str] = field(default_factory=list)
    outbreak_cluster_id: Optional[str] = None
    transmission_risk: float = 0.0

    def __post_init__(self):
        if not self.source:
            self.source = PatternSource.PATHOGEN
        super().__post_init__()


@dataclass
class MultiomicPattern(Pattern):
    """Pattern from multiomic integration."""
    integrated_score: float = 0.0
    contributing_omics: List[str] = field(default_factory=list)
    biomarker_candidates: List[str] = field(default_factory=list)
    pathway_enrichment: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.source:
            self.source = PatternSource.MULTIOMIC
        super().__post_init__()


@dataclass
class AlertPattern(Pattern):
    """Pattern from alert system."""
    alert_state: str = ""
    previous_state: Optional[str] = None
    state_duration_hours: float = 0.0
    escalation_count: int = 0
    alert_type: str = ""

    def __post_init__(self):
        if not self.source:
            self.source = PatternSource.ALERT_SYSTEM
        super().__post_init__()


@dataclass
class SurveillancePattern(Pattern):
    """Pattern from population surveillance."""
    affected_count: int = 0
    location: Optional[str] = None
    trend_direction: str = ""
    baseline_rate: float = 0.0
    current_rate: float = 0.0
    statistical_significance: float = 0.0

    def __post_init__(self):
        if not self.source:
            self.source = PatternSource.SURVEILLANCE
        super().__post_init__()


@dataclass
class ClinicalPattern(Pattern):
    """Aggregated clinical pattern from domain classification."""
    domain: str = ""
    domain_confidence: float = 0.0
    primary_markers: List[str] = field(default_factory=list)
    secondary_markers: List[str] = field(default_factory=list)
    missing_markers: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.source:
            self.source = PatternSource.DOMAIN_CLASSIFIER
        super().__post_init__()
