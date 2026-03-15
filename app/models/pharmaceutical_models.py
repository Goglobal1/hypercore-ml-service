"""
Pydantic models for Drug Response Predictor Pipeline

Supports:
- FDA FAERS adverse event data
- ClinicalTrials.gov AACT data
- Pharmacogenomic predictions
- Drug-drug interactions
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class DrugRole(str, Enum):
    """Drug role codes from FAERS."""
    PRIMARY_SUSPECT = "PS"
    SECONDARY_SUSPECT = "SS"
    CONCOMITANT = "C"
    INTERACTING = "I"


class MetabolizerStatus(str, Enum):
    """Pharmacogenomic metabolizer status."""
    ULTRA_RAPID = "ultra_rapid"
    EXTENSIVE = "extensive"  # Normal
    INTERMEDIATE = "intermediate"
    POOR = "poor"


class AdverseEvent(BaseModel):
    """Single adverse event record from FAERS."""
    primary_id: str = Field(..., description="FAERS primary ID")
    case_id: str = Field(..., description="Case ID")
    drug_name: str = Field(..., description="Drug name")
    active_ingredient: Optional[str] = Field(None, description="Active ingredient")
    reaction: str = Field(..., description="Adverse reaction (preferred term)")
    role: Optional[str] = Field(None, description="Drug role (PS/SS/C/I)")
    route: Optional[str] = Field(None, description="Route of administration")
    dose: Optional[str] = Field(None, description="Dose information")
    outcome: Optional[str] = Field(None, description="Patient outcome")


class DrugProfile(BaseModel):
    """Comprehensive drug profile."""
    drug_name: str
    active_ingredients: List[str] = Field(default_factory=list)
    pharmacogenomic_genes: List[str] = Field(default_factory=list)
    common_adverse_events: List[Dict[str, Any]] = Field(default_factory=list)
    serious_adverse_events: List[Dict[str, Any]] = Field(default_factory=list)
    drug_interactions: List[str] = Field(default_factory=list)
    clinical_trials_count: int = 0
    fda_approval_status: Optional[str] = None


class ClinicalTrial(BaseModel):
    """Clinical trial record from AACT."""
    nct_id: str = Field(..., description="ClinicalTrials.gov NCT ID")
    title: Optional[str] = Field(None, description="Study title")
    status: Optional[str] = Field(None, description="Study status")
    phase: Optional[str] = Field(None, description="Trial phase")
    conditions: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    enrollment: Optional[int] = Field(None, description="Enrollment count")
    start_date: Optional[str] = Field(None)
    completion_date: Optional[str] = Field(None)


class DrugResponseRequest(BaseModel):
    """Request for drug response prediction."""
    drug_name: str = Field(..., description="Drug name to analyze")
    patient_genes: Optional[Dict[str, str]] = Field(
        None,
        description="Patient genetic variants (gene -> variant/status)"
    )
    metabolizer_status: Optional[Dict[str, MetabolizerStatus]] = Field(
        None,
        description="Metabolizer status for relevant genes"
    )
    concurrent_medications: Optional[List[str]] = Field(
        None,
        description="Other medications patient is taking"
    )
    age: Optional[int] = Field(None, description="Patient age")
    weight_kg: Optional[float] = Field(None, description="Patient weight in kg")


class DrugResponsePrediction(BaseModel):
    """Drug response prediction result."""
    drug_name: str
    efficacy_prediction: str  # "standard", "reduced", "enhanced"
    efficacy_confidence: float
    toxicity_risk: str  # "low", "moderate", "high"
    toxicity_confidence: float
    dose_adjustment: Optional[str] = None
    pharmacogenomic_factors: List[Dict[str, Any]] = Field(default_factory=list)
    interaction_warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    evidence_sources: List[str] = Field(default_factory=list)


class InteractionCheckRequest(BaseModel):
    """Request for drug interaction check."""
    drugs: List[str] = Field(..., description="List of drugs to check")
    include_severity: bool = Field(True, description="Include severity assessment")


class DrugInteraction(BaseModel):
    """Drug-drug interaction record."""
    drug_a: str
    drug_b: str
    interaction_type: str  # "pharmacokinetic", "pharmacodynamic", "unknown"
    severity: str  # "minor", "moderate", "major", "contraindicated"
    mechanism: Optional[str] = None
    clinical_effect: Optional[str] = None
    management: Optional[str] = None
    evidence_level: str = "theoretical"


class InteractionCheckResponse(BaseModel):
    """Response from drug interaction check."""
    drugs_checked: List[str]
    interactions_found: List[DrugInteraction]
    interaction_count: int
    max_severity: str
    recommendations: List[str]


class AdverseEventQuery(BaseModel):
    """Query for adverse events."""
    drug_name: str = Field(..., description="Drug name to search")
    reaction_filter: Optional[str] = Field(None, description="Filter by reaction type")
    limit: int = Field(100, ge=1, le=1000)
    include_statistics: bool = Field(True)


class AdverseEventResponse(BaseModel):
    """Response with adverse event data."""
    drug_name: str
    total_reports: int
    adverse_events: List[AdverseEvent]
    event_statistics: Dict[str, int]  # reaction -> count
    serious_event_percentage: float
    common_outcomes: Dict[str, int]


class TrialSearchRequest(BaseModel):
    """Request for clinical trial search."""
    condition: Optional[str] = Field(None, description="Condition/disease to search")
    intervention: Optional[str] = Field(None, description="Intervention/drug to search")
    phase: Optional[str] = Field(None, description="Trial phase filter")
    status: Optional[str] = Field(None, description="Trial status filter")
    limit: int = Field(50, ge=1, le=500)


class TrialSearchResponse(BaseModel):
    """Response with clinical trial search results."""
    query: Dict[str, Any]
    total_trials: int
    trials: List[ClinicalTrial]
    phase_distribution: Dict[str, int]
    status_distribution: Dict[str, int]
