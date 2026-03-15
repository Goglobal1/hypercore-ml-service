"""
Pydantic models for Pathogen Detection Pipeline

Supports:
- WHO surveillance indicators (AMR, mortality, disease trends)
- COVID-19 vaccination data
- CDC WONDER mortality data
- Outbreak detection and tracking
- Pathogen-clinical outcome correlations
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class PathogenType(str, Enum):
    """Types of pathogens tracked."""
    BACTERIAL = "bacterial"
    VIRAL = "viral"
    FUNGAL = "fungal"
    PARASITIC = "parasitic"
    UNKNOWN = "unknown"


class SurveillanceIndicator(str, Enum):
    """WHO surveillance indicator categories."""
    AMR = "antimicrobial_resistance"
    MORTALITY = "mortality"
    INFECTIOUS_DISEASE = "infectious_disease"
    VACCINATION = "vaccination"
    OUTBREAK = "outbreak"


class AlertLevel(str, Enum):
    """Alert levels for pathogen detection."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class WHOIndicator(BaseModel):
    """WHO surveillance indicator data."""
    indicator_id: str
    indicator_code: str
    indicator_name: str
    country: str
    country_code: Optional[str] = None
    year: int
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    category: Optional[str] = None


class OutbreakAlert(BaseModel):
    """Outbreak detection alert."""
    pathogen: str
    pathogen_type: PathogenType
    alert_level: AlertLevel
    affected_regions: List[str]
    case_count: int
    trend: str  # "increasing", "stable", "decreasing"
    detection_date: datetime
    confidence: float
    evidence: List[str]
    recommendations: List[str]


class AMRProfile(BaseModel):
    """Antimicrobial resistance profile."""
    pathogen: str
    antibiotic: str
    resistance_rate: float
    sample_size: int
    country: str
    year: int
    trend: str  # "increasing", "stable", "decreasing"
    alert_level: AlertLevel


class VaccinationCoverage(BaseModel):
    """Vaccination coverage data."""
    disease: str
    country: str
    coverage_percentage: float
    target_population: str
    year: int
    quarter: Optional[int] = None


class MortalityData(BaseModel):
    """Mortality surveillance data."""
    cause: str
    icd_code: Optional[str] = None
    country: str
    year: int
    rate_per_100k: float
    age_group: Optional[str] = None
    sex: Optional[str] = None


class PathogenSearchRequest(BaseModel):
    """Request for pathogen search."""
    pathogen_name: Optional[str] = Field(None, description="Pathogen name to search")
    disease: Optional[str] = Field(None, description="Disease name to search")
    country: Optional[str] = Field(None, description="Filter by country")
    year_from: Optional[int] = Field(None, description="Start year")
    year_to: Optional[int] = Field(None, description="End year")
    indicator_type: Optional[SurveillanceIndicator] = Field(None)
    limit: int = Field(100, ge=1, le=1000)


class PathogenSearchResponse(BaseModel):
    """Response from pathogen search."""
    query: Dict[str, Any]
    indicators: List[WHOIndicator]
    total_results: int
    countries_affected: List[str]
    year_range: List[int]


class OutbreakDetectionRequest(BaseModel):
    """Request for outbreak detection analysis."""
    regions: Optional[List[str]] = Field(None, description="Regions to analyze")
    pathogens: Optional[List[str]] = Field(None, description="Pathogens to monitor")
    threshold_multiplier: float = Field(1.5, description="Alert threshold (std devs)")
    lookback_years: int = Field(5, ge=1, le=20)


class OutbreakDetectionResponse(BaseModel):
    """Response from outbreak detection."""
    alerts: List[OutbreakAlert]
    total_alerts: int
    critical_alerts: int
    regions_analyzed: List[str]
    analysis_period: str


class AMRAnalysisRequest(BaseModel):
    """Request for AMR analysis."""
    pathogen: Optional[str] = Field(None, description="Specific pathogen")
    antibiotic: Optional[str] = Field(None, description="Specific antibiotic")
    country: Optional[str] = Field(None, description="Country filter")
    year: Optional[int] = Field(None, description="Year filter")


class AMRAnalysisResponse(BaseModel):
    """Response from AMR analysis."""
    profiles: List[AMRProfile]
    high_risk_combinations: List[Dict[str, Any]]
    trend_summary: Dict[str, str]
    recommendations: List[str]


class ClinicalPathogenCorrelation(BaseModel):
    """Correlation between pathogen and clinical outcomes."""
    pathogen: str
    clinical_outcome: str
    icd10_code: str
    correlation_strength: float
    sample_size: int
    mortality_rate: Optional[float] = None
    icu_admission_rate: Optional[float] = None
    evidence_sources: List[str]
