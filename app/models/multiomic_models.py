"""
Pydantic models for Multi-Omic Fusion Engine

Supports unified queries across:
- Genomics (GEO, ClinVar)
- Proteomics (Human Protein Atlas)
- Clinical (MIMIC-IV, eICU, Northwestern ICU)
- Pharmacological (FDA FAERS, ClinicalTrials AACT)
- Population (NHANES)
- Surveillance (WHO, CDC WONDER)
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class DataSource(str, Enum):
    """Available data sources in the fusion engine."""
    # Genomics
    GEO = "geo"
    CLINVAR = "clinvar"
    # Proteomics
    HPA = "human_protein_atlas"
    # Clinical
    MIMIC = "mimic"
    EICU = "eicu"
    NORTHWESTERN = "northwestern"
    # Pharmacological
    FAERS = "faers"
    AACT = "aact"
    # Population
    NHANES = "nhanes"
    # Surveillance
    WHO = "who"
    CDC_WONDER = "cdc_wonder"


class OmicLayer(str, Enum):
    """Biological/data layers for multi-omic integration."""
    GENOMIC = "genomic"
    TRANSCRIPTOMIC = "transcriptomic"
    PROTEOMIC = "proteomic"
    METABOLOMIC = "metabolomic"
    CLINICAL = "clinical"
    PHARMACOLOGICAL = "pharmacological"
    EPIDEMIOLOGICAL = "epidemiological"


class QueryType(str, Enum):
    """Types of unified queries."""
    GENE_CENTRIC = "gene_centric"
    DISEASE_CENTRIC = "disease_centric"
    DRUG_CENTRIC = "drug_centric"
    PATIENT_CENTRIC = "patient_centric"
    POPULATION_CENTRIC = "population_centric"


class DataSourceStatus(BaseModel):
    """Status of a data source."""
    source: DataSource
    available: bool
    path: str
    file_count: int = 0
    last_checked: Optional[datetime] = None
    layer: OmicLayer
    description: str = ""


class UnifiedQueryRequest(BaseModel):
    """Request for unified multi-omic query."""
    query_type: QueryType = Field(..., description="Type of query to perform")

    # Gene-centric parameters
    genes: Optional[List[str]] = Field(None, description="Gene symbols for gene-centric query")

    # Disease-centric parameters
    disease: Optional[str] = Field(None, description="Disease name or ICD code")
    icd_codes: Optional[List[str]] = Field(None, description="ICD-10 codes")

    # Drug-centric parameters
    drug: Optional[str] = Field(None, description="Drug name")
    drug_class: Optional[str] = Field(None, description="Drug class/category")

    # Patient-centric parameters
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    cohort: Optional[str] = Field(None, description="Cohort name")

    # Data source filters
    sources: Optional[List[DataSource]] = Field(None, description="Limit to specific data sources")
    layers: Optional[List[OmicLayer]] = Field(None, description="Limit to specific omic layers")

    # Result options
    max_results_per_source: int = Field(100, ge=1, le=1000)
    include_metadata: bool = Field(True)


class SourceResult(BaseModel):
    """Results from a single data source."""
    source: DataSource
    layer: OmicLayer
    query_successful: bool
    record_count: int
    data: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class UnifiedQueryResponse(BaseModel):
    """Response from unified multi-omic query."""
    query_type: QueryType
    query_parameters: Dict[str, Any]
    sources_queried: List[DataSource]
    results: List[SourceResult]
    cross_references: List[Dict[str, Any]]
    summary: Dict[str, Any]
    execution_time_ms: float


class FusionAnalysisRequest(BaseModel):
    """Request for multi-omic fusion analysis."""
    target_gene: Optional[str] = Field(None, description="Primary gene of interest")
    target_disease: Optional[str] = Field(None, description="Disease of interest")

    include_genomic: bool = Field(True)
    include_proteomic: bool = Field(True)
    include_clinical: bool = Field(True)
    include_pharmacological: bool = Field(True)

    correlation_analysis: bool = Field(True, description="Perform cross-layer correlation")
    pathway_enrichment: bool = Field(False, description="Perform pathway analysis")


class FusionAnalysisResponse(BaseModel):
    """Response from multi-omic fusion analysis."""
    target: str
    layers_analyzed: List[OmicLayer]

    # Per-layer results
    genomic_summary: Optional[Dict[str, Any]] = None
    proteomic_summary: Optional[Dict[str, Any]] = None
    clinical_summary: Optional[Dict[str, Any]] = None
    pharmacological_summary: Optional[Dict[str, Any]] = None

    # Cross-layer analysis
    cross_layer_correlations: List[Dict[str, Any]]
    integrated_score: float
    confidence: float

    # Recommendations
    biomarker_candidates: List[str]
    drug_candidates: List[str]
    clinical_implications: List[str]


class CrossReferenceResult(BaseModel):
    """Cross-reference between data sources."""
    source_a: DataSource
    source_b: DataSource
    link_type: str  # e.g., "gene", "disease", "drug"
    link_value: str
    confidence: float
    evidence: List[str]
