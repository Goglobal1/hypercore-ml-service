"""
Pydantic models for Genomics Integration Pipeline
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class ClinicalSignificance(str, Enum):
    """ClinVar clinical significance categories."""
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely pathogenic"
    PATHOGENIC_LIKELY = "Pathogenic/Likely pathogenic"
    UNCERTAIN = "Uncertain significance"
    LIKELY_BENIGN = "Likely benign"
    BENIGN = "Benign"


class GeneExpressionSample(BaseModel):
    """Single gene expression measurement from GEO."""
    gene_symbol: str = Field(..., description="Gene symbol (e.g., APOE)")
    probe_id: str = Field(..., description="Probe/feature ID from array")
    sample_id: str = Field(..., description="GEO sample ID (GSMxxxx)")
    expression_value: float = Field(..., description="Expression value (log2)")
    series_id: str = Field(..., description="GEO series ID (GSExxxx)")
    platform: Optional[str] = Field(None, description="Platform ID (GPLxxxx)")


class GEOSeriesMetadata(BaseModel):
    """Metadata from GEO series_matrix file."""
    series_id: str = Field(..., description="GEO series accession")
    title: str = Field("", description="Study title")
    summary: str = Field("", description="Study summary")
    platform_id: str = Field("", description="Platform ID")
    sample_count: int = Field(0, description="Number of samples")
    organism: str = Field("Homo sapiens", description="Organism")
    pubmed_id: Optional[str] = Field(None, description="PubMed ID")


class ClinVarVariant(BaseModel):
    """Variant record from ClinVar."""
    allele_id: int = Field(..., description="ClinVar allele ID")
    gene_symbol: str = Field(..., description="Gene symbol")
    gene_id: int = Field(..., description="NCBI Gene ID")
    variant_name: str = Field(..., description="Variant name/description")
    variant_type: str = Field(..., description="Variant type (SNV, Deletion, etc.)")
    clinical_significance: str = Field(..., description="Clinical significance")
    phenotype_list: List[str] = Field(default_factory=list, description="Associated phenotypes")
    review_status: str = Field("", description="Review status")
    chromosome: Optional[str] = Field(None, description="Chromosome")
    position_start: Optional[int] = Field(None, description="Start position")
    position_end: Optional[int] = Field(None, description="End position")
    rs_id: Optional[str] = Field(None, description="dbSNP rs ID")


class GenePhenotypeAssociation(BaseModel):
    """Association between gene and clinical phenotype."""
    gene_symbol: str
    phenotype: str
    icd10_codes: List[str] = Field(default_factory=list)
    variant_count: int = 0
    pathogenic_variants: List[str] = Field(default_factory=list)
    evidence_level: str = "variant_association"


class GenomicsAnalysisRequest(BaseModel):
    """Request for comprehensive genomics analysis."""
    genes: List[str] = Field(..., description="List of gene symbols to analyze")
    cohort: str = Field("geriatric", description="Cohort to analyze against")
    include_variants: bool = Field(True, description="Include ClinVar variant lookup")
    include_expression: bool = Field(True, description="Include GEO expression data")
    max_expression_files: int = Field(10, description="Max GEO files to scan")


class GenomicsAnalysisResponse(BaseModel):
    """Response from comprehensive genomics analysis."""
    genes_analyzed: List[str]
    gene_phenotype_associations: List[Dict[str, Any]]
    expression_patterns: List[Dict[str, Any]]
    variant_impacts: List[Dict[str, Any]]
    clinical_correlations: List[Dict[str, Any]]
    cohort_overlap: Dict[str, Any]
    analysis_metadata: Dict[str, Any]


class GeneExpressionRequest(BaseModel):
    """Request for gene expression lookup."""
    gene: str = Field(..., description="Gene symbol to search")
    max_files: int = Field(5, description="Maximum GEO files to scan")
    min_expression: Optional[float] = Field(None, description="Minimum expression threshold")


class GeneExpressionResponse(BaseModel):
    """Response with gene expression data."""
    gene: str
    probe_ids: List[str]
    samples: List[GeneExpressionSample]
    statistics: Dict[str, float]
    series_count: int
    sample_count: int


class VariantLookupRequest(BaseModel):
    """Request for variant lookup."""
    gene: str = Field(..., description="Gene symbol")
    pathogenic_only: bool = Field(True, description="Filter to pathogenic variants only")
    max_variants: int = Field(100, description="Maximum variants to return")


class VariantLookupResponse(BaseModel):
    """Response with variant data."""
    gene: str
    variants: List[ClinVarVariant]
    pathogenic_count: int
    total_count: int
    phenotypes: List[str]
