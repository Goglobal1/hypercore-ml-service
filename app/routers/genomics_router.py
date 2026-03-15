"""
Genomics API Router for HyperCore

Provides endpoints for:
- Gene expression lookup from GEO datasets
- Variant lookup from ClinVar
- Comprehensive genomics analysis with clinical correlation
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from app.models.genomics_models import (
    GeneExpressionRequest,
    GeneExpressionResponse,
    VariantLookupRequest,
    VariantLookupResponse,
    GenomicsAnalysisRequest,
    GenomicsAnalysisResponse,
    ClinVarVariant,
    GeneExpressionSample,
)
from app.core.genomics_integration import (
    get_gene_expression,
    get_gene_variants,
    analyze_genomics,
    get_geo_parser,
)

router = APIRouter(prefix="/genomics", tags=["genomics"])


@router.get("/health")
async def genomics_health():
    """Check genomics module health and data availability."""
    parser = get_geo_parser()
    series = parser.list_available_series()

    return {
        "status": "healthy",
        "geo_series_available": len(series),
        "sample_series": series[:5] if series else [],
        "module": "genomics_integration",
        "version": "1.0.0"
    }


@router.get("/expression/{gene}", response_model=GeneExpressionResponse)
async def get_expression(
    gene: str,
    max_files: int = Query(5, ge=1, le=50, description="Max GEO files to scan")
):
    """
    Get gene expression data from GEO datasets.

    Searches available GEO series_matrix files for expression data
    matching the specified gene symbol.

    Args:
        gene: Gene symbol (e.g., APOE, BRCA1)
        max_files: Maximum number of GEO files to scan

    Returns:
        Expression data with statistics across datasets
    """
    try:
        result = get_gene_expression(gene.upper(), max_files)

        # Convert to response model format
        samples = [
            GeneExpressionSample(
                gene_symbol=gene.upper(),
                probe_id=s["probe_id"],
                sample_id=f"aggregated_{i}",
                expression_value=s["mean_expression"],
                series_id=s["series_id"],
                platform=s.get("platform")
            )
            for i, s in enumerate(result.get("samples", []))
        ]

        return GeneExpressionResponse(
            gene=gene.upper(),
            probe_ids=result.get("probe_ids", []),
            samples=samples,
            statistics=result.get("statistics", {}),
            series_count=result.get("series_count", 0),
            sample_count=result.get("sample_count", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Expression lookup failed: {str(e)}")


@router.get("/variants/{gene}", response_model=VariantLookupResponse)
async def get_variants(
    gene: str,
    pathogenic_only: bool = Query(True, description="Filter to pathogenic variants only"),
    max_variants: int = Query(100, ge=1, le=1000, description="Maximum variants to return")
):
    """
    Get ClinVar variants for a gene.

    Retrieves variant information from ClinVar including clinical
    significance, associated phenotypes, and genomic coordinates.

    Args:
        gene: Gene symbol (e.g., BRCA1, TP53)
        pathogenic_only: If True, return only pathogenic/likely pathogenic variants
        max_variants: Maximum number of variants to return

    Returns:
        List of variants with phenotype associations
    """
    try:
        result = get_gene_variants(gene.upper(), pathogenic_only, max_variants)

        variants = [
            ClinVarVariant(
                allele_id=v["allele_id"],
                gene_symbol=v["gene_symbol"],
                gene_id=v["gene_id"],
                variant_name=v["variant_name"],
                variant_type=v["variant_type"],
                clinical_significance=v["clinical_significance"],
                phenotype_list=v["phenotypes"],
                review_status=v.get("review_status", ""),
                chromosome=v.get("chromosome"),
                position_start=v.get("start"),
                position_end=v.get("end"),
                rs_id=v.get("rs_id")
            )
            for v in result.get("variants", [])
        ]

        return VariantLookupResponse(
            gene=gene.upper(),
            variants=variants,
            pathogenic_count=result.get("pathogenic_count", 0),
            total_count=result.get("total_count", 0),
            phenotypes=result.get("phenotypes", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Variant lookup failed: {str(e)}")


@router.post("/analyze", response_model=GenomicsAnalysisResponse)
async def analyze(request: GenomicsAnalysisRequest):
    """
    Comprehensive genomics analysis with clinical correlation.

    Analyzes multiple genes, combining:
    - ClinVar variant data and phenotype associations
    - GEO expression patterns
    - Clinical trajectory correlations from the specified cohort

    Args:
        request: Analysis request with gene list and options

    Returns:
        Comprehensive analysis with gene-phenotype associations,
        expression patterns, variant impacts, and clinical correlations
    """
    if not request.genes:
        raise HTTPException(status_code=400, detail="At least one gene must be specified")

    if len(request.genes) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 genes per request")

    try:
        result = analyze_genomics(
            genes=[g.upper() for g in request.genes],
            cohort=request.cohort,
            include_variants=request.include_variants,
            include_expression=request.include_expression,
            max_expression_files=request.max_expression_files
        )

        return GenomicsAnalysisResponse(
            genes_analyzed=result.get("genes_analyzed", []),
            gene_phenotype_associations=result.get("gene_phenotype_associations", []),
            expression_patterns=result.get("expression_patterns", []),
            variant_impacts=result.get("variant_impacts", []),
            clinical_correlations=result.get("clinical_correlations", {}).get("correlations", []),
            cohort_overlap=result.get("cohort_overlap", {}),
            analysis_metadata=result.get("analysis_metadata", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/series")
async def list_series(limit: int = Query(50, ge=1, le=200)):
    """
    List available GEO series in the dataset.

    Returns a list of GEO series IDs that can be queried
    for gene expression data.
    """
    parser = get_geo_parser()
    series = parser.list_available_series()

    return {
        "total_series": len(series),
        "series": series[:limit]
    }


@router.get("/phenotype-icd10/{phenotype}")
async def get_phenotype_icd10(phenotype: str):
    """
    Get ICD-10 code mappings for a phenotype term.

    Useful for understanding how genomic phenotypes
    map to clinical diagnoses.
    """
    from app.core.genomics_integration import PHENOTYPE_ICD10_MAP

    phenotype_lower = phenotype.lower()
    matches = {}

    for key, codes in PHENOTYPE_ICD10_MAP.items():
        if phenotype_lower in key or key in phenotype_lower:
            matches[key] = codes

    return {
        "query": phenotype,
        "matches": matches,
        "all_mappings_count": len(PHENOTYPE_ICD10_MAP)
    }
