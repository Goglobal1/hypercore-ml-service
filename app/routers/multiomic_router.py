"""
Multi-Omic Fusion API Router for HyperCore

Provides endpoints for:
- Data source status monitoring
- Unified cross-source queries
- Multi-omic fusion analysis
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from app.models.multiomic_models import (
    DataSource,
    OmicLayer,
    QueryType,
    UnifiedQueryRequest,
    UnifiedQueryResponse,
    FusionAnalysisRequest,
    FusionAnalysisResponse,
    SourceResult,
)
from app.core.multiomic_fusion import (
    get_source_status,
    unified_query,
    fusion_analysis,
    get_fusion_engine,
)

router = APIRouter(prefix="/multiomic", tags=["multiomic"])


@router.get("/health")
async def multiomic_health():
    """Check multi-omic fusion module health and data source availability."""
    statuses = get_source_status()

    available_count = sum(1 for s in statuses if s["available"])
    total_files = sum(s.get("file_count", 0) for s in statuses if s["available"])

    return {
        "status": "healthy",
        "sources_available": available_count,
        "total_sources": len(statuses),
        "total_files_indexed": total_files,
        "module": "multiomic_fusion",
        "version": "1.0.0"
    }


@router.get("/sources")
async def list_sources():
    """
    List all data sources and their availability status.

    Returns detailed information about each data source including:
    - Availability status
    - File counts
    - Omic layer classification
    """
    statuses = get_source_status()

    return {
        "sources": statuses,
        "summary": {
            "total": len(statuses),
            "available": sum(1 for s in statuses if s["available"]),
            "by_layer": _group_by_layer(statuses)
        }
    }


def _group_by_layer(statuses):
    """Group sources by omic layer."""
    layers = {}
    for s in statuses:
        layer = s.get("layer", "unknown")
        if layer not in layers:
            layers[layer] = {"count": 0, "sources": []}
        layers[layer]["count"] += 1
        if s["available"]:
            layers[layer]["sources"].append(s["source"])
    return layers


@router.get("/query/gene/{gene}")
async def query_gene(
    gene: str,
    sources: Optional[str] = Query(None, description="Comma-separated source names"),
    max_results: int = Query(50, ge=1, le=500)
):
    """
    Gene-centric query across multiple omic layers.

    Retrieves data about a gene from:
    - GEO (transcriptomic expression)
    - ClinVar (genomic variants)
    - HPA (proteomic expression)

    Args:
        gene: Gene symbol (e.g., APOE, BRCA1)
        sources: Optional comma-separated list of sources to query
        max_results: Maximum results per source
    """
    source_list = sources.split(",") if sources else None

    result = unified_query(
        query_type="gene_centric",
        genes=[gene.upper()],
        sources=source_list,
        max_results_per_source=max_results
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/query/disease/{disease}")
async def query_disease(
    disease: str,
    max_results: int = Query(50, ge=1, le=500)
):
    """
    Disease-centric query across data sources.

    Retrieves:
    - Related genes
    - Clinical trials (AACT)
    - WHO surveillance data

    Args:
        disease: Disease name (e.g., alzheimer, diabetes, cancer)
        max_results: Maximum results per source
    """
    result = unified_query(
        query_type="disease_centric",
        disease=disease,
        max_results_per_source=max_results
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.get("/query/drug/{drug}")
async def query_drug(
    drug: str,
    max_results: int = Query(50, ge=1, le=500)
):
    """
    Drug-centric query across data sources.

    Retrieves:
    - Pharmacogenomic gene associations
    - FDA FAERS adverse events
    - Clinical trials (AACT)

    Args:
        drug: Drug name (e.g., warfarin, metformin)
        max_results: Maximum results per source
    """
    result = unified_query(
        query_type="drug_centric",
        drug=drug,
        max_results_per_source=max_results
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/query/unified")
async def unified_query_endpoint(request: UnifiedQueryRequest):
    """
    Perform unified multi-omic query.

    Supports multiple query types:
    - gene_centric: Query by gene symbols
    - disease_centric: Query by disease/condition
    - drug_centric: Query by drug name

    Args:
        request: UnifiedQueryRequest with query parameters

    Returns:
        Results aggregated across selected data sources
    """
    query_type = request.query_type.value

    if query_type == "gene_centric":
        if not request.genes:
            raise HTTPException(status_code=400, detail="genes required for gene_centric query")
        result = unified_query(
            query_type=query_type,
            genes=[g.upper() for g in request.genes],
            sources=[s.value for s in request.sources] if request.sources else None,
            max_results_per_source=request.max_results_per_source
        )
    elif query_type == "disease_centric":
        if not request.disease:
            raise HTTPException(status_code=400, detail="disease required for disease_centric query")
        result = unified_query(
            query_type=query_type,
            disease=request.disease,
            max_results_per_source=request.max_results_per_source
        )
    elif query_type == "drug_centric":
        if not request.drug:
            raise HTTPException(status_code=400, detail="drug required for drug_centric query")
        result = unified_query(
            query_type=query_type,
            drug=request.drug,
            max_results_per_source=request.max_results_per_source
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported query type: {query_type}")

    return result


@router.post("/analyze/fusion")
async def fusion_analysis_endpoint(request: FusionAnalysisRequest):
    """
    Comprehensive multi-omic fusion analysis.

    Integrates data across multiple omic layers to provide:
    - Cross-layer correlations
    - Biomarker candidates
    - Drug candidates
    - Clinical implications

    Supports analysis by:
    - Target gene (pharmacogenomics, expression, variants)
    - Target disease (related genes, trials, surveillance)

    Args:
        request: FusionAnalysisRequest with analysis parameters

    Returns:
        Integrated analysis with recommendations
    """
    if not request.target_gene and not request.target_disease:
        raise HTTPException(
            status_code=400,
            detail="At least one of target_gene or target_disease must be specified"
        )

    result = fusion_analysis(
        target_gene=request.target_gene.upper() if request.target_gene else None,
        target_disease=request.target_disease,
        include_genomic=request.include_genomic,
        include_proteomic=request.include_proteomic,
        include_clinical=request.include_clinical,
        include_pharmacological=request.include_pharmacological
    )

    return FusionAnalysisResponse(
        target=request.target_gene or request.target_disease or "",
        layers_analyzed=[OmicLayer(l) for l in result.get("layers_analyzed", [])],
        genomic_summary=result.get("genomic_summary"),
        proteomic_summary=result.get("proteomic_summary"),
        clinical_summary=result.get("clinical_summary"),
        pharmacological_summary=result.get("pharmacological_summary"),
        cross_layer_correlations=result.get("cross_layer_correlations", []),
        integrated_score=result.get("integrated_score", 0.0),
        confidence=result.get("confidence", 0.0),
        biomarker_candidates=result.get("biomarker_candidates", []),
        drug_candidates=result.get("drug_candidates", []),
        clinical_implications=result.get("clinical_implications", [])
    )


@router.get("/layers")
async def list_omic_layers():
    """List available omic layers and their associated data sources."""
    return {
        "layers": [
            {
                "layer": "genomic",
                "description": "DNA-level data (variants, mutations)",
                "sources": ["clinvar"]
            },
            {
                "layer": "transcriptomic",
                "description": "Gene expression data (mRNA levels)",
                "sources": ["geo"]
            },
            {
                "layer": "proteomic",
                "description": "Protein expression and localization",
                "sources": ["hpa"]
            },
            {
                "layer": "clinical",
                "description": "Patient clinical data",
                "sources": ["mimic", "eicu", "northwestern"]
            },
            {
                "layer": "pharmacological",
                "description": "Drug-related data",
                "sources": ["faers", "aact"]
            },
            {
                "layer": "epidemiological",
                "description": "Population-level surveillance",
                "sources": ["nhanes", "who", "cdc_wonder"]
            }
        ]
    }


@router.get("/pharmacogenomics/{drug}")
async def get_pharmacogenomics(drug: str):
    """
    Get pharmacogenomic associations for a drug.

    Returns genes that affect drug response/metabolism.
    """
    from app.core.multiomic_fusion import DRUG_GENE_MAP

    drug_lower = drug.lower()
    associations = {}

    for drug_name, genes in DRUG_GENE_MAP.items():
        if drug_lower in drug_name.lower():
            associations[drug_name] = genes

    return {
        "query": drug,
        "associations": associations,
        "total_drugs_indexed": len(DRUG_GENE_MAP)
    }


@router.get("/gene-disease/{gene}")
async def get_gene_disease_associations(gene: str):
    """
    Get disease associations for a gene.

    Returns diseases linked to the specified gene.
    """
    from app.core.multiomic_fusion import GENE_DISEASE_MAP

    gene_upper = gene.upper()

    if gene_upper in GENE_DISEASE_MAP:
        return {
            "gene": gene_upper,
            "associated_diseases": GENE_DISEASE_MAP[gene_upper],
            "association_count": len(GENE_DISEASE_MAP[gene_upper])
        }
    else:
        return {
            "gene": gene_upper,
            "associated_diseases": [],
            "association_count": 0,
            "note": f"No curated associations found for {gene_upper}"
        }
