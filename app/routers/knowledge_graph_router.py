"""
Knowledge Graph API Router for HyperCore

Provides unified access to biomedical knowledge graphs:
- PrimeKG: 8.1M relationships across diseases, genes, drugs
- Hetionet: 2.25M edges connecting compounds, diseases, genes

Endpoints support cross-validation between knowledge sources.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from app.models.knowledge_graph_models import (
    KGHealthResponse,
    DrugTargetsResponse,
    DrugTarget,
    DiseaseGenesResponse,
    DiseaseGene,
    GeneDiseaseResponse,
    DrugIndicationsResponse,
    DrugDisease,
    ProteinInteractionsResponse,
    ProteinInteraction,
    DrugSideEffectsResponse,
    SideEffect,
    GenePathwaysResponse,
    Pathway,
    SearchResponse,
    SearchResult,
    EntitySummaryResponse,
    ConnectionsRequest,
    ConnectionsResponse,
    KnowledgeGraphSource,
    EntityType,
)
from app.data import get_kg_manager

router = APIRouter(prefix="/kg", tags=["knowledge-graph"])


@router.get("/health", response_model=KGHealthResponse)
async def kg_health():
    """
    Check knowledge graph module health and data availability.

    Returns availability status and statistics for PrimeKG and Hetionet.
    """
    kg = get_kg_manager()
    stats = kg.get_stats()

    return KGHealthResponse(
        status="healthy" if kg.any_available else "no_data",
        available_graphs=stats.get("available_graphs", {}),
        primekg_stats=stats.get("primekg"),
        hetionet_stats=stats.get("hetionet"),
    )


@router.get("/drug/{drug_name}/targets", response_model=DrugTargetsResponse)
async def get_drug_targets(
    drug_name: str,
    source: KnowledgeGraphSource = Query(KnowledgeGraphSource.ALL, description="KG source to query"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results")
):
    """
    Get protein/gene targets for a drug.

    Queries PrimeKG and Hetionet for drug-target interactions.
    Cross-validated targets (found in both KGs) are prioritized.

    Args:
        drug_name: Drug name (e.g., Metformin, Aspirin)
        source: Knowledge graph source (primekg, hetionet, or all)
        limit: Maximum number of targets to return

    Returns:
        List of drug targets with validation information
    """
    kg = get_kg_manager()

    sources = None if source == KnowledgeGraphSource.ALL else [source.value]
    targets = kg.get_drug_targets(drug_name, sources=sources)[:limit]

    cross_validated = sum(1 for t in targets if len(t.get("validated_in", [])) > 1)

    return DrugTargetsResponse(
        drug=drug_name,
        targets=[DrugTarget(**t) for t in targets],
        total_count=len(targets),
        cross_validated_count=cross_validated,
    )


@router.get("/disease/{disease_name}/genes", response_model=DiseaseGenesResponse)
async def get_disease_genes(
    disease_name: str,
    source: KnowledgeGraphSource = Query(KnowledgeGraphSource.ALL, description="KG source to query"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results")
):
    """
    Get genes associated with a disease.

    Queries knowledge graphs for disease-gene associations.
    Cross-validated genes are prioritized.

    Args:
        disease_name: Disease name (e.g., diabetes, breast cancer)
        source: Knowledge graph source
        limit: Maximum results

    Returns:
        List of associated genes
    """
    kg = get_kg_manager()

    sources = None if source == KnowledgeGraphSource.ALL else [source.value]
    genes = kg.get_disease_genes(disease_name, sources=sources)[:limit]

    cross_validated = sum(1 for g in genes if len(g.get("validated_in", [])) > 1)

    return DiseaseGenesResponse(
        disease=disease_name,
        genes=[DiseaseGene(**g) for g in genes],
        total_count=len(genes),
        cross_validated_count=cross_validated,
    )


@router.get("/gene/{gene_name}/diseases", response_model=GeneDiseaseResponse)
async def get_gene_diseases(
    gene_name: str,
    source: KnowledgeGraphSource = Query(KnowledgeGraphSource.ALL, description="KG source to query"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results")
):
    """
    Get diseases associated with a gene.

    Args:
        gene_name: Gene symbol (e.g., BRCA1, TP53, APOE)
        source: Knowledge graph source
        limit: Maximum results

    Returns:
        List of associated diseases
    """
    kg = get_kg_manager()

    sources = None if source == KnowledgeGraphSource.ALL else [source.value]
    diseases = kg.get_gene_diseases(gene_name, sources=sources)[:limit]

    return GeneDiseaseResponse(
        gene=gene_name,
        diseases=[DrugDisease(
            disease_id=d.get("disease_id", ""),
            disease_name=d.get("disease_name", ""),
            relation=d.get("relation", ""),
            source=d.get("source", ""),
            validated_in=d.get("validated_in", []),
        ) for d in diseases],
        total_count=len(diseases),
    )


@router.get("/drug/{drug_name}/indications", response_model=DrugIndicationsResponse)
async def get_drug_indications(
    drug_name: str,
    limit: int = Query(100, ge=1, le=500, description="Maximum results")
):
    """
    Get disease indications for a drug.

    Returns diseases that the drug treats or palliates.

    Args:
        drug_name: Drug name
        limit: Maximum results

    Returns:
        List of disease indications
    """
    kg = get_kg_manager()
    indications = kg.get_drug_diseases(drug_name)[:limit]

    return DrugIndicationsResponse(
        drug=drug_name,
        indications=[DrugDisease(
            disease_id=i.get("disease_id", ""),
            disease_name=i.get("disease_name", ""),
            relation=i.get("relation", ""),
            source=i.get("source", ""),
            validated_in=i.get("validated_in", []),
        ) for i in indications],
        total_count=len(indications),
    )


@router.get("/gene/{gene_name}/interactions", response_model=ProteinInteractionsResponse)
async def get_protein_interactions(
    gene_name: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum results")
):
    """
    Get protein-protein interactions for a gene.

    Returns genes/proteins that interact with the query gene.
    Cross-validated interactions are prioritized.

    Args:
        gene_name: Gene symbol
        limit: Maximum results

    Returns:
        List of protein interactions
    """
    kg = get_kg_manager()
    interactions = kg.get_protein_interactions(gene_name, limit=limit)

    cross_validated = sum(1 for i in interactions if len(i.get("validated_in", [])) > 1)

    return ProteinInteractionsResponse(
        gene=gene_name,
        interactions=[ProteinInteraction(**i) for i in interactions],
        total_count=len(interactions),
        cross_validated_count=cross_validated,
    )


@router.get("/drug/{drug_name}/side-effects", response_model=DrugSideEffectsResponse)
async def get_drug_side_effects(
    drug_name: str,
    limit: int = Query(100, ge=1, le=500, description="Maximum results")
):
    """
    Get side effects for a drug.

    Data sourced from Hetionet (SIDER database).

    Args:
        drug_name: Drug name
        limit: Maximum results

    Returns:
        List of side effects
    """
    kg = get_kg_manager()
    side_effects = kg.get_drug_side_effects(drug_name)[:limit]

    return DrugSideEffectsResponse(
        drug=drug_name,
        side_effects=[SideEffect(
            side_effect_id=se.get("side_effect_id", ""),
            side_effect_name=se.get("side_effect_name", ""),
            source=se.get("source", "hetionet"),
        ) for se in side_effects],
        total_count=len(side_effects),
    )


@router.get("/gene/{gene_name}/pathways", response_model=GenePathwaysResponse)
async def get_gene_pathways(
    gene_name: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum results")
):
    """
    Get biological pathways a gene participates in.

    Data sourced from Hetionet (Reactome pathways).

    Args:
        gene_name: Gene symbol
        limit: Maximum results

    Returns:
        List of pathways
    """
    kg = get_kg_manager()
    pathways = kg.get_gene_pathways(gene_name)[:limit]

    return GenePathwaysResponse(
        gene=gene_name,
        pathways=[Pathway(
            pathway_id=p.get("pathway_id", ""),
            pathway_name=p.get("pathway_name", ""),
            source=p.get("source", "hetionet"),
        ) for p in pathways],
        total_count=len(pathways),
    )


@router.get("/search", response_model=SearchResponse)
async def search_entities(
    q: str = Query(..., min_length=2, description="Search query"),
    type: Optional[EntityType] = Query(None, description="Filter by entity type"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
):
    """
    Search for entities across knowledge graphs.

    Searches by name (partial match) across PrimeKG and Hetionet.

    Args:
        q: Search query (minimum 2 characters)
        type: Optional filter by entity type (gene, drug, disease)
        limit: Maximum results

    Returns:
        List of matching entities
    """
    kg = get_kg_manager()

    entity_type = type.value if type else None
    results = kg.search(q, entity_type=entity_type, limit=limit)

    return SearchResponse(
        query=q,
        results=[SearchResult(
            id=r.get("id", ""),
            name=r.get("name", ""),
            type=r.get("type", ""),
            source=r.get("source", ""),
        ) for r in results],
        total_count=len(results),
    )


@router.get("/entity/{entity_name}/summary", response_model=EntitySummaryResponse)
async def get_entity_summary(entity_name: str):
    """
    Get comprehensive summary of an entity from all knowledge graphs.

    Automatically detects entity type (gene, drug, disease) and returns
    relevant information from all available knowledge sources.

    Args:
        entity_name: Entity name (gene symbol, drug name, or disease name)

    Returns:
        Comprehensive entity information including related entities
    """
    kg = get_kg_manager()
    summary = kg.get_entity_summary(entity_name)

    return EntitySummaryResponse(
        name=summary.get("name", entity_name),
        type=summary.get("type"),
        found_in=summary.get("found_in", []),
        primekg_id=summary.get("primekg_id"),
        hetionet_id=summary.get("hetionet_id"),
        targets=summary.get("targets"),
        indications=summary.get("indications"),
        side_effects=summary.get("side_effects"),
        diseases=summary.get("diseases"),
        interactions=summary.get("interactions"),
        pathways=summary.get("pathways"),
        genes=summary.get("genes"),
    )


@router.post("/connections", response_model=ConnectionsResponse)
async def find_connections(request: ConnectionsRequest):
    """
    Find connections between two entities in the knowledge graph.

    Uses breadth-first search to find paths connecting two entities.

    Args:
        request: Connection request with entity names and max depth

    Returns:
        Paths connecting the entities from each knowledge graph
    """
    kg = get_kg_manager()
    paths = kg.find_connections(request.entity1, request.entity2, max_depth=request.max_depth)

    total_paths = sum(len(p) for p in paths.values())

    return ConnectionsResponse(
        entity1=request.entity1,
        entity2=request.entity2,
        paths=paths,
        total_paths=total_paths,
    )


@router.get("/stats")
async def get_kg_stats():
    """
    Get detailed statistics for all knowledge graphs.

    Returns node counts, edge counts, and relationship type distributions.
    """
    kg = get_kg_manager()
    return kg.get_stats()
