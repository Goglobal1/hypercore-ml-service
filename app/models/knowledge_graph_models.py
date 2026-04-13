"""
Pydantic models for Knowledge Graph API endpoints

Supports queries across:
- PrimeKG: 8.1M relationships across diseases, genes, drugs
- Hetionet: 2.25M edges connecting compounds, diseases, genes
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class KnowledgeGraphSource(str, Enum):
    """Available knowledge graph sources."""
    PRIMEKG = "primekg"
    HETIONET = "hetionet"
    ALL = "all"


class EntityType(str, Enum):
    """Entity types in knowledge graphs."""
    GENE = "gene"
    DRUG = "drug"
    DISEASE = "disease"
    PATHWAY = "pathway"
    SIDE_EFFECT = "side_effect"
    ANATOMY = "anatomy"


class DrugTarget(BaseModel):
    """Drug-gene/protein target relationship."""
    gene_id: str = Field(..., description="Gene/protein identifier")
    gene_name: str = Field(..., description="Gene/protein name")
    relation: str = Field(..., description="Relationship type (e.g., target, binds)")
    source: str = Field(..., description="Knowledge graph source")
    validated_in: List[str] = Field(default_factory=list, description="KGs where this was found")


class DiseaseGene(BaseModel):
    """Disease-gene association."""
    gene_id: str = Field(..., description="Gene identifier")
    gene_name: str = Field(..., description="Gene name/symbol")
    relation: str = Field(..., description="Association type")
    source: str = Field(..., description="Knowledge graph source")
    validated_in: List[str] = Field(default_factory=list, description="KGs where this was found")


class DrugDisease(BaseModel):
    """Drug-disease indication/association."""
    disease_id: str = Field(..., description="Disease identifier")
    disease_name: str = Field(..., description="Disease name")
    relation: str = Field(..., description="Relationship (treats, palliates, etc.)")
    source: str = Field(..., description="Knowledge graph source")
    validated_in: List[str] = Field(default_factory=list, description="KGs where this was found")


class ProteinInteraction(BaseModel):
    """Protein-protein interaction."""
    partner_id: str = Field(..., description="Interaction partner ID")
    partner_name: str = Field(..., description="Interaction partner name")
    relation: str = Field(..., description="Interaction type")
    source: str = Field(..., description="Knowledge graph source")
    validated_in: List[str] = Field(default_factory=list, description="KGs where this was found")


class SideEffect(BaseModel):
    """Drug side effect."""
    side_effect_id: str = Field(..., description="Side effect identifier")
    side_effect_name: str = Field(..., description="Side effect name")
    source: str = Field("hetionet", description="Knowledge graph source")


class Pathway(BaseModel):
    """Biological pathway."""
    pathway_id: str = Field(..., description="Pathway identifier")
    pathway_name: str = Field(..., description="Pathway name")
    source: str = Field(..., description="Knowledge graph source")


class SearchResult(BaseModel):
    """Search result from knowledge graph."""
    id: str = Field(..., description="Entity identifier")
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    source: str = Field(..., description="Knowledge graph source")


class KGHealthResponse(BaseModel):
    """Knowledge graph health check response."""
    status: str = Field(..., description="Health status")
    available_graphs: Dict[str, bool] = Field(..., description="Availability of each KG")
    primekg_stats: Optional[Dict[str, Any]] = Field(None, description="PrimeKG statistics")
    hetionet_stats: Optional[Dict[str, Any]] = Field(None, description="Hetionet statistics")


class DrugTargetsResponse(BaseModel):
    """Response for drug targets query."""
    drug: str = Field(..., description="Query drug name")
    targets: List[DrugTarget] = Field(default_factory=list, description="List of targets")
    total_count: int = Field(0, description="Total number of targets")
    cross_validated_count: int = Field(0, description="Targets found in multiple KGs")


class DiseaseGenesResponse(BaseModel):
    """Response for disease genes query."""
    disease: str = Field(..., description="Query disease name")
    genes: List[DiseaseGene] = Field(default_factory=list, description="Associated genes")
    total_count: int = Field(0, description="Total number of genes")
    cross_validated_count: int = Field(0, description="Genes found in multiple KGs")


class GeneDiseaseResponse(BaseModel):
    """Response for gene-disease associations."""
    gene: str = Field(..., description="Query gene name")
    diseases: List[DrugDisease] = Field(default_factory=list, description="Associated diseases")
    total_count: int = Field(0, description="Total number of diseases")


class DrugIndicationsResponse(BaseModel):
    """Response for drug indications query."""
    drug: str = Field(..., description="Query drug name")
    indications: List[DrugDisease] = Field(default_factory=list, description="Disease indications")
    total_count: int = Field(0, description="Total number of indications")


class ProteinInteractionsResponse(BaseModel):
    """Response for protein interactions query."""
    gene: str = Field(..., description="Query gene/protein name")
    interactions: List[ProteinInteraction] = Field(default_factory=list, description="Interactions")
    total_count: int = Field(0, description="Total interactions")
    cross_validated_count: int = Field(0, description="Interactions in multiple KGs")


class DrugSideEffectsResponse(BaseModel):
    """Response for drug side effects query."""
    drug: str = Field(..., description="Query drug name")
    side_effects: List[SideEffect] = Field(default_factory=list, description="Side effects")
    total_count: int = Field(0, description="Total side effects")


class GenePathwaysResponse(BaseModel):
    """Response for gene pathways query."""
    gene: str = Field(..., description="Query gene name")
    pathways: List[Pathway] = Field(default_factory=list, description="Pathways")
    total_count: int = Field(0, description="Total pathways")


class SearchResponse(BaseModel):
    """Response for entity search."""
    query: str = Field(..., description="Search query")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_count: int = Field(0, description="Total results")


class EntitySummaryResponse(BaseModel):
    """Comprehensive entity summary from all KGs."""
    name: str = Field(..., description="Entity name")
    type: Optional[str] = Field(None, description="Entity type")
    found_in: List[str] = Field(default_factory=list, description="KGs where entity was found")
    primekg_id: Optional[str] = Field(None, description="PrimeKG identifier")
    hetionet_id: Optional[str] = Field(None, description="Hetionet identifier")
    targets: Optional[List[Dict[str, Any]]] = Field(None, description="Drug targets (if drug)")
    indications: Optional[List[Dict[str, Any]]] = Field(None, description="Disease indications (if drug)")
    side_effects: Optional[List[Dict[str, Any]]] = Field(None, description="Side effects (if drug)")
    diseases: Optional[List[Dict[str, Any]]] = Field(None, description="Associated diseases (if gene)")
    interactions: Optional[List[Dict[str, Any]]] = Field(None, description="Protein interactions (if gene)")
    pathways: Optional[List[Dict[str, Any]]] = Field(None, description="Pathways (if gene)")
    genes: Optional[List[Dict[str, Any]]] = Field(None, description="Associated genes (if disease)")


class ConnectionsRequest(BaseModel):
    """Request for finding connections between entities."""
    entity1: str = Field(..., description="First entity name")
    entity2: str = Field(..., description="Second entity name")
    max_depth: int = Field(2, ge=1, le=3, description="Maximum path length")


class ConnectionsResponse(BaseModel):
    """Response for entity connections query."""
    entity1: str = Field(..., description="First entity")
    entity2: str = Field(..., description="Second entity")
    paths: Dict[str, List[Any]] = Field(default_factory=dict, description="Paths by KG source")
    total_paths: int = Field(0, description="Total paths found")
