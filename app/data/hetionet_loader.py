"""
Hetionet Knowledge Graph Loader

Hetionet is a network of biology that unifies biomedical knowledge.
It contains 47K nodes of 11 types and 2.25M edges of 24 types.

Data source: https://github.com/hetio/hetionet
Publication: https://doi.org/10.7554/eLife.26726
"""

import os
import gzip
import json
import bz2
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Path configuration
_BASE_DIR = Path(__file__).parent.parent.parent
HETIONET_PATH = Path(os.environ.get('HETIONET_PATH', _BASE_DIR / 'data' / 'external' / 'hetionet'))
HETIONET_JSON = HETIONET_PATH / 'hetnet' / 'json' / 'hetionet-v1.0.json.bz2'
HETIONET_NODES = HETIONET_PATH / 'hetnet' / 'tsv' / 'hetionet-v1.0-nodes.tsv'
HETIONET_EDGES = HETIONET_PATH / 'hetnet' / 'tsv' / 'hetionet-v1.0-edges.sif.gz'
HETIONET_METAGRAPH = HETIONET_PATH / 'hetnet' / 'json' / 'hetionet-v1.0-metagraph.json'


@dataclass
class HetioNode:
    """Represents a node in Hetionet."""
    id: str
    name: str
    kind: str


@dataclass
class HetioEdge:
    """Represents an edge in Hetionet."""
    source_id: str
    metaedge: str
    target_id: str
    source_name: str = ""
    target_name: str = ""
    source_kind: str = ""
    target_kind: str = ""


class HetionetLoader:
    """
    Loader for Hetionet knowledge graph.

    Node types (11):
    - Anatomy, Biological Process, Cellular Component, Compound
    - Disease, Gene, Molecular Function, Pathway
    - Pharmacologic Class, Side Effect, Symptom

    Edge types (24):
    - Compound-treats-Disease, Compound-palliates-Disease
    - Compound-causes-Side Effect, Gene-associates-Disease
    - Gene-participates-Biological Process, etc.
    """

    # Metaedge abbreviations
    METAEDGE_NAMES = {
        'CbG': 'Compound-binds-Gene',
        'CdG': 'Compound-downregulates-Gene',
        'CuG': 'Compound-upregulates-Gene',
        'CtD': 'Compound-treats-Disease',
        'CpD': 'Compound-palliates-Disease',
        'CcSE': 'Compound-causes-Side Effect',
        'GiG': 'Gene-interacts-Gene',
        'GcG': 'Gene-covaries-Gene',
        'Gr>G': 'Gene-regulates-Gene',
        'GaD': 'Gene-associates-Disease',
        'GdD': 'Gene-downregulates-Disease',
        'GuD': 'Gene-upregulates-Disease',
        'DaG': 'Disease-associates-Gene',
        'DdG': 'Disease-downregulates-Gene',
        'DuG': 'Disease-upregulates-Gene',
        'DrD': 'Disease-resembles-Disease',
        'DpS': 'Disease-presents-Symptom',
        'DlA': 'Disease-localizes-Anatomy',
        'AeG': 'Anatomy-expresses-Gene',
        'AuG': 'Anatomy-upregulates-Gene',
        'AdG': 'Anatomy-downregulates-Gene',
        'GpBP': 'Gene-participates-Biological Process',
        'GpCC': 'Gene-participates-Cellular Component',
        'GpMF': 'Gene-participates-Molecular Function',
        'GpPW': 'Gene-participates-Pathway',
        'PCiC': 'Pharmacologic Class-includes-Compound',
    }

    # Node kind abbreviations
    NODE_KINDS = {
        'Anatomy': 'Anatomy',
        'Biological Process': 'BP',
        'Cellular Component': 'CC',
        'Compound': 'Compound',
        'Disease': 'Disease',
        'Gene': 'Gene',
        'Molecular Function': 'MF',
        'Pathway': 'Pathway',
        'Pharmacologic Class': 'PC',
        'Side Effect': 'SE',
        'Symptom': 'Symptom',
    }

    # Essential metaedges for diagnostics (reduces 2.25M -> ~700K edges)
    ESSENTIAL_METAEDGES = {
        # Gene-Disease associations (critical for diagnostics)
        'GaD', 'GdD', 'GuD', 'DaG', 'DdG', 'DuG',
        # Compound-Disease (drug indications)
        'CtD', 'CpD',
        # Compound-Gene (drug targets)
        'CbG', 'CdG', 'CuG',
        # Gene-Gene interactions
        'GiG', 'Gr>G',
        # Disease relationships
        'DrD', 'DpS',
        # Gene-Pathway (mechanism understanding)
        'GpPW',
    }

    def __init__(self, lazy_load: bool = True):
        """
        Initialize Hetionet loader.

        Args:
            lazy_load: If True, load data on first query. If False, load immediately.
        """
        self._nodes: Dict[str, HetioNode] = {}
        self._edges: List[HetioEdge] = []
        self._loaded = False
        self._available = HETIONET_NODES.exists() or HETIONET_JSON.exists()

        # Indexes for fast lookups
        self._by_source: Dict[str, List[HetioEdge]] = defaultdict(list)
        self._by_target: Dict[str, List[HetioEdge]] = defaultdict(list)
        self._by_metaedge: Dict[str, List[HetioEdge]] = defaultdict(list)
        self._by_kind: Dict[str, Dict[str, HetioNode]] = defaultdict(dict)

        # Name-based lookups
        self._genes: Dict[str, HetioNode] = {}
        self._compounds: Dict[str, HetioNode] = {}
        self._diseases: Dict[str, HetioNode] = {}

        if not lazy_load and self._available:
            self._load()

    @property
    def available(self) -> bool:
        """Check if Hetionet data is available."""
        return self._available

    @property
    def loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._loaded

    def _ensure_loaded(self):
        """Ensure data is loaded before querying."""
        if not self._loaded and self._available:
            self._load()

    def _load(self, filter_essential: bool = True):
        """Load Hetionet data.

        Args:
            filter_essential: If True, only load essential metaedges for diagnostics.
                            Reduces 2.25M edges to ~700K edges (69% memory reduction).
        """
        if self._loaded:
            return

        self._filter_essential = filter_essential

        # Try TSV format first (faster)
        if HETIONET_NODES.exists():
            self._load_tsv()
        elif HETIONET_JSON.exists():
            self._load_json()
        else:
            logger.warning("Hetionet data not found")
            return

        self._loaded = True
        logger.info(f"Loaded {len(self._nodes):,} nodes and {len(self._edges):,} edges from Hetionet")

    def _load_tsv(self):
        """Load from TSV format."""
        logger.info(f"Loading Hetionet nodes from {HETIONET_NODES}...")

        # Load nodes
        with open(HETIONET_NODES, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    node_id, name, kind = parts[0], parts[1], parts[2]
                    node = HetioNode(id=node_id, name=name, kind=kind)
                    self._nodes[node_id] = node
                    self._by_kind[kind][node_id] = node

                    # Index by name
                    name_lower = name.lower()
                    if kind == 'Gene':
                        self._genes[name_lower] = node
                        self._genes[node_id] = node
                    elif kind == 'Compound':
                        self._compounds[name_lower] = node
                        self._compounds[node_id] = node
                    elif kind == 'Disease':
                        self._diseases[name_lower] = node
                        self._diseases[node_id] = node

        # Load edges
        if HETIONET_EDGES.exists():
            filter_mode = getattr(self, '_filter_essential', True)
            logger.info(f"Loading Hetionet edges from {HETIONET_EDGES} (filtered={filter_mode})...")
            skipped = 0
            with gzip.open(HETIONET_EDGES, 'rt', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        source_id, metaedge, target_id = parts[0], parts[1], parts[2]

                        # Filter to essential metaedges only
                        if filter_mode and metaedge not in self.ESSENTIAL_METAEDGES:
                            skipped += 1
                            continue

                        source_node = self._nodes.get(source_id)
                        target_node = self._nodes.get(target_id)

                        edge = HetioEdge(
                            source_id=source_id,
                            metaedge=metaedge,
                            target_id=target_id,
                            source_name=source_node.name if source_node else "",
                            target_name=target_node.name if target_node else "",
                            source_kind=source_node.kind if source_node else "",
                            target_kind=target_node.kind if target_node else "",
                        )

                        self._edges.append(edge)
                        self._by_source[source_id].append(edge)
                        self._by_target[target_id].append(edge)
                        self._by_metaedge[metaedge].append(edge)

            if skipped > 0:
                logger.info(f"Skipped {skipped:,} non-essential edges")

    def _load_json(self):
        """Load from JSON format (bz2 compressed)."""
        logger.info(f"Loading Hetionet from {HETIONET_JSON}...")

        with bz2.open(HETIONET_JSON, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        # Load nodes
        for node_data in data.get('nodes', []):
            node = HetioNode(
                id=node_data['identifier'],
                name=node_data['name'],
                kind=node_data['kind']
            )
            self._nodes[node.id] = node
            self._by_kind[node.kind][node.id] = node

            name_lower = node.name.lower()
            if node.kind == 'Gene':
                self._genes[name_lower] = node
                self._genes[node.id] = node
            elif node.kind == 'Compound':
                self._compounds[name_lower] = node
                self._compounds[node.id] = node
            elif node.kind == 'Disease':
                self._diseases[name_lower] = node
                self._diseases[node.id] = node

        # Load edges
        filter_mode = getattr(self, '_filter_essential', True)
        skipped = 0
        for edge_data in data.get('edges', []):
            metaedge = edge_data['kind']

            # Filter to essential metaedges only
            if filter_mode and metaedge not in self.ESSENTIAL_METAEDGES:
                skipped += 1
                continue

            source_id = edge_data['source']
            target_id = edge_data['target']
            source_node = self._nodes.get(source_id)
            target_node = self._nodes.get(target_id)

            edge = HetioEdge(
                source_id=source_id,
                metaedge=metaedge,
                target_id=target_id,
                source_name=source_node.name if source_node else "",
                target_name=target_node.name if target_node else "",
                source_kind=source_node.kind if source_node else "",
                target_kind=target_node.kind if target_node else "",
            )

            self._edges.append(edge)
            self._by_source[source_id].append(edge)
            self._by_target[target_id].append(edge)
            self._by_metaedge[edge.metaedge].append(edge)

        if skipped > 0:
            logger.info(f"Skipped {skipped:,} non-essential edges (JSON)")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded knowledge graph."""
        self._ensure_loaded()

        kind_counts = {k: len(v) for k, v in self._by_kind.items()}
        metaedge_counts = {k: len(v) for k, v in self._by_metaedge.items()}

        return {
            'available': self._available,
            'loaded': self._loaded,
            'total_nodes': len(self._nodes),
            'total_edges': len(self._edges),
            'genes': len(self._genes) // 2,
            'compounds': len(self._compounds) // 2,
            'diseases': len(self._diseases) // 2,
            'node_kinds': kind_counts,
            'metaedge_types': len(metaedge_counts),
            'top_metaedges': dict(sorted(metaedge_counts.items(), key=lambda x: -x[1])[:10]),
        }

    def find_gene(self, gene_name: str) -> Optional[HetioNode]:
        """Find a gene by name or ID."""
        self._ensure_loaded()
        return self._genes.get(gene_name.lower()) or self._genes.get(gene_name)

    def find_compound(self, compound_name: str) -> Optional[HetioNode]:
        """Find a compound/drug by name or ID."""
        self._ensure_loaded()
        return self._compounds.get(compound_name.lower()) or self._compounds.get(compound_name)

    def find_disease(self, disease_name: str) -> Optional[HetioNode]:
        """Find a disease by name or ID."""
        self._ensure_loaded()
        return self._diseases.get(disease_name.lower()) or self._diseases.get(disease_name)

    def get_compound_targets(self, compound_name: str) -> List[Dict[str, Any]]:
        """Get gene targets for a compound."""
        self._ensure_loaded()

        compound = self.find_compound(compound_name)
        if not compound:
            return []

        edges = self._by_source.get(compound.id, [])
        targets = [e for e in edges if e.metaedge in ('CbG', 'CdG', 'CuG')]

        return [
            {
                'gene_id': e.target_id,
                'gene_name': e.target_name,
                'interaction': self.METAEDGE_NAMES.get(e.metaedge, e.metaedge),
            }
            for e in targets
        ]

    def get_compound_diseases(self, compound_name: str) -> List[Dict[str, Any]]:
        """Get diseases treated or palliated by a compound."""
        self._ensure_loaded()

        compound = self.find_compound(compound_name)
        if not compound:
            return []

        edges = self._by_source.get(compound.id, [])
        diseases = [e for e in edges if e.metaedge in ('CtD', 'CpD')]

        return [
            {
                'disease_id': e.target_id,
                'disease_name': e.target_name,
                'relation': 'treats' if e.metaedge == 'CtD' else 'palliates',
            }
            for e in diseases
        ]

    def get_compound_side_effects(self, compound_name: str) -> List[Dict[str, Any]]:
        """Get side effects caused by a compound."""
        self._ensure_loaded()

        compound = self.find_compound(compound_name)
        if not compound:
            return []

        edges = [e for e in self._by_source.get(compound.id, []) if e.metaedge == 'CcSE']

        return [
            {
                'side_effect_id': e.target_id,
                'side_effect_name': e.target_name,
            }
            for e in edges
        ]

    def get_disease_genes(self, disease_name: str) -> List[Dict[str, Any]]:
        """Get genes associated with a disease."""
        self._ensure_loaded()

        disease = self.find_disease(disease_name)
        if not disease:
            # Try partial match
            disease_lower = disease_name.lower()
            for name, node in self._diseases.items():
                if isinstance(name, str) and disease_lower in name:
                    disease = node
                    break

        if not disease:
            return []

        # Disease can be source or target
        edges = self._by_source.get(disease.id, []) + self._by_target.get(disease.id, [])
        gene_edges = [e for e in edges if e.metaedge in ('DaG', 'DdG', 'DuG', 'GaD', 'GdD', 'GuD')]

        results = []
        seen = set()
        for e in gene_edges:
            if e.source_kind == 'Gene':
                gene_id, gene_name = e.source_id, e.source_name
            else:
                gene_id, gene_name = e.target_id, e.target_name

            if gene_id not in seen:
                seen.add(gene_id)
                results.append({
                    'gene_id': gene_id,
                    'gene_name': gene_name,
                    'relation': self.METAEDGE_NAMES.get(e.metaedge, e.metaedge),
                })

        return results

    def get_disease_compounds(self, disease_name: str) -> List[Dict[str, Any]]:
        """Get compounds that treat a disease."""
        self._ensure_loaded()

        disease = self.find_disease(disease_name)
        if not disease:
            return []

        edges = self._by_target.get(disease.id, [])
        compounds = [e for e in edges if e.metaedge in ('CtD', 'CpD')]

        return [
            {
                'compound_id': e.source_id,
                'compound_name': e.source_name,
                'relation': 'treats' if e.metaedge == 'CtD' else 'palliates',
            }
            for e in compounds
        ]

    def get_gene_interactions(self, gene_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get gene-gene interactions."""
        self._ensure_loaded()

        gene = self.find_gene(gene_name)
        if not gene:
            return []

        edges = self._by_source.get(gene.id, []) + self._by_target.get(gene.id, [])
        interactions = [e for e in edges if e.metaedge in ('GiG', 'GcG', 'Gr>G')]

        results = []
        seen = set()
        for e in interactions[:limit * 2]:
            if e.source_id == gene.id:
                partner_id, partner_name = e.target_id, e.target_name
            else:
                partner_id, partner_name = e.source_id, e.source_name

            if partner_id not in seen:
                seen.add(partner_id)
                results.append({
                    'partner_id': partner_id,
                    'partner_name': partner_name,
                    'interaction': self.METAEDGE_NAMES.get(e.metaedge, e.metaedge),
                })

            if len(results) >= limit:
                break

        return results

    def get_gene_pathways(self, gene_name: str) -> List[Dict[str, Any]]:
        """Get pathways a gene participates in."""
        self._ensure_loaded()

        gene = self.find_gene(gene_name)
        if not gene:
            return []

        edges = [e for e in self._by_source.get(gene.id, []) if e.metaedge == 'GpPW']

        return [
            {
                'pathway_id': e.target_id,
                'pathway_name': e.target_name,
            }
            for e in edges
        ]

    def get_gene_diseases(self, gene_name: str) -> List[Dict[str, Any]]:
        """Get diseases associated with a gene."""
        self._ensure_loaded()

        gene = self.find_gene(gene_name)
        if not gene:
            return []

        edges = self._by_source.get(gene.id, []) + self._by_target.get(gene.id, [])
        disease_edges = [e for e in edges if e.metaedge in ('GaD', 'GdD', 'GuD', 'DaG', 'DdG', 'DuG')]

        results = []
        seen = set()
        for e in disease_edges:
            if e.source_kind == 'Disease':
                disease_id, disease_name = e.source_id, e.source_name
            else:
                disease_id, disease_name = e.target_id, e.target_name

            if disease_id not in seen:
                seen.add(disease_id)
                results.append({
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'relation': self.METAEDGE_NAMES.get(e.metaedge, e.metaedge),
                })

        return results

    def search_nodes(self, query: str, kind: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for nodes by name.

        Args:
            query: Search query (case-insensitive partial match)
            kind: Optional filter by node kind
            limit: Maximum results to return

        Returns:
            List of matching nodes
        """
        self._ensure_loaded()

        query_lower = query.lower()
        results = []

        nodes_to_search = self._nodes.values()
        if kind:
            nodes_to_search = self._by_kind.get(kind, {}).values()

        for node in nodes_to_search:
            if query_lower in node.name.lower():
                results.append({
                    'id': node.id,
                    'name': node.name,
                    'kind': node.kind,
                })

                if len(results) >= limit:
                    break

        return results


# Singleton instance
_hetionet: Optional[HetionetLoader] = None


def get_hetionet(lazy_load: bool = True) -> HetionetLoader:
    """Get the singleton Hetionet loader instance."""
    global _hetionet
    if _hetionet is None:
        _hetionet = HetionetLoader(lazy_load=lazy_load)
    return _hetionet
