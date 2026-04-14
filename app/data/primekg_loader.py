"""
PrimeKG Knowledge Graph Loader

PrimeKG (Precision Medicine Knowledge Graph) integrates 20 biomedical resources
to describe 17,080 diseases with 4,050,249 relationships across ten biological scales.

Data source: https://github.com/mims-harvard/PrimeKG
Publication: https://www.nature.com/articles/s41597-023-01960-3
"""

import os
import csv
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Path configuration
_BASE_DIR = Path(__file__).parent.parent.parent
PRIMEKG_PATH = Path(os.environ.get('PRIMEKG_PATH', _BASE_DIR / 'data' / 'external' / 'PrimeKG'))
PRIMEKG_CSV = PRIMEKG_PATH / 'kg.csv'


@dataclass
class KGEdge:
    """Represents an edge in the knowledge graph."""
    relation: str
    display_relation: str
    source_id: str
    source_type: str
    source_name: str
    target_id: str
    target_type: str
    target_name: str
    source_db: str = ""
    target_db: str = ""


@dataclass
class KGNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: str
    name: str
    source: str = ""


class PrimeKGLoader:
    """
    Loader for PrimeKG knowledge graph.

    Provides efficient querying of:
    - Gene/protein relationships
    - Drug-disease associations
    - Disease-gene associations
    - Drug-target interactions
    - Phenotype associations
    """

    # Relation type mappings
    RELATION_TYPES = {
        'protein_protein': 'ppi',
        'drug_protein': 'drug_target',
        'disease_protein': 'disease_gene',
        'drug_disease': 'indication',
        'disease_disease': 'comorbidity',
        'drug_drug': 'drug_interaction',
        'phenotype_protein': 'phenotype_gene',
        'disease_phenotype': 'disease_phenotype',
        'anatomy_protein': 'expression',
        'drug_effect': 'side_effect',
    }

    # Node type mappings
    NODE_TYPES = {
        'gene/protein': 'gene',
        'drug': 'drug',
        'disease': 'disease',
        'effect/phenotype': 'phenotype',
        'anatomy': 'anatomy',
        'biological_process': 'biological_process',
        'molecular_function': 'molecular_function',
        'cellular_component': 'cellular_component',
        'pathway': 'pathway',
    }

    # MINIMAL relationships for Railway deployment (reduces 8.1M -> ~200K edges)
    # Aggressive filtering to fit in constrained memory environment
    ESSENTIAL_RELATIONS = {
        'disease_protein',      # ~160K edges - gene-disease associations (critical)
        'drug_protein',         # Drug-target interactions (critical)
        'drug_disease',         # Drug indications
        # NOTE: Removed protein_protein (642K edges) to reduce memory
        # Can be re-enabled on higher-memory deployments
    }

    def __init__(self, lazy_load: bool = True):
        """
        Initialize PrimeKG loader.

        Args:
            lazy_load: If True, load data on first query. If False, load immediately.
        """
        self._edges: List[KGEdge] = []
        self._nodes: Dict[str, KGNode] = {}
        self._loaded = False
        self._available = PRIMEKG_CSV.exists()

        # Indexes for fast lookups
        self._by_source: Dict[str, List[KGEdge]] = defaultdict(list)
        self._by_target: Dict[str, List[KGEdge]] = defaultdict(list)
        self._by_relation: Dict[str, List[KGEdge]] = defaultdict(list)
        self._by_source_type: Dict[str, List[KGEdge]] = defaultdict(list)
        self._by_target_type: Dict[str, List[KGEdge]] = defaultdict(list)
        self._genes: Dict[str, KGNode] = {}
        self._drugs: Dict[str, KGNode] = {}
        self._diseases: Dict[str, KGNode] = {}

        if not lazy_load and self._available:
            self._load()

    @property
    def available(self) -> bool:
        """Check if PrimeKG data is available."""
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
        """Load PrimeKG data from CSV.

        Args:
            filter_essential: If True, only load essential relationships for diagnostics.
                            Reduces 8.1M edges to ~1M edges (88% memory reduction).
        """
        if self._loaded:
            return

        if not PRIMEKG_CSV.exists():
            logger.warning(f"PrimeKG CSV not found at {PRIMEKG_CSV}")
            return

        logger.info(f"Loading PrimeKG from {PRIMEKG_CSV} (filtered={filter_essential})...")

        skipped = 0
        try:
            with open(PRIMEKG_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    relation = row.get('relation', '')

                    # Filter to essential relationships only
                    if filter_essential and relation not in self.ESSENTIAL_RELATIONS:
                        skipped += 1
                        continue

                    edge = KGEdge(
                        relation=relation,
                        display_relation=row.get('display_relation', ''),
                        source_id=row.get('x_id', ''),
                        source_type=row.get('x_type', ''),
                        source_name=row.get('x_name', ''),
                        target_id=row.get('y_id', ''),
                        target_type=row.get('y_type', ''),
                        target_name=row.get('y_name', ''),
                        source_db=row.get('x_source', ''),
                        target_db=row.get('y_source', ''),
                    )

                    self._edges.append(edge)

                    # Build indexes
                    source_key = f"{edge.source_type}:{edge.source_id}"
                    target_key = f"{edge.target_type}:{edge.target_id}"

                    self._by_source[source_key].append(edge)
                    self._by_target[target_key].append(edge)
                    self._by_relation[edge.relation].append(edge)
                    self._by_source_type[edge.source_type].append(edge)
                    self._by_target_type[edge.target_type].append(edge)

                    # Track nodes
                    if source_key not in self._nodes:
                        node = KGNode(
                            id=edge.source_id,
                            type=edge.source_type,
                            name=edge.source_name,
                            source=edge.source_db
                        )
                        self._nodes[source_key] = node
                        self._categorize_node(node)

                    if target_key not in self._nodes:
                        node = KGNode(
                            id=edge.target_id,
                            type=edge.target_type,
                            name=edge.target_name,
                            source=edge.target_db
                        )
                        self._nodes[target_key] = node
                        self._categorize_node(node)

            self._loaded = True
            logger.info(f"Loaded {len(self._edges):,} edges and {len(self._nodes):,} nodes from PrimeKG (skipped {skipped:,} non-essential)")

        except Exception as e:
            logger.error(f"Error loading PrimeKG: {e}")
            raise

    def _categorize_node(self, node: KGNode):
        """Categorize node by type for fast lookup."""
        name_lower = node.name.lower()

        if node.type == 'gene/protein':
            self._genes[name_lower] = node
            self._genes[node.id] = node
        elif node.type == 'drug':
            self._drugs[name_lower] = node
            self._drugs[node.id] = node
        elif node.type == 'disease':
            self._diseases[name_lower] = node
            self._diseases[node.id] = node

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded knowledge graph."""
        self._ensure_loaded()

        relation_counts = {k: len(v) for k, v in self._by_relation.items()}
        source_type_counts = {k: len(v) for k, v in self._by_source_type.items()}

        return {
            'available': self._available,
            'loaded': self._loaded,
            'total_edges': len(self._edges),
            'total_nodes': len(self._nodes),
            'genes': len(self._genes) // 2,  # Divide by 2 since we store by name and ID
            'drugs': len(self._drugs) // 2,
            'diseases': len(self._diseases) // 2,
            'relation_types': len(relation_counts),
            'top_relations': dict(sorted(relation_counts.items(), key=lambda x: -x[1])[:10]),
            'node_types': source_type_counts,
        }

    def find_gene(self, gene_name: str) -> Optional[KGNode]:
        """Find a gene by name or ID."""
        self._ensure_loaded()
        return self._genes.get(gene_name.lower()) or self._genes.get(gene_name)

    def find_drug(self, drug_name: str) -> Optional[KGNode]:
        """Find a drug by name or ID."""
        self._ensure_loaded()
        return self._drugs.get(drug_name.lower()) or self._drugs.get(drug_name)

    def find_disease(self, disease_name: str) -> Optional[KGNode]:
        """Find a disease by name or ID."""
        self._ensure_loaded()
        return self._diseases.get(disease_name.lower()) or self._diseases.get(disease_name)

    def get_gene_relationships(self, gene_name: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all relationships for a gene.

        Args:
            gene_name: Gene name or ID
            relation_type: Optional filter by relation type

        Returns:
            List of relationship dictionaries
        """
        self._ensure_loaded()

        gene = self.find_gene(gene_name)
        if not gene:
            return []

        key = f"{gene.type}:{gene.id}"
        edges = self._by_source.get(key, []) + self._by_target.get(key, [])

        if relation_type:
            edges = [e for e in edges if e.relation == relation_type]

        results = []
        for edge in edges:
            is_source = edge.source_id == gene.id
            results.append({
                'relation': edge.relation,
                'display_relation': edge.display_relation,
                'direction': 'outgoing' if is_source else 'incoming',
                'connected_id': edge.target_id if is_source else edge.source_id,
                'connected_type': edge.target_type if is_source else edge.source_type,
                'connected_name': edge.target_name if is_source else edge.source_name,
            })

        return results

    def get_drug_targets(self, drug_name: str) -> List[Dict[str, Any]]:
        """Get protein targets for a drug."""
        self._ensure_loaded()

        drug = self.find_drug(drug_name)
        if not drug:
            return []

        key = f"{drug.type}:{drug.id}"
        edges = [e for e in self._by_source.get(key, []) if e.relation == 'drug_protein']

        return [
            {
                'gene_id': e.target_id,
                'gene_name': e.target_name,
                'relation': e.display_relation,
            }
            for e in edges
        ]

    def get_drug_diseases(self, drug_name: str) -> List[Dict[str, Any]]:
        """Get diseases associated with a drug (indications)."""
        self._ensure_loaded()

        drug = self.find_drug(drug_name)
        if not drug:
            return []

        key = f"{drug.type}:{drug.id}"
        edges = [e for e in self._by_source.get(key, []) if 'disease' in e.target_type.lower()]

        return [
            {
                'disease_id': e.target_id,
                'disease_name': e.target_name,
                'relation': e.display_relation,
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
                if disease_lower in name:
                    disease = node
                    break

        if not disease:
            return []

        key = f"{disease.type}:{disease.id}"
        edges = self._by_source.get(key, []) + self._by_target.get(key, [])
        gene_edges = [e for e in edges if 'gene' in e.source_type.lower() or 'gene' in e.target_type.lower()]

        results = []
        seen = set()
        for e in gene_edges:
            is_source = e.source_id == disease.id
            gene_id = e.target_id if is_source else e.source_id
            gene_name = e.target_name if is_source else e.source_name

            if gene_id not in seen:
                seen.add(gene_id)
                results.append({
                    'gene_id': gene_id,
                    'gene_name': gene_name,
                    'relation': e.display_relation,
                })

        return results

    def get_gene_diseases(self, gene_name: str) -> List[Dict[str, Any]]:
        """Get diseases associated with a gene."""
        self._ensure_loaded()

        gene = self.find_gene(gene_name)
        if not gene:
            return []

        key = f"{gene.type}:{gene.id}"
        edges = self._by_source.get(key, []) + self._by_target.get(key, [])
        disease_edges = [e for e in edges if 'disease' in e.source_type.lower() or 'disease' in e.target_type.lower()]

        results = []
        seen = set()
        for e in disease_edges:
            is_source = e.source_id == gene.id
            disease_id = e.target_id if is_source else e.source_id
            disease_name = e.target_name if is_source else e.source_name

            if disease_id not in seen:
                seen.add(disease_id)
                results.append({
                    'disease_id': disease_id,
                    'disease_name': disease_name,
                    'relation': e.display_relation,
                })

        return results

    def get_protein_interactions(self, gene_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get protein-protein interactions for a gene."""
        self._ensure_loaded()

        gene = self.find_gene(gene_name)
        if not gene:
            return []

        key = f"{gene.type}:{gene.id}"
        edges = [e for e in self._by_source.get(key, []) + self._by_target.get(key, [])
                 if e.relation == 'protein_protein']

        results = []
        seen = set()
        for e in edges[:limit * 2]:  # Get more than limit to account for duplicates
            is_source = e.source_id == gene.id
            partner_id = e.target_id if is_source else e.source_id
            partner_name = e.target_name if is_source else e.source_name

            if partner_id not in seen:
                seen.add(partner_id)
                results.append({
                    'partner_id': partner_id,
                    'partner_name': partner_name,
                    'relation': 'ppi',
                })

            if len(results) >= limit:
                break

        return results

    def search_nodes(self, query: str, node_type: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for nodes by name.

        Args:
            query: Search query (case-insensitive partial match)
            node_type: Optional filter by node type
            limit: Maximum results to return

        Returns:
            List of matching nodes
        """
        self._ensure_loaded()

        query_lower = query.lower()
        results = []

        for key, node in self._nodes.items():
            if node_type and node.type != node_type:
                continue

            if query_lower in node.name.lower():
                results.append({
                    'id': node.id,
                    'type': node.type,
                    'name': node.name,
                    'source': node.source,
                })

                if len(results) >= limit:
                    break

        return results

    def find_path(self, source_name: str, target_name: str, max_depth: int = 2) -> List[List[Dict[str, Any]]]:
        """
        Find paths between two entities in the knowledge graph.

        Args:
            source_name: Source entity name
            target_name: Target entity name
            max_depth: Maximum path length

        Returns:
            List of paths, each path is a list of edge dictionaries
        """
        self._ensure_loaded()

        # Find source and target nodes
        source = (self.find_gene(source_name) or
                  self.find_drug(source_name) or
                  self.find_disease(source_name))
        target = (self.find_gene(target_name) or
                  self.find_drug(target_name) or
                  self.find_disease(target_name))

        if not source or not target:
            return []

        source_key = f"{source.type}:{source.id}"
        target_key = f"{target.type}:{target.id}"

        # BFS for paths
        paths = []
        queue = [(source_key, [])]
        visited = {source_key}

        while queue and len(paths) < 10:
            current_key, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            # Get all edges from current node
            edges = self._by_source.get(current_key, [])

            for edge in edges:
                next_key = f"{edge.target_type}:{edge.target_id}"

                edge_dict = {
                    'source': edge.source_name,
                    'relation': edge.display_relation,
                    'target': edge.target_name,
                }

                if next_key == target_key:
                    paths.append(path + [edge_dict])
                elif next_key not in visited and len(path) < max_depth - 1:
                    visited.add(next_key)
                    queue.append((next_key, path + [edge_dict]))

        return paths


# Singleton instance
_primekg: Optional[PrimeKGLoader] = None


def get_primekg(lazy_load: bool = True) -> PrimeKGLoader:
    """Get the singleton PrimeKG loader instance."""
    global _primekg
    if _primekg is None:
        _primekg = PrimeKGLoader(lazy_load=lazy_load)
    return _primekg
