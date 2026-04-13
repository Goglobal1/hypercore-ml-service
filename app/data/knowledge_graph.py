"""
Unified Knowledge Graph Manager

Provides a single interface to query multiple biomedical knowledge graphs:
- PrimeKG: 8.1M relationships across diseases, genes, drugs
- Hetionet: 2.25M edges connecting compounds, diseases, genes

This enables cross-graph queries and integrated biomedical reasoning.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .primekg_loader import PrimeKGLoader, get_primekg
from .hetionet_loader import HetionetLoader, get_hetionet

logger = logging.getLogger(__name__)


@dataclass
class UnifiedResult:
    """Result from a unified knowledge graph query."""
    entity_id: str
    entity_name: str
    entity_type: str
    source_kg: str
    relation: str
    connected_to: str
    confidence: float = 1.0


class KnowledgeGraphManager:
    """
    Unified manager for multiple knowledge graphs.

    Provides:
    - Combined queries across PrimeKG and Hetionet
    - Cross-validation between knowledge sources
    - Aggregated drug-target and disease-gene associations
    """

    def __init__(self, lazy_load: bool = True):
        """
        Initialize the knowledge graph manager.

        Args:
            lazy_load: If True, load KGs on first query. If False, load immediately.
        """
        self._primekg = get_primekg(lazy_load=lazy_load)
        self._hetionet = get_hetionet(lazy_load=lazy_load)

    @property
    def available_graphs(self) -> Dict[str, bool]:
        """Get availability status of each knowledge graph."""
        return {
            'primekg': self._primekg.available,
            'hetionet': self._hetionet.available,
        }

    @property
    def any_available(self) -> bool:
        """Check if any knowledge graph is available."""
        return self._primekg.available or self._hetionet.available

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all knowledge graphs."""
        stats = {
            'available_graphs': self.available_graphs,
        }

        if self._primekg.available:
            stats['primekg'] = self._primekg.get_stats()

        if self._hetionet.available:
            stats['hetionet'] = self._hetionet.get_stats()

        return stats

    def get_drug_targets(self, drug_name: str, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get protein/gene targets for a drug from all available KGs.

        Args:
            drug_name: Drug name or identifier
            sources: Optional list of KG sources to query ('primekg', 'hetionet')

        Returns:
            Combined list of drug targets with source information
        """
        sources = sources or ['primekg', 'hetionet']
        results = []
        seen_genes = set()

        if 'primekg' in sources and self._primekg.available:
            for target in self._primekg.get_drug_targets(drug_name):
                gene_key = target['gene_name'].lower()
                if gene_key not in seen_genes:
                    seen_genes.add(gene_key)
                    results.append({
                        **target,
                        'source': 'primekg',
                        'validated_in': ['primekg'],
                    })

        if 'hetionet' in sources and self._hetionet.available:
            for target in self._hetionet.get_compound_targets(drug_name):
                gene_key = target['gene_name'].lower()
                if gene_key in seen_genes:
                    # Cross-validate: found in both KGs
                    for r in results:
                        if r['gene_name'].lower() == gene_key:
                            r['validated_in'].append('hetionet')
                            break
                else:
                    seen_genes.add(gene_key)
                    results.append({
                        'gene_id': target['gene_id'],
                        'gene_name': target['gene_name'],
                        'relation': target['interaction'],
                        'source': 'hetionet',
                        'validated_in': ['hetionet'],
                    })

        # Sort by validation count (cross-validated targets first)
        results.sort(key=lambda x: -len(x['validated_in']))

        return results

    def get_disease_genes(self, disease_name: str, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get genes associated with a disease from all available KGs.

        Args:
            disease_name: Disease name or identifier
            sources: Optional list of KG sources to query

        Returns:
            Combined list of disease-gene associations with source information
        """
        sources = sources or ['primekg', 'hetionet']
        results = []
        seen_genes = set()

        if 'primekg' in sources and self._primekg.available:
            for gene in self._primekg.get_disease_genes(disease_name):
                gene_key = gene['gene_name'].lower()
                if gene_key not in seen_genes:
                    seen_genes.add(gene_key)
                    results.append({
                        **gene,
                        'source': 'primekg',
                        'validated_in': ['primekg'],
                    })

        if 'hetionet' in sources and self._hetionet.available:
            for gene in self._hetionet.get_disease_genes(disease_name):
                gene_key = gene['gene_name'].lower()
                if gene_key in seen_genes:
                    for r in results:
                        if r['gene_name'].lower() == gene_key:
                            r['validated_in'].append('hetionet')
                            break
                else:
                    seen_genes.add(gene_key)
                    results.append({
                        'gene_id': gene['gene_id'],
                        'gene_name': gene['gene_name'],
                        'relation': gene['relation'],
                        'source': 'hetionet',
                        'validated_in': ['hetionet'],
                    })

        results.sort(key=lambda x: -len(x['validated_in']))
        return results

    def get_gene_diseases(self, gene_name: str, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get diseases associated with a gene from all available KGs.

        Args:
            gene_name: Gene name or symbol
            sources: Optional list of KG sources to query

        Returns:
            Combined list of gene-disease associations
        """
        sources = sources or ['primekg', 'hetionet']
        results = []
        seen_diseases = set()

        if 'primekg' in sources and self._primekg.available:
            for disease in self._primekg.get_gene_diseases(gene_name):
                disease_key = disease['disease_name'].lower()
                if disease_key not in seen_diseases:
                    seen_diseases.add(disease_key)
                    results.append({
                        **disease,
                        'source': 'primekg',
                        'validated_in': ['primekg'],
                    })

        if 'hetionet' in sources and self._hetionet.available:
            for disease in self._hetionet.get_gene_diseases(gene_name):
                disease_key = disease['disease_name'].lower()
                if disease_key in seen_diseases:
                    for r in results:
                        if r['disease_name'].lower() == disease_key:
                            r['validated_in'].append('hetionet')
                            break
                else:
                    seen_diseases.add(disease_key)
                    results.append({
                        'disease_id': disease['disease_id'],
                        'disease_name': disease['disease_name'],
                        'relation': disease['relation'],
                        'source': 'hetionet',
                        'validated_in': ['hetionet'],
                    })

        results.sort(key=lambda x: -len(x['validated_in']))
        return results

    def get_protein_interactions(self, gene_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get protein-protein interactions from all available KGs.

        Args:
            gene_name: Gene/protein name
            limit: Maximum results per source

        Returns:
            Combined list of protein interactions
        """
        results = []
        seen_partners = set()

        if self._primekg.available:
            for interaction in self._primekg.get_protein_interactions(gene_name, limit=limit):
                partner_key = interaction['partner_name'].lower()
                if partner_key not in seen_partners:
                    seen_partners.add(partner_key)
                    results.append({
                        **interaction,
                        'source': 'primekg',
                        'validated_in': ['primekg'],
                    })

        if self._hetionet.available:
            for interaction in self._hetionet.get_gene_interactions(gene_name, limit=limit):
                partner_key = interaction['partner_name'].lower()
                if partner_key in seen_partners:
                    for r in results:
                        if r['partner_name'].lower() == partner_key:
                            r['validated_in'].append('hetionet')
                            break
                else:
                    seen_partners.add(partner_key)
                    results.append({
                        'partner_id': interaction['partner_id'],
                        'partner_name': interaction['partner_name'],
                        'relation': interaction['interaction'],
                        'source': 'hetionet',
                        'validated_in': ['hetionet'],
                    })

        results.sort(key=lambda x: -len(x['validated_in']))
        return results[:limit]

    def get_drug_diseases(self, drug_name: str) -> List[Dict[str, Any]]:
        """
        Get diseases associated with a drug (indications) from all KGs.

        Args:
            drug_name: Drug name

        Returns:
            Combined list of drug-disease associations
        """
        results = []
        seen_diseases = set()

        if self._primekg.available:
            for disease in self._primekg.get_drug_diseases(drug_name):
                disease_key = disease['disease_name'].lower()
                if disease_key not in seen_diseases:
                    seen_diseases.add(disease_key)
                    results.append({
                        **disease,
                        'source': 'primekg',
                        'validated_in': ['primekg'],
                    })

        if self._hetionet.available:
            for disease in self._hetionet.get_compound_diseases(drug_name):
                disease_key = disease['disease_name'].lower()
                if disease_key in seen_diseases:
                    for r in results:
                        if r['disease_name'].lower() == disease_key:
                            r['validated_in'].append('hetionet')
                            break
                else:
                    seen_diseases.add(disease_key)
                    results.append({
                        'disease_id': disease['disease_id'],
                        'disease_name': disease['disease_name'],
                        'relation': disease['relation'],
                        'source': 'hetionet',
                        'validated_in': ['hetionet'],
                    })

        results.sort(key=lambda x: -len(x['validated_in']))
        return results

    def get_drug_side_effects(self, drug_name: str) -> List[Dict[str, Any]]:
        """
        Get side effects for a drug (from Hetionet).

        Args:
            drug_name: Drug name

        Returns:
            List of side effects
        """
        if not self._hetionet.available:
            return []

        return [
            {**se, 'source': 'hetionet'}
            for se in self._hetionet.get_compound_side_effects(drug_name)
        ]

    def get_gene_pathways(self, gene_name: str) -> List[Dict[str, Any]]:
        """
        Get pathways a gene participates in (from Hetionet).

        Args:
            gene_name: Gene name

        Returns:
            List of pathways
        """
        if not self._hetionet.available:
            return []

        return [
            {**pathway, 'source': 'hetionet'}
            for pathway in self._hetionet.get_gene_pathways(gene_name)
        ]

    def search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search across all knowledge graphs for entities.

        Args:
            query: Search term (partial match)
            entity_type: Optional filter: 'gene', 'drug', 'disease'
            limit: Maximum results

        Returns:
            Combined search results
        """
        results = []
        seen = set()

        # Map entity types to KG-specific types
        primekg_type = None
        hetionet_type = None

        if entity_type:
            type_map = {
                'gene': ('gene/protein', 'Gene'),
                'drug': ('drug', 'Compound'),
                'disease': ('disease', 'Disease'),
            }
            if entity_type in type_map:
                primekg_type, hetionet_type = type_map[entity_type]

        if self._primekg.available:
            for node in self._primekg.search_nodes(query, node_type=primekg_type, limit=limit):
                key = (node['name'].lower(), node['type'])
                if key not in seen:
                    seen.add(key)
                    results.append({
                        **node,
                        'source': 'primekg',
                    })

        if self._hetionet.available:
            for node in self._hetionet.search_nodes(query, kind=hetionet_type, limit=limit):
                key = (node['name'].lower(), node['kind'])
                if key not in seen:
                    seen.add(key)
                    results.append({
                        'id': node['id'],
                        'name': node['name'],
                        'type': node['kind'],
                        'source': 'hetionet',
                    })

        return results[:limit]

    def find_connections(
        self,
        entity1: str,
        entity2: str,
        max_depth: int = 2
    ) -> Dict[str, List[Any]]:
        """
        Find connections between two entities across knowledge graphs.

        Args:
            entity1: First entity name
            entity2: Second entity name
            max_depth: Maximum path length

        Returns:
            Dictionary with paths from each KG source
        """
        paths = {}

        if self._primekg.available:
            primekg_paths = self._primekg.find_path(entity1, entity2, max_depth=max_depth)
            if primekg_paths:
                paths['primekg'] = primekg_paths

        # Note: Hetionet doesn't have find_path implemented
        # Could be added if needed

        return paths

    def get_entity_summary(self, entity_name: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of an entity from all KGs.

        Args:
            entity_name: Entity name (gene, drug, or disease)

        Returns:
            Comprehensive entity information
        """
        summary = {
            'name': entity_name,
            'found_in': [],
            'type': None,
        }

        # Check PrimeKG
        if self._primekg.available:
            gene = self._primekg.find_gene(entity_name)
            drug = self._primekg.find_drug(entity_name)
            disease = self._primekg.find_disease(entity_name)

            if gene:
                summary['found_in'].append('primekg')
                summary['type'] = 'gene'
                summary['primekg_id'] = gene.id
            elif drug:
                summary['found_in'].append('primekg')
                summary['type'] = 'drug'
                summary['primekg_id'] = drug.id
            elif disease:
                summary['found_in'].append('primekg')
                summary['type'] = 'disease'
                summary['primekg_id'] = disease.id

        # Check Hetionet
        if self._hetionet.available:
            gene = self._hetionet.find_gene(entity_name)
            compound = self._hetionet.find_compound(entity_name)
            disease = self._hetionet.find_disease(entity_name)

            if gene:
                if 'hetionet' not in summary['found_in']:
                    summary['found_in'].append('hetionet')
                if not summary['type']:
                    summary['type'] = 'gene'
                summary['hetionet_id'] = gene.id
            elif compound:
                if 'hetionet' not in summary['found_in']:
                    summary['found_in'].append('hetionet')
                if not summary['type']:
                    summary['type'] = 'drug'
                summary['hetionet_id'] = compound.id
            elif disease:
                if 'hetionet' not in summary['found_in']:
                    summary['found_in'].append('hetionet')
                if not summary['type']:
                    summary['type'] = 'disease'
                summary['hetionet_id'] = disease.id

        # Get related information based on type
        if summary['type'] == 'gene':
            summary['diseases'] = self.get_gene_diseases(entity_name)[:10]
            summary['interactions'] = self.get_protein_interactions(entity_name, limit=10)
            summary['pathways'] = self.get_gene_pathways(entity_name)[:10]
        elif summary['type'] == 'drug':
            summary['targets'] = self.get_drug_targets(entity_name)[:10]
            summary['indications'] = self.get_drug_diseases(entity_name)[:10]
            summary['side_effects'] = self.get_drug_side_effects(entity_name)[:10]
        elif summary['type'] == 'disease':
            summary['genes'] = self.get_disease_genes(entity_name)[:10]

        return summary


# Singleton instance
_kg_manager: Optional[KnowledgeGraphManager] = None


def get_kg_manager(lazy_load: bool = True) -> KnowledgeGraphManager:
    """Get the singleton Knowledge Graph Manager instance."""
    global _kg_manager
    if _kg_manager is None:
        _kg_manager = KnowledgeGraphManager(lazy_load=lazy_load)
    return _kg_manager
