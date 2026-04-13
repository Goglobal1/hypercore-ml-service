"""
HyperCore Knowledge Graph Loaders

Provides unified access to biomedical knowledge graphs:
- PrimeKG: 8.1M relationships across diseases, genes, drugs
- Hetionet: 2.25M edges connecting compounds, diseases, genes
"""

from .primekg_loader import PrimeKGLoader, get_primekg
from .hetionet_loader import HetionetLoader, get_hetionet
from .knowledge_graph import KnowledgeGraphManager, get_kg_manager

__all__ = [
    'PrimeKGLoader',
    'get_primekg',
    'HetionetLoader',
    'get_hetionet',
    'KnowledgeGraphManager',
    'get_kg_manager',
]
