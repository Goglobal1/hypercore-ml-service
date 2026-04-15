"""
Layer 4h: Hetionet Gene-Disease Mapping

Supplements DisGeNET (Layer 4g) with gene-disease associations from Hetionet.
Uses lazy loading - only loads data when genetic markers are detected.

Key Relationships Used:
- GaD: Gene-associates-Disease
- GdD: Gene-downregulates-Disease
- GuD: Gene-upregulates-Disease
- CtD: Compound-treats-Disease (for drug-gene-disease chains)

Data: 2.25M edges, 47K nodes from Hetionet v1.0
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from functools import lru_cache

logger = logging.getLogger(__name__)

# Lazy import - only load when needed
_hetionet = None


def _get_hetionet():
    """Lazy load Hetionet - only called when genetic markers detected."""
    global _hetionet
    if _hetionet is None:
        try:
            from app.data import get_hetionet
            _hetionet = get_hetionet()
            if _hetionet.available:
                _hetionet._load()  # Force load on first use
                logger.info(f"[HetionetMapper] Loaded {len(_hetionet._edges):,} edges")
        except Exception as e:
            logger.warning(f"[HetionetMapper] Failed to load Hetionet: {e}")
            _hetionet = None
    return _hetionet


@dataclass
class HetionetDiagnosis:
    """Diagnosis from Hetionet gene-disease associations."""
    disease_id: str
    disease_name: str
    confidence: float
    confidence_label: str
    associated_genes: List[Dict[str, Any]]
    gene_count: int
    evidence: List[str]
    source: str = "hetionet_gene_disease"


class HetionetMapper:
    """
    Layer 4h: Gene-Disease mapping via Hetionet knowledge graph.

    Uses lazy loading - Hetionet only loads when genetic markers are present.
    Implements LRU cache for repeated queries.
    """

    # Common gene symbols (same as DisGeNET for compatibility)
    COMMON_GENES = {
        'brca1': 'BRCA1', 'brca2': 'BRCA2', 'tp53': 'TP53', 'p53': 'TP53',
        'egfr': 'EGFR', 'kras': 'KRAS', 'braf': 'BRAF', 'her2': 'ERBB2',
        'apoe': 'APOE', 'cftr': 'CFTR', 'hbb': 'HBB',
        'cyp2d6': 'CYP2D6', 'cyp2c19': 'CYP2C19', 'cyp2c9': 'CYP2C9',
        'ldlr': 'LDLR', 'pcsk9': 'PCSK9', 'scn5a': 'SCN5A',
        'mlh1': 'MLH1', 'msh2': 'MSH2', 'apc': 'APC',
    }

    GENE_FIELD_PATTERNS = [
        'gene', 'genes', 'gene_symbol', 'genetic_marker',
        'variant', 'mutation', 'snp', 'brca', 'egfr', 'kras'
    ]

    def __init__(self):
        """Initialize mapper. Data loads lazily on first query."""
        self._loaded = False

    @property
    def available(self) -> bool:
        """Check if Hetionet data is available (without loading)."""
        try:
            from app.data import get_hetionet
            return get_hetionet().available
        except:
            return False

    def _ensure_loaded(self):
        """Ensure Hetionet is loaded."""
        if not self._loaded:
            hetio = _get_hetionet()
            self._loaded = hetio is not None and hetio._loaded

    @lru_cache(maxsize=100)
    def _cached_gene_diseases(self, gene_symbol: str) -> tuple:
        """
        LRU-cached lookup of diseases for a gene.
        Returns tuple for hashability.
        """
        hetio = _get_hetionet()
        if not hetio or not hetio._loaded:
            return ()

        diseases = hetio.get_gene_diseases(gene_symbol)
        return tuple(
            (d['disease_id'], d['disease_name'], d['relation'])
            for d in diseases
        )

    def extract_genes_from_data(
        self,
        raw_data: Dict[str, Any],
        features: Dict[str, Any] = None
    ) -> List[str]:
        """Extract gene symbols from patient data."""
        genes = set()
        all_data = {**(raw_data or {}), **(features or {})}

        for key, value in all_data.items():
            key_lower = key.lower()

            # Check gene field patterns
            is_gene_field = any(p in key_lower for p in self.GENE_FIELD_PATTERNS)

            if is_gene_field or key_lower in self.COMMON_GENES:
                extracted = self._extract_genes(value)
                genes.update(extracted)

            # Check if key itself is a gene
            if key.upper() in self.COMMON_GENES.values():
                genes.add(key.upper())

        return list(genes)

    def _extract_genes(self, value: Any) -> Set[str]:
        """Extract gene symbols from a value."""
        genes = set()

        if isinstance(value, str):
            value_upper = value.upper().strip()
            if value_upper in self.COMMON_GENES.values():
                genes.add(value_upper)
            elif value.lower() in self.COMMON_GENES:
                genes.add(self.COMMON_GENES[value.lower()])
            # Try splitting
            for delim in [',', ';', '/', ' ']:
                if delim in value:
                    for part in value.split(delim):
                        part = part.strip().upper()
                        if part in self.COMMON_GENES.values():
                            genes.add(part)

        elif isinstance(value, list):
            for item in value:
                genes.update(self._extract_genes(item))

        elif isinstance(value, dict):
            for v in value.values():
                genes.update(self._extract_genes(v))

        return genes

    def find_diseases_by_genes(
        self,
        genes: List[str],
        min_confidence: float = 0.3
    ) -> List[HetionetDiagnosis]:
        """Find diseases associated with genes via Hetionet."""
        self._ensure_loaded()

        if not genes:
            return []

        # Score diseases
        disease_scores: Dict[str, Dict] = {}

        for gene in genes:
            gene_upper = gene.upper()
            if gene.lower() in self.COMMON_GENES:
                gene_upper = self.COMMON_GENES[gene.lower()]

            # Use cached lookup
            disease_tuples = self._cached_gene_diseases(gene_upper)

            for disease_id, disease_name, relation in disease_tuples:
                if disease_id not in disease_scores:
                    disease_scores[disease_id] = {
                        'disease_id': disease_id,
                        'disease_name': disease_name,
                        'genes': [],
                        'relations': set()
                    }

                disease_scores[disease_id]['genes'].append({
                    'gene': gene_upper,
                    'relation': relation
                })
                disease_scores[disease_id]['relations'].add(relation)

        # Convert to diagnoses
        diagnoses = []

        for disease_id, data in disease_scores.items():
            gene_count = len(data['genes'])
            relation_count = len(data['relations'])

            # Confidence based on gene count and relation diversity
            confidence = min(1.0, 0.4 + 0.15 * gene_count + 0.1 * relation_count)

            if confidence < min_confidence:
                continue

            if confidence >= 0.7:
                conf_label = 'high'
            elif confidence >= 0.4:
                conf_label = 'moderate'
            else:
                conf_label = 'low'

            evidence = [
                f"{g['gene']} {g['relation']}"
                for g in data['genes']
            ]

            diagnoses.append(HetionetDiagnosis(
                disease_id=disease_id,
                disease_name=data['disease_name'],
                confidence=round(confidence, 3),
                confidence_label=conf_label,
                associated_genes=data['genes'],
                gene_count=gene_count,
                evidence=evidence
            ))

        diagnoses.sort(key=lambda d: d.confidence, reverse=True)
        return diagnoses[:20]

    def analyze(
        self,
        raw_data: Dict[str, Any],
        features: Dict[str, Any] = None,
        axis_scores: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Main analysis method for Layer 4h.

        Only queries Hetionet if already loaded (to prevent memory issues).
        Use /kg/load endpoint to preload before calling this.
        """
        # Check if Hetionet is already loaded (don't trigger load during request)
        hetio = _get_hetionet()
        if not hetio or not hetio._loaded:
            logger.debug("[HetionetMapper] Hetionet not loaded - skipping (use /kg/load to preload)")
            return []

        # Extract genes first
        genes = self.extract_genes_from_data(raw_data, features)

        if not genes:
            logger.debug("[HetionetMapper] No genetic markers - skipping")
            return []

        logger.info(f"[HetionetMapper] Found genes: {genes}, querying Hetionet...")

        # Now load Hetionet and query
        diagnoses = self.find_diseases_by_genes(genes)
        logger.info(f"[HetionetMapper] Found {len(diagnoses)} disease associations")

        return [asdict(d) for d in diagnoses]

    def get_stats(self) -> Dict[str, Any]:
        """Get mapper statistics."""
        hetio = _get_hetionet()
        return {
            'available': self.available,
            'loaded': self._loaded,
            'hetionet_edges': len(hetio._edges) if hetio and hetio._loaded else 0,
            'hetionet_nodes': len(hetio._nodes) if hetio and hetio._loaded else 0,
            'cache_info': self._cached_gene_diseases.cache_info()._asdict() if self._loaded else {}
        }


# Singleton
_hetionet_mapper: Optional[HetionetMapper] = None


def get_hetionet_mapper() -> HetionetMapper:
    """Get singleton Hetionet mapper."""
    global _hetionet_mapper
    if _hetionet_mapper is None:
        _hetionet_mapper = HetionetMapper()
    return _hetionet_mapper
