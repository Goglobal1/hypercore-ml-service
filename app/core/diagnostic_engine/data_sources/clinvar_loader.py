"""
ClinVar Data Loader for Diagnostic Engine
Loads and indexes ClinVar variant_summary.txt.gz for disease detection.

ClinVar contains 1.5+ million genetic variant → disease mappings.
"""

import gzip
import threading
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ClinVar path from environment variable or relative fallback
CLINVAR_PATH = Path(os.environ.get(
    'CLINVAR_PATH',
    os.path.join(os.path.dirname(__file__), '..', 'data', 'clinvar', 'variant_summary.txt.gz')
))


@dataclass
class ClinVarVariant:
    """Parsed ClinVar variant record."""
    allele_id: int
    gene_symbol: str
    gene_id: int
    variant_name: str
    clinical_significance: str
    phenotype_list: List[str]
    phenotype_ids: str  # OMIM IDs
    review_status: str
    chromosome: str
    rs_id: str


class ClinVarLoader:
    """
    Load and index ClinVar for disease detection.

    Provides:
    - Disease → Gene mappings
    - Gene → Disease mappings
    - Phenotype search
    - Pathogenic variant lookup
    """

    _instance = None
    _load_lock = threading.Lock()

    def __init__(self, clinvar_path: Path = CLINVAR_PATH):
        self.clinvar_path = clinvar_path

        # Main indices
        self.disease_gene_map: Dict[str, Set[str]] = defaultdict(set)
        self.gene_disease_map: Dict[str, List[dict]] = defaultdict(list)
        self.phenotype_index: Dict[str, List[dict]] = defaultdict(list)

        # Statistics
        self.total_variants = 0
        self.pathogenic_count = 0
        self.disease_count = 0
        self.gene_count = 0

        self._loaded = False
        self._load_error = None

    def load(self) -> bool:
        """Load ClinVar data into memory indices (thread-safe)."""
        if self._loaded:
            return True

        with ClinVarLoader._load_lock:
            # Double-check after acquiring lock
            if self._loaded:
                return True

            if not self.clinvar_path.exists():
                self._load_error = f"ClinVar file not found: {self.clinvar_path}"
                logger.warning(self._load_error)
                return False

            logger.info(f"[ClinVar] Loading from {self.clinvar_path}...")

            try:
                self._load_variants()
                self._loaded = True

                logger.info(f"[ClinVar] Loaded {self.total_variants:,} variants")
                logger.info(f"[ClinVar] Indexed {self.disease_count:,} diseases, {self.gene_count:,} genes")
                logger.info(f"[ClinVar] Pathogenic variants: {self.pathogenic_count:,}")

                return True

            except Exception as e:
                self._load_error = str(e)
                logger.error(f"[ClinVar] Load error: {e}")
                return False

    def _load_variants(self):
        """Parse variant_summary.txt.gz and build indices."""

        # Column indices in variant_summary.txt
        # Standard columns: AlleleID, Type, Name, GeneID, GeneSymbol, ...
        COL_ALLELE_ID = 0
        COL_TYPE = 1
        COL_NAME = 2
        COL_GENE_ID = 3
        COL_GENE_SYMBOL = 4
        COL_CLINICAL_SIG = 6
        COL_RS_ID = 9
        COL_PHENOTYPE_IDS = 12
        COL_PHENOTYPE_LIST = 13
        COL_REVIEW_STATUS = 24
        COL_CHROMOSOME = 18

        pathogenic_terms = {'Pathogenic', 'Likely pathogenic', 'Pathogenic/Likely pathogenic'}

        with gzip.open(self.clinvar_path, 'rt', encoding='utf-8', errors='replace') as f:
            header = None

            for line_num, line in enumerate(f):
                # Skip header
                if line_num == 0:
                    header = line.strip().split('\t')
                    continue

                parts = line.strip().split('\t')
                if len(parts) < 25:
                    continue

                try:
                    gene_symbol = parts[COL_GENE_SYMBOL].strip()
                    clinical_sig = parts[COL_CLINICAL_SIG].strip()
                    phenotype_list_raw = parts[COL_PHENOTYPE_LIST].strip()
                    phenotype_ids = parts[COL_PHENOTYPE_IDS].strip()

                    # Skip if no gene or phenotype
                    if not gene_symbol or gene_symbol == '-':
                        continue
                    if not phenotype_list_raw or phenotype_list_raw in ('not provided', 'not specified', '-'):
                        continue

                    self.total_variants += 1

                    # Track pathogenic
                    is_pathogenic = any(term in clinical_sig for term in pathogenic_terms)
                    if is_pathogenic:
                        self.pathogenic_count += 1

                    # Parse phenotypes (semicolon-separated)
                    phenotypes = [p.strip() for p in phenotype_list_raw.split(';') if p.strip()]

                    # Build disease → gene map
                    for phenotype in phenotypes:
                        if phenotype.lower() in ('not provided', 'not specified'):
                            continue
                        self.disease_gene_map[phenotype].add(gene_symbol)

                    # Build gene → disease map (only pathogenic/likely pathogenic)
                    if is_pathogenic:
                        for phenotype in phenotypes:
                            if phenotype.lower() in ('not provided', 'not specified'):
                                continue

                            self.gene_disease_map[gene_symbol.upper()].append({
                                'disease': phenotype,
                                'significance': clinical_sig,
                                'review_status': parts[COL_REVIEW_STATUS] if len(parts) > COL_REVIEW_STATUS else '',
                                'omim_ids': phenotype_ids,
                                'variant': parts[COL_NAME] if len(parts) > COL_NAME else '',
                                'rs_id': parts[COL_RS_ID] if len(parts) > COL_RS_ID else ''
                            })

                    # Build phenotype index for search
                    phenotype_lower = phenotypes[0].lower() if phenotypes else ''
                    if phenotype_lower and phenotype_lower not in ('not provided', 'not specified'):
                        self.phenotype_index[phenotype_lower].append({
                            'gene': gene_symbol,
                            'variant': parts[COL_NAME] if len(parts) > COL_NAME else '',
                            'significance': clinical_sig,
                            'chromosome': parts[COL_CHROMOSOME] if len(parts) > COL_CHROMOSOME else '',
                            'omim': phenotype_ids
                        })

                except Exception as e:
                    # Skip malformed lines
                    continue

        # Calculate stats
        self.disease_count = len(self.disease_gene_map)
        self.gene_count = len(self.gene_disease_map)

    def get_diseases_for_gene(self, gene: str, pathogenic_only: bool = True) -> List[dict]:
        """
        Get all diseases associated with a gene.

        Args:
            gene: Gene symbol (e.g., 'BRCA1')
            pathogenic_only: If True, only return pathogenic/likely pathogenic

        Returns:
            List of disease records
        """
        if not self._loaded:
            self.load()

        return self.gene_disease_map.get(gene.upper(), [])

    def get_genes_for_disease(self, disease: str) -> List[str]:
        """
        Get all genes associated with a disease.

        Args:
            disease: Disease/phenotype name

        Returns:
            List of gene symbols
        """
        if not self._loaded:
            self.load()

        return list(self.disease_gene_map.get(disease, set()))

    def search_phenotype(self, query: str, max_results: int = 100) -> List[dict]:
        """
        Search for phenotypes matching query.

        Args:
            query: Search term
            max_results: Maximum results to return

        Returns:
            List of matching phenotype records
        """
        if not self._loaded:
            self.load()

        query_lower = query.lower()
        results = []

        for phenotype, entries in self.phenotype_index.items():
            if query_lower in phenotype:
                results.extend(entries[:10])  # Limit per phenotype
                if len(results) >= max_results:
                    break

        return results[:max_results]

    def get_disease_info(self, disease: str) -> Optional[dict]:
        """
        Get detailed info about a disease.

        Args:
            disease: Disease/phenotype name

        Returns:
            Disease info dict or None
        """
        if not self._loaded:
            self.load()

        entries = self.phenotype_index.get(disease.lower(), [])

        if not entries:
            # Try partial match
            for phenotype in self.phenotype_index:
                if disease.lower() in phenotype:
                    entries = self.phenotype_index[phenotype]
                    break

        if not entries:
            return None

        genes = list(set(e['gene'] for e in entries if e.get('gene')))

        return {
            'disease': disease,
            'genes': genes,
            'gene_count': len(genes),
            'variant_count': len(entries),
            'sample_variants': entries[:5]
        }

    def get_all_diseases(self) -> List[str]:
        """Get list of all disease names."""
        if not self._loaded:
            self.load()
        return list(self.disease_gene_map.keys())

    def get_stats(self) -> dict:
        """Get loader statistics."""
        return {
            'loaded': self._loaded,
            'error': self._load_error,
            'total_variants': self.total_variants,
            'pathogenic_count': self.pathogenic_count,
            'disease_count': self.disease_count,
            'gene_count': self.gene_count,
            'clinvar_path': str(self.clinvar_path)
        }


# Singleton instance
_clinvar_loader: Optional[ClinVarLoader] = None


def get_clinvar_loader() -> ClinVarLoader:
    """Get singleton ClinVar loader instance."""
    global _clinvar_loader
    if _clinvar_loader is None:
        _clinvar_loader = ClinVarLoader()
    return _clinvar_loader
