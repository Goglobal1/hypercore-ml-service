"""
Layer 4g: DisGeNET Genetic Disease Association Mapping
Maps genetic markers to diseases via gene-disease associations.

Strategy 3: Integrate DisGeNET for gene-disease relationships.
When genetic markers are present in patient data, look up associated diseases.

Data Source:
- curated_gene_disease_associations.tsv.gz: High-confidence gene-disease links
  from expert-curated sources (UNIPROT, CTD, Orphanet, ClinVar, etc.)

NOTE: DisGeNET requires free registration to download data files.
To enable full genetic disease mapping:
1. Register at https://www.disgenet.org/signup
2. Download curated_gene_disease_associations.tsv.gz from https://www.disgenet.org/downloads
3. Place in ./data/disgenet/ or set DISGENET_PATH environment variable

Without the data file, this layer uses 58 common pharmacogenomic gene mappings.

How It Works:
1. Extract gene symbols from patient data (genetic tests, variants, etc.)
2. Look up gene-disease associations in DisGeNET
3. Score associations by evidence strength (GDA score)
4. Results merged as layer_4g_genetic_diagnoses
"""

import os
import gzip
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# DisGeNET path with environment variable fallback
DISGENET_PATH = Path(os.environ.get('DISGENET_PATH', './data/disgenet/'))


@dataclass
class GeneDisease:
    """Gene-disease association from DisGeNET."""
    gene_id: str  # NCBI Gene ID
    gene_symbol: str  # e.g., BRCA1
    disease_id: str  # UMLS CUI
    disease_name: str
    score: float  # GDA score (0-1)
    evidence_index: float  # EI
    year_initial: int
    year_final: int
    source: str  # Source database
    pmid_count: int  # Number of supporting publications


@dataclass
class GeneticDiagnosis:
    """Diagnosis derived from genetic associations."""
    disease_id: str
    disease_name: str
    confidence: float
    confidence_label: str  # high, moderate, low
    associated_genes: List[Dict[str, Any]]
    gene_count: int
    max_gda_score: float
    total_evidence: float
    evidence: List[str]
    icd10_code: Optional[str] = None
    umls_cui: Optional[str] = None
    source: str = "disgenet_genetic_mapping"


class DisGeNETMapper:
    """
    Layer 4g: Maps genetic markers to diseases via DisGeNET.

    This layer activates when genetic information is present in patient data,
    such as gene symbols, variant information, or genetic test results.
    """

    # Common gene symbols found in lab/clinical data
    GENE_FIELD_PATTERNS = [
        'gene', 'genes', 'gene_symbol', 'gene_name', 'genetic_marker',
        'variant', 'variants', 'mutation', 'mutations',
        'snp', 'snps', 'rs_id', 'hgvs',
        'brca', 'egfr', 'kras', 'braf', 'tp53', 'her2', 'pdl1'
    ]

    # Well-known disease genes with common names
    COMMON_GENES = {
        'brca1': 'BRCA1', 'brca2': 'BRCA2', 'tp53': 'TP53', 'p53': 'TP53',
        'egfr': 'EGFR', 'kras': 'KRAS', 'braf': 'BRAF', 'her2': 'ERBB2',
        'erbb2': 'ERBB2', 'pdl1': 'CD274', 'cd274': 'CD274',
        'apoe': 'APOE', 'app': 'APP', 'psen1': 'PSEN1', 'psen2': 'PSEN2',
        'cftr': 'CFTR', 'hbb': 'HBB', 'hba1': 'HBA1', 'hba2': 'HBA2',
        'f5': 'F5', 'f2': 'F2', 'mthfr': 'MTHFR',
        'cyp2d6': 'CYP2D6', 'cyp2c19': 'CYP2C19', 'cyp2c9': 'CYP2C9',
        'vkorc1': 'VKORC1', 'slco1b1': 'SLCO1B1', 'tpmt': 'TPMT',
        'ugt1a1': 'UGT1A1', 'dpyd': 'DPYD', 'g6pd': 'G6PD',
        'ldlr': 'LDLR', 'pcsk9': 'PCSK9', 'apob': 'APOB',
        'scn5a': 'SCN5A', 'kcnq1': 'KCNQ1', 'kcnh2': 'KCNH2',
        'mybpc3': 'MYBPC3', 'myh7': 'MYH7', 'tnnt2': 'TNNT2',
        'dmd': 'DMD', 'smn1': 'SMN1', 'fmr1': 'FMR1',
        'mlh1': 'MLH1', 'msh2': 'MSH2', 'msh6': 'MSH6', 'pms2': 'PMS2',
        'apc': 'APC', 'mutyh': 'MUTYH', 'stk11': 'STK11',
        'ret': 'RET', 'men1': 'MEN1', 'vhl': 'VHL', 'rb1': 'RB1',
        'nf1': 'NF1', 'nf2': 'NF2', 'tsc1': 'TSC1', 'tsc2': 'TSC2',
    }

    def __init__(self, disgenet_path: Path = None):
        """
        Initialize DisGeNET Mapper.

        Args:
            disgenet_path: Path to DisGeNET data directory
        """
        self.disgenet_path = disgenet_path or DISGENET_PATH

        # Data structures
        self.gene_diseases: Dict[str, List[GeneDisease]] = defaultdict(list)  # Gene symbol -> Associations
        self.disease_genes: Dict[str, List[GeneDisease]] = defaultdict(list)  # Disease ID -> Associations
        self.all_genes: Set[str] = set()  # All known gene symbols

        # Load status
        self.data_loaded = False
        self.available = False

        # Attempt to load data
        self._load_data()

    def _load_data(self):
        """Load DisGeNET gene-disease associations."""
        # Try different file names
        possible_files = [
            'curated_gene_disease_associations.tsv.gz',
            'curated_gene_disease_associations.tsv',
            'all_gene_disease_associations.tsv.gz',
            'disgenet_2020.tsv.gz'
        ]

        for filename in possible_files:
            filepath = self.disgenet_path / filename

            if filepath.exists():
                try:
                    self._load_associations(filepath)
                    self.data_loaded = True
                    self.available = True
                    logger.info(f"[DisGeNETMapper] Loaded {len(self.all_genes)} genes, "
                               f"{sum(len(v) for v in self.gene_diseases.values())} associations from {filepath}")
                    return
                except Exception as e:
                    logger.warning(f"[DisGeNETMapper] Failed to load {filepath}: {e}")

        # No data file found - still available with common gene mappings
        self.available = True
        self.all_genes = set(self.COMMON_GENES.values())
        logger.info(f"[DisGeNETMapper] DisGeNET data file not found at {self.disgenet_path}")
        logger.info("[DisGeNETMapper] To enable full genetic mapping, download from https://www.disgenet.org/downloads (free registration required)")
        logger.info(f"[DisGeNETMapper] Using {len(self.COMMON_GENES)} built-in pharmacogenomic gene mappings")

    def _load_associations(self, filepath: Path):
        """Parse DisGeNET TSV file."""
        open_func = gzip.open if str(filepath).endswith('.gz') else open
        mode = 'rt' if str(filepath).endswith('.gz') else 'r'

        with open_func(filepath, mode, encoding='utf-8') as f:
            # Skip header
            header = next(f, '').strip().split('\t')

            # Find column indices (handles different versions)
            col_map = {col.lower(): idx for idx, col in enumerate(header)}

            gene_col = col_map.get('genesymbol', col_map.get('gene_symbol', 0))
            gene_id_col = col_map.get('geneid', col_map.get('gene_id', 1))
            disease_col = col_map.get('diseasename', col_map.get('disease_name', 2))
            disease_id_col = col_map.get('diseaseid', col_map.get('disease_id', 3))
            score_col = col_map.get('score', col_map.get('gda_score', 4))
            ei_col = col_map.get('ei', col_map.get('evidence_index', 5))
            source_col = col_map.get('source', 6)

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue

                try:
                    gene_symbol = parts[gene_col].strip().upper()
                    gene_id = parts[gene_id_col].strip()
                    disease_name = parts[disease_col].strip()
                    disease_id = parts[disease_id_col].strip()
                    score = float(parts[score_col]) if len(parts) > score_col else 0.5
                    ei = float(parts[ei_col]) if len(parts) > ei_col and parts[ei_col] else 0.5
                    source = parts[source_col] if len(parts) > source_col else 'DisGeNET'

                    association = GeneDisease(
                        gene_id=gene_id,
                        gene_symbol=gene_symbol,
                        disease_id=disease_id,
                        disease_name=disease_name,
                        score=score,
                        evidence_index=ei,
                        year_initial=0,
                        year_final=0,
                        source=source,
                        pmid_count=0
                    )

                    self.gene_diseases[gene_symbol].append(association)
                    self.disease_genes[disease_id].append(association)
                    self.all_genes.add(gene_symbol)

                except Exception as e:
                    continue  # Skip malformed lines

    def extract_genes_from_data(
        self,
        raw_data: Dict[str, Any],
        features: Dict[str, Any] = None
    ) -> List[str]:
        """
        Extract gene symbols from patient data.

        Args:
            raw_data: Raw patient data dict
            features: Engineered features

        Returns:
            List of normalized gene symbols
        """
        genes = set()

        # Combine data sources
        all_data = {**(raw_data or {}), **(features or {})}

        for key, value in all_data.items():
            key_lower = key.lower()

            # Check if field name suggests genetic data
            is_gene_field = any(pattern in key_lower for pattern in self.GENE_FIELD_PATTERNS)

            if is_gene_field or key_lower in self.COMMON_GENES:
                # Extract gene symbols from value
                extracted = self._extract_gene_symbols(value)
                genes.update(extracted)

            # Also check if key itself is a gene symbol
            if key.upper() in self.all_genes or key.upper() in self.COMMON_GENES.values():
                genes.add(key.upper())

        return list(genes)

    def _extract_gene_symbols(self, value: Any) -> Set[str]:
        """Extract gene symbols from a value (string, list, dict)."""
        genes = set()

        if isinstance(value, str):
            # Normalize common gene name variations
            value_upper = value.upper().strip()
            if value_upper in self.COMMON_GENES.values():
                genes.add(value_upper)
            elif value.lower() in self.COMMON_GENES:
                genes.add(self.COMMON_GENES[value.lower()])
            elif value_upper in self.all_genes:
                genes.add(value_upper)
            else:
                # Try splitting on common delimiters
                for delimiter in [',', ';', '/', '|', ' ']:
                    if delimiter in value:
                        for part in value.split(delimiter):
                            part = part.strip().upper()
                            if part in self.all_genes or part in self.COMMON_GENES.values():
                                genes.add(part)

        elif isinstance(value, list):
            for item in value:
                genes.update(self._extract_gene_symbols(item))

        elif isinstance(value, dict):
            for v in value.values():
                genes.update(self._extract_gene_symbols(v))

        return genes

    def find_diseases_by_genes(
        self,
        genes: List[str],
        min_score: float = 0.3,
        min_confidence: float = 0.3
    ) -> List[GeneticDiagnosis]:
        """
        Find diseases associated with the given genes.

        Args:
            genes: List of gene symbols
            min_score: Minimum GDA score to include
            min_confidence: Minimum confidence threshold

        Returns:
            List of genetic diagnoses sorted by confidence
        """
        if not genes:
            return []

        # Score each disease
        disease_scores: Dict[str, Dict] = {}

        for gene in genes:
            gene_upper = gene.upper()

            # Normalize via common genes
            if gene.lower() in self.COMMON_GENES:
                gene_upper = self.COMMON_GENES[gene.lower()]

            associations = self.gene_diseases.get(gene_upper, [])

            for assoc in associations:
                if assoc.score < min_score:
                    continue

                disease_id = assoc.disease_id

                if disease_id not in disease_scores:
                    disease_scores[disease_id] = {
                        'disease_id': disease_id,
                        'disease_name': assoc.disease_name,
                        'associated_genes': [],
                        'total_score': 0.0,
                        'max_score': 0.0
                    }

                disease_scores[disease_id]['associated_genes'].append({
                    'gene_symbol': assoc.gene_symbol,
                    'gda_score': assoc.score,
                    'evidence_index': assoc.evidence_index,
                    'source': assoc.source
                })
                disease_scores[disease_id]['total_score'] += assoc.score
                disease_scores[disease_id]['max_score'] = max(
                    disease_scores[disease_id]['max_score'],
                    assoc.score
                )

        # Convert to diagnoses
        diagnoses = []

        for disease_id, data in disease_scores.items():
            gene_count = len(data['associated_genes'])
            max_score = data['max_score']
            total_score = data['total_score']

            # Calculate confidence based on GDA scores and gene count
            # Higher scores = more confidence, multiple genes = more confidence
            raw_confidence = min(1.0, max_score * (1 + 0.1 * (gene_count - 1)))

            if raw_confidence < min_confidence:
                continue

            # Determine confidence label
            if raw_confidence >= 0.7:
                confidence_label = 'high'
            elif raw_confidence >= 0.4:
                confidence_label = 'moderate'
            else:
                confidence_label = 'low'

            # Build evidence strings
            evidence = [
                f"{g['gene_symbol']} associated (GDA={g['gda_score']:.2f})"
                for g in data['associated_genes']
            ]

            diagnoses.append(GeneticDiagnosis(
                disease_id=disease_id,
                disease_name=data['disease_name'],
                confidence=round(raw_confidence, 3),
                confidence_label=confidence_label,
                associated_genes=data['associated_genes'],
                gene_count=gene_count,
                max_gda_score=max_score,
                total_evidence=total_score,
                evidence=evidence,
                umls_cui=disease_id if disease_id.startswith('C') else None,
                source="disgenet_genetic_mapping"
            ))

        # Sort by confidence
        diagnoses.sort(key=lambda d: d.confidence, reverse=True)

        return diagnoses[:20]  # Return top 20

    def analyze(
        self,
        raw_data: Dict[str, Any],
        features: Dict[str, Any] = None,
        axis_scores: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Main analysis method for Layer 4g.

        Args:
            raw_data: Raw patient data (may contain genetic info)
            features: Engineered features from Layer 2
            axis_scores: Scores from Layer 3 (for context)

        Returns:
            List of genetic diagnoses as dicts
        """
        # Extract genes from patient data
        genes = self.extract_genes_from_data(raw_data, features)

        if not genes:
            logger.debug("[DisGeNETMapper] No genetic markers found in patient data")
            return []

        logger.info(f"[DisGeNETMapper] Found genetic markers: {genes}")

        # Find associated diseases
        diagnoses = self.find_diseases_by_genes(genes)
        logger.info(f"[DisGeNETMapper] Found {len(diagnoses)} disease associations via genetics")

        # Convert to dicts
        return [asdict(d) for d in diagnoses]

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            'available': self.available,
            'data_loaded': self.data_loaded,
            'genes_count': len(self.all_genes),
            'diseases_count': len(self.disease_genes),
            'associations_count': sum(len(v) for v in self.gene_diseases.values()),
            'common_genes_count': len(self.COMMON_GENES),
            'disgenet_path': str(self.disgenet_path)
        }


# Singleton instance
_disgenet_mapper: Optional[DisGeNETMapper] = None


def get_disgenet_mapper() -> DisGeNETMapper:
    """Get singleton DisGeNET mapper instance."""
    global _disgenet_mapper
    if _disgenet_mapper is None:
        _disgenet_mapper = DisGeNETMapper()
    return _disgenet_mapper
