"""
Layer 4f: HPO Phenotype Mapping
Maps lab abnormalities to Human Phenotype Ontology (HPO) terms,
then links phenotypes to diseases via HPO annotation files.

Strategy 2: Use existing phenotype ontologies (HPO, OMIM, Orphanet)
to map from lab values -> phenotypes -> diseases.

Data Sources (HPO v2026-02-16):
- genes_to_disease.txt: Gene symbol -> OMIM/Orphanet disease mappings
- genes_to_phenotype.txt: Gene symbol -> HPO phenotype mappings
- phenotype_to_genes.txt: HPO phenotype -> Gene -> Disease mappings

How It Works:
1. Abnormal lab values trigger specific HPO phenotype terms
2. HPO terms are linked to genes via phenotype_to_genes.txt
3. Genes are linked to diseases via genes_to_disease.txt
4. Multiple phenotype matches increase disease confidence
5. Results merged as layer_4f_hpo_diagnoses
"""

import os
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# HPO path with environment variable fallback
HPO_PATH = Path(os.environ.get('HPO_PATH', './data/hpo/'))


@dataclass
class HPOTerm:
    """Represents an HPO phenotype term."""
    id: str  # e.g., HP:0001943
    name: str  # e.g., "Hypoglycemia"
    definition: str = ""
    synonyms: List[str] = field(default_factory=list)
    is_a: List[str] = field(default_factory=list)  # Parent terms


@dataclass
class GeneDisease:
    """Gene to disease association."""
    gene_symbol: str
    disease_id: str  # OMIM:123456 or ORPHA:789
    association_type: str  # MENDELIAN, POLYGENIC, etc.
    source: str


@dataclass
class GenePhenotype:
    """Gene to phenotype association."""
    gene_symbol: str
    hpo_id: str
    hpo_name: str
    frequency: str
    disease_id: str


@dataclass
class PhenotypeGene:
    """Phenotype to gene association."""
    hpo_id: str
    hpo_name: str
    gene_symbol: str
    disease_id: str


@dataclass
class HPODiagnosis:
    """Diagnosis derived from HPO phenotype matching."""
    disease_id: str
    disease_name: str
    confidence: float
    confidence_label: str  # high, moderate, low
    matched_phenotypes: List[Dict[str, str]]
    matched_genes: List[str]
    phenotype_count: int
    gene_count: int
    evidence: List[str]
    icd10_code: Optional[str] = None
    source: str = "hpo_phenotype_mapping"


class HPOMapper:
    """
    Layer 4f: Maps abnormalities to HPO phenotypes, then to diseases.

    This layer provides ontology-driven disease detection based on
    standardized phenotype terminology from the Human Phenotype Ontology.
    """

    # Lab abnormality to HPO term mappings
    # Format: (marker, condition) -> (HPO_ID, HPO_Name)
    LAB_TO_HPO = {
        # Glucose abnormalities
        ('glucose', 'high'): ('HP:0003074', 'Hyperglycemia'),
        ('glucose', 'low'): ('HP:0001943', 'Hypoglycemia'),
        ('hba1c', 'high'): ('HP:0003074', 'Hyperglycemia'),

        # Renal markers
        ('creatinine', 'high'): ('HP:0003259', 'Elevated serum creatinine'),
        ('bun', 'high'): ('HP:0003138', 'Increased blood urea nitrogen'),
        ('egfr', 'low'): ('HP:0012622', 'Chronic kidney disease'),
        ('potassium', 'high'): ('HP:0002153', 'Hyperkalemia'),
        ('potassium', 'low'): ('HP:0002900', 'Hypokalemia'),
        ('sodium', 'high'): ('HP:0003228', 'Hypernatremia'),
        ('sodium', 'low'): ('HP:0002902', 'Hyponatremia'),

        # Hepatic markers
        ('alt', 'high'): ('HP:0031964', 'Elevated circulating alanine aminotransferase concentration'),
        ('ast', 'high'): ('HP:0031956', 'Elevated circulating aspartate aminotransferase concentration'),
        ('bilirubin', 'high'): ('HP:0002904', 'Hyperbilirubinemia'),
        ('albumin', 'low'): ('HP:0003073', 'Hypoalbuminemia'),
        ('alkaline_phosphatase', 'high'): ('HP:0003155', 'Elevated circulating alkaline phosphatase concentration'),

        # Hematologic markers
        ('hemoglobin', 'low'): ('HP:0001903', 'Anemia'),
        ('hemoglobin', 'high'): ('HP:0001899', 'Polycythemia'),
        ('wbc', 'high'): ('HP:0001974', 'Leukocytosis'),
        ('wbc', 'low'): ('HP:0001882', 'Leukopenia'),
        ('platelets', 'low'): ('HP:0001873', 'Thrombocytopenia'),
        ('platelets', 'high'): ('HP:0001894', 'Thrombocytosis'),
        ('neutrophils', 'low'): ('HP:0001875', 'Neutropenia'),

        # Inflammatory markers
        ('crp', 'high'): ('HP:0011227', 'Elevated circulating C-reactive protein concentration'),
        ('esr', 'high'): ('HP:0003565', 'Elevated erythrocyte sedimentation rate'),
        ('ferritin', 'high'): ('HP:0003281', 'Increased circulating ferritin concentration'),
        ('procalcitonin', 'high'): ('HP:0011227', 'Elevated circulating C-reactive protein concentration'),

        # Cardiac markers
        ('troponin', 'high'): ('HP:0410173', 'Increased circulating troponin I concentration'),
        ('bnp', 'high'): ('HP:0031546', 'Abnormal brain natriuretic peptide level'),
        ('ck_mb', 'high'): ('HP:0500212', 'Increased serum creatine kinase MB'),

        # Coagulation
        ('inr', 'high'): ('HP:0008151', 'Prolonged prothrombin time'),
        ('pt', 'high'): ('HP:0008151', 'Prolonged prothrombin time'),
        ('ptt', 'high'): ('HP:0003645', 'Prolonged partial thromboplastin time'),
        ('d_dimer', 'high'): ('HP:0033106', 'Elevated D-dimer level'),

        # Endocrine
        ('tsh', 'high'): ('HP:0002925', 'Elevated circulating thyroid-stimulating hormone concentration'),
        ('tsh', 'low'): ('HP:0031099', 'Reduced circulating thyroid-stimulating hormone concentration'),
        ('t4', 'high'): ('HP:0002486', 'Myotonia'),
        ('t4', 'low'): ('HP:0000821', 'Hypothyroidism'),
        ('cortisol', 'high'): ('HP:0003118', 'Increased circulating cortisol level'),
        ('cortisol', 'low'): ('HP:0008163', 'Decreased circulating cortisol level'),

        # Lipids
        ('cholesterol', 'high'): ('HP:0003124', 'Hypercholesterolemia'),
        ('triglycerides', 'high'): ('HP:0002155', 'Hypertriglyceridemia'),
        ('ldl', 'high'): ('HP:0003141', 'Increased LDL cholesterol concentration'),
        ('hdl', 'low'): ('HP:0003233', 'Decreased HDL cholesterol concentration'),

        # Other
        ('lactate', 'high'): ('HP:0002151', 'Increased serum lactate'),
        ('ammonia', 'high'): ('HP:0001987', 'Hyperammonemia'),
        ('uric_acid', 'high'): ('HP:0002149', 'Hyperuricemia'),
        ('calcium', 'high'): ('HP:0003072', 'Hypercalcemia'),
        ('calcium', 'low'): ('HP:0002901', 'Hypocalcemia'),
        ('phosphorus', 'high'): ('HP:0002905', 'Hyperphosphatemia'),
        ('phosphorus', 'low'): ('HP:0002148', 'Hypophosphatemia'),
        ('magnesium', 'low'): ('HP:0002917', 'Hypomagnesemia'),

        # Vitals-derived phenotypes
        ('temperature', 'high'): ('HP:0001945', 'Fever'),
        ('temperature', 'low'): ('HP:0002045', 'Hypothermia'),
        ('heart_rate', 'high'): ('HP:0001649', 'Tachycardia'),
        ('heart_rate', 'low'): ('HP:0001662', 'Bradycardia'),
        ('respiratory_rate', 'high'): ('HP:0002789', 'Tachypnea'),
        ('spo2', 'low'): ('HP:0012418', 'Hypoxemia'),
        ('sbp', 'high'): ('HP:0000822', 'Hypertension'),
        ('sbp', 'low'): ('HP:0002615', 'Hypotension'),
    }

    def __init__(self, hpo_path: Path = None):
        """
        Initialize HPO Mapper.

        Args:
            hpo_path: Path to HPO data directory
        """
        self.hpo_path = hpo_path or HPO_PATH

        # Data structures
        self.hpo_terms: Dict[str, HPOTerm] = {}  # HPO ID -> Term

        # Gene-Disease mappings (from genes_to_disease.txt)
        self.gene_diseases: Dict[str, List[GeneDisease]] = defaultdict(list)  # Gene -> Diseases
        self.disease_genes: Dict[str, List[str]] = defaultdict(list)  # Disease -> Genes

        # Phenotype-Gene mappings (from phenotype_to_genes.txt)
        self.phenotype_genes: Dict[str, List[PhenotypeGene]] = defaultdict(list)  # HPO -> Genes
        self.gene_phenotypes: Dict[str, List[str]] = defaultdict(list)  # Gene -> HPO IDs

        # Disease names
        self.disease_names: Dict[str, str] = {}  # Disease ID -> Name (will be populated)

        # Load status
        self.genes_to_disease_loaded = False
        self.genes_to_phenotype_loaded = False
        self.phenotype_to_genes_loaded = False
        self.available = False

        # Statistics
        self.gene_disease_count = 0
        self.phenotype_gene_count = 0

        # Attempt to load data
        self._load_data()

    def _load_data(self):
        """Load HPO annotation files."""

        # Load genes_to_disease.txt
        genes_to_disease_path = self.hpo_path / "genes_to_disease.txt"
        if genes_to_disease_path.exists():
            try:
                self._load_genes_to_disease(genes_to_disease_path)
                self.genes_to_disease_loaded = True
                logger.info(f"[HPOMapper] Loaded {self.gene_disease_count:,} gene-disease associations")
            except Exception as e:
                logger.warning(f"[HPOMapper] Failed to load genes_to_disease.txt: {e}")
        else:
            logger.info(f"[HPOMapper] genes_to_disease.txt not found at {genes_to_disease_path}")

        # Load phenotype_to_genes.txt
        phenotype_to_genes_path = self.hpo_path / "phenotype_to_genes.txt"
        if phenotype_to_genes_path.exists():
            try:
                self._load_phenotype_to_genes(phenotype_to_genes_path)
                self.phenotype_to_genes_loaded = True
                logger.info(f"[HPOMapper] Loaded {self.phenotype_gene_count:,} phenotype-gene associations")
            except Exception as e:
                logger.warning(f"[HPOMapper] Failed to load phenotype_to_genes.txt: {e}")
        else:
            logger.info(f"[HPOMapper] phenotype_to_genes.txt not found at {phenotype_to_genes_path}")

        # Load genes_to_phenotype.txt (optional, for additional coverage)
        genes_to_phenotype_path = self.hpo_path / "genes_to_phenotype.txt"
        if genes_to_phenotype_path.exists():
            try:
                self._load_genes_to_phenotype(genes_to_phenotype_path)
                self.genes_to_phenotype_loaded = True
                logger.info(f"[HPOMapper] Loaded genes_to_phenotype.txt")
            except Exception as e:
                logger.warning(f"[HPOMapper] Failed to load genes_to_phenotype.txt: {e}")

        self.available = True  # Always available with built-in mappings
        logger.info(f"[HPOMapper] Initialized (genes_to_disease={self.genes_to_disease_loaded}, "
                   f"phenotype_to_genes={self.phenotype_to_genes_loaded})")

    def _load_genes_to_disease(self, path: Path):
        """
        Load genes_to_disease.txt file.
        Format: ncbi_gene_id, gene_symbol, association_type, disease_id, source
        """
        with open(path, 'r', encoding='utf-8') as f:
            # Skip header
            header = next(f, '')

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue

                try:
                    gene_symbol = parts[1].strip().upper()
                    association_type = parts[2].strip()
                    disease_id = parts[3].strip()
                    source = parts[4].strip() if len(parts) > 4 else ''

                    gd = GeneDisease(
                        gene_symbol=gene_symbol,
                        disease_id=disease_id,
                        association_type=association_type,
                        source=source
                    )

                    self.gene_diseases[gene_symbol].append(gd)
                    self.disease_genes[disease_id].append(gene_symbol)
                    self.gene_disease_count += 1

                except Exception:
                    continue

    def _load_phenotype_to_genes(self, path: Path):
        """
        Load phenotype_to_genes.txt file.
        Format: hpo_id, hpo_name, ncbi_gene_id, gene_symbol, disease_id
        """
        with open(path, 'r', encoding='utf-8') as f:
            # Skip header
            header = next(f, '')

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue

                try:
                    hpo_id = parts[0].strip()
                    hpo_name = parts[1].strip()
                    gene_symbol = parts[3].strip().upper()
                    disease_id = parts[4].strip() if len(parts) > 4 else ''

                    if not hpo_id.startswith('HP:'):
                        continue

                    pg = PhenotypeGene(
                        hpo_id=hpo_id,
                        hpo_name=hpo_name,
                        gene_symbol=gene_symbol,
                        disease_id=disease_id
                    )

                    self.phenotype_genes[hpo_id].append(pg)
                    self.gene_phenotypes[gene_symbol].append(hpo_id)

                    # Store HPO term info
                    if hpo_id not in self.hpo_terms:
                        self.hpo_terms[hpo_id] = HPOTerm(id=hpo_id, name=hpo_name)

                    self.phenotype_gene_count += 1

                except Exception:
                    continue

    def _load_genes_to_phenotype(self, path: Path):
        """
        Load genes_to_phenotype.txt file for additional coverage.
        Format: ncbi_gene_id, gene_symbol, hpo_id, hpo_name, frequency, disease_id
        """
        with open(path, 'r', encoding='utf-8') as f:
            # Skip header
            header = next(f, '')

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue

                try:
                    gene_symbol = parts[1].strip().upper()
                    hpo_id = parts[2].strip()
                    hpo_name = parts[3].strip()
                    frequency = parts[4].strip() if len(parts) > 4 else ''
                    disease_id = parts[5].strip() if len(parts) > 5 else ''

                    if not hpo_id.startswith('HP:'):
                        continue

                    # Add to gene_phenotypes if not already present
                    if hpo_id not in self.gene_phenotypes.get(gene_symbol, []):
                        self.gene_phenotypes[gene_symbol].append(hpo_id)

                    # Store HPO term info
                    if hpo_id not in self.hpo_terms:
                        self.hpo_terms[hpo_id] = HPOTerm(id=hpo_id, name=hpo_name)

                except Exception:
                    continue

    def map_abnormalities_to_phenotypes(
        self,
        abnormal_markers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Map abnormal lab values to HPO phenotype terms.

        Args:
            abnormal_markers: List of abnormal markers with status

        Returns:
            List of matched HPO phenotypes
        """
        phenotypes = []

        for marker in abnormal_markers:
            marker_name = marker.get('marker', marker.get('name', '')).lower()
            status = marker.get('status', '')
            value = marker.get('value')

            # Determine direction (high/low)
            direction = None
            if 'high' in status.lower() or 'elevated' in status.lower():
                direction = 'high'
            elif 'low' in status.lower() or 'decreased' in status.lower():
                direction = 'low'
            elif status in ['abnormal', 'borderline']:
                ref = marker.get('reference', '')
                if ref and '-' in str(ref):
                    try:
                        low, high = map(float, str(ref).split('-'))
                        if value and float(value) > high:
                            direction = 'high'
                        elif value and float(value) < low:
                            direction = 'low'
                    except:
                        pass

            # Look up HPO term
            key = (marker_name, direction)
            if key in self.LAB_TO_HPO:
                hpo_id, hpo_name = self.LAB_TO_HPO[key]

                # Get associated genes if available
                associated_genes = [pg.gene_symbol for pg in self.phenotype_genes.get(hpo_id, [])]

                phenotypes.append({
                    'hpo_id': hpo_id,
                    'hpo_name': hpo_name,
                    'source_marker': marker_name,
                    'source_value': value,
                    'source_status': status,
                    'direction': direction,
                    'associated_genes': associated_genes[:10]  # Top 10 genes
                })

        return phenotypes

    def find_diseases_by_phenotypes(
        self,
        phenotypes: List[Dict[str, Any]],
        min_confidence: float = 0.3
    ) -> List[HPODiagnosis]:
        """
        Find diseases that match the detected phenotypes via gene associations.

        Args:
            phenotypes: List of HPO phenotypes detected
            min_confidence: Minimum confidence threshold

        Returns:
            List of HPO diagnoses sorted by confidence
        """
        if not phenotypes:
            return []

        # Collect all genes associated with detected phenotypes
        phenotype_gene_map: Dict[str, Set[str]] = {}  # HPO ID -> Genes

        for p in phenotypes:
            hpo_id = p['hpo_id']
            genes = set()

            # Get genes from phenotype_to_genes
            for pg in self.phenotype_genes.get(hpo_id, []):
                genes.add(pg.gene_symbol)

            if genes:
                phenotype_gene_map[hpo_id] = genes

        if not phenotype_gene_map:
            return []

        # Find diseases associated with these genes
        disease_scores: Dict[str, Dict] = {}

        for hpo_id, genes in phenotype_gene_map.items():
            hpo_name = self.hpo_terms.get(hpo_id, HPOTerm(id=hpo_id, name=hpo_id)).name

            for gene in genes:
                # Get diseases for this gene
                for gd in self.gene_diseases.get(gene, []):
                    disease_id = gd.disease_id

                    if disease_id not in disease_scores:
                        disease_scores[disease_id] = {
                            'disease_id': disease_id,
                            'matched_phenotypes': [],
                            'matched_genes': set(),
                            'total_score': 0.0
                        }

                    # Add phenotype match
                    if hpo_id not in [p['hpo_id'] for p in disease_scores[disease_id]['matched_phenotypes']]:
                        disease_scores[disease_id]['matched_phenotypes'].append({
                            'hpo_id': hpo_id,
                            'hpo_name': hpo_name,
                            'via_gene': gene
                        })

                    disease_scores[disease_id]['matched_genes'].add(gene)
                    disease_scores[disease_id]['total_score'] += 0.5  # Base score per match

        # Convert to diagnoses
        diagnoses = []

        for disease_id, data in disease_scores.items():
            phenotype_count = len(data['matched_phenotypes'])
            gene_count = len(data['matched_genes'])

            # Calculate confidence based on phenotype and gene matches
            raw_confidence = min(1.0, 0.3 + (phenotype_count * 0.15) + (gene_count * 0.1))

            if raw_confidence < min_confidence:
                continue

            # Determine confidence label
            if raw_confidence >= 0.7:
                confidence_label = 'high'
            elif raw_confidence >= 0.5:
                confidence_label = 'moderate'
            else:
                confidence_label = 'low'

            # Build evidence strings
            evidence = []
            for p in data['matched_phenotypes']:
                evidence.append(f"{p['hpo_name']} via {p['via_gene']}")

            # Get disease name from OMIM/Orphanet ID
            disease_name = self._get_disease_name(disease_id)

            diagnoses.append(HPODiagnosis(
                disease_id=disease_id,
                disease_name=disease_name,
                confidence=round(raw_confidence, 3),
                confidence_label=confidence_label,
                matched_phenotypes=data['matched_phenotypes'],
                matched_genes=list(data['matched_genes']),
                phenotype_count=phenotype_count,
                gene_count=gene_count,
                evidence=evidence,
                icd10_code=None,
                source="hpo_phenotype_mapping"
            ))

        # Sort by confidence
        diagnoses.sort(key=lambda d: d.confidence, reverse=True)

        return diagnoses[:20]  # Return top 20

    def _get_disease_name(self, disease_id: str) -> str:
        """Get disease name from OMIM/Orphanet ID."""
        # Check cached names
        if disease_id in self.disease_names:
            return self.disease_names[disease_id]

        # Return formatted ID if no name found
        return disease_id.replace('OMIM:', 'OMIM #').replace('ORPHA:', 'Orphanet ')

    def analyze(
        self,
        axis_scores: Dict[str, Any],
        features: Dict[str, Any] = None,
        raw_data: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Main analysis method for Layer 4f.

        Args:
            axis_scores: Scores from Layer 3 axis scoring
            features: Engineered features from Layer 2
            raw_data: Raw patient data

        Returns:
            List of HPO diagnoses as dicts
        """
        # Collect abnormal markers from axis scores
        abnormal_markers = []

        for axis_name, axis_data in axis_scores.items():
            if not isinstance(axis_data, dict):
                continue

            abnormal = axis_data.get('abnormal_markers', [])
            for marker in abnormal:
                if isinstance(marker, dict):
                    abnormal_markers.append(marker)

        if not abnormal_markers:
            logger.debug("[HPOMapper] No abnormal markers to map")
            return []

        # Map to phenotypes
        phenotypes = self.map_abnormalities_to_phenotypes(abnormal_markers)
        logger.info(f"[HPOMapper] Mapped {len(abnormal_markers)} abnormal markers to {len(phenotypes)} phenotypes")

        if not phenotypes:
            return []

        # Find matching diseases
        diagnoses = self.find_diseases_by_phenotypes(phenotypes)
        logger.info(f"[HPOMapper] Found {len(diagnoses)} disease matches via HPO")

        # Convert to dicts
        return [asdict(d) for d in diagnoses]

    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            'available': self.available,
            'genes_to_disease_loaded': self.genes_to_disease_loaded,
            'phenotype_to_genes_loaded': self.phenotype_to_genes_loaded,
            'genes_to_phenotype_loaded': self.genes_to_phenotype_loaded,
            'gene_disease_associations': self.gene_disease_count,
            'phenotype_gene_associations': self.phenotype_gene_count,
            'unique_genes': len(self.gene_diseases),
            'unique_phenotypes': len(self.phenotype_genes),
            'hpo_terms_count': len(self.hpo_terms),
            'builtin_lab_mappings': len(self.LAB_TO_HPO),
            'hpo_path': str(self.hpo_path)
        }


# Singleton instance
_hpo_mapper: Optional[HPOMapper] = None


def get_hpo_mapper() -> HPOMapper:
    """Get singleton HPO mapper instance."""
    global _hpo_mapper
    if _hpo_mapper is None:
        _hpo_mapper = HPOMapper()
    return _hpo_mapper
