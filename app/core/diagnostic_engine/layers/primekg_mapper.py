"""
Layer 4i: PrimeKG Disease Mechanism Paths

When abnormal biomarkers are detected, queries PrimeKG for:
  biomarker gene → protein interactions → pathways → diseases

This explains WHY a disease is suspected, not just THAT it's suspected.

Data: 8.1M relationships, 129K nodes from PrimeKG
Uses lazy loading - only loads when abnormal markers need mechanism explanation.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from functools import lru_cache

logger = logging.getLogger(__name__)

# Lazy import
_primekg = None


def _get_primekg():
    """Lazy load PrimeKG - only called when mechanism paths needed."""
    global _primekg
    if _primekg is None:
        try:
            from app.data import get_primekg
            _primekg = get_primekg()
            if _primekg.available:
                _primekg._load()
                logger.info(f"[PrimeKGMapper] Loaded {len(_primekg._edges):,} edges")
        except Exception as e:
            logger.warning(f"[PrimeKGMapper] Failed to load PrimeKG: {e}")
            _primekg = None
    return _primekg


@dataclass
class MechanismPath:
    """A mechanistic path from biomarker to disease."""
    biomarker: str
    gene: str
    pathway: str
    disease: str
    disease_name: str
    path_steps: List[str]
    confidence: float
    evidence: List[str]


@dataclass
class MechanismDiagnosis:
    """Disease diagnosis with mechanistic explanation."""
    disease_id: str
    disease_name: str
    confidence: float
    confidence_label: str
    mechanism_paths: List[Dict[str, Any]]
    biomarkers_explained: List[str]
    pathways_involved: List[str]
    evidence: List[str]
    source: str = "primekg_mechanism"


class PrimeKGMapper:
    """
    Layer 4i: Disease mechanism paths via PrimeKG knowledge graph.

    Explains disease suspicions by tracing:
    abnormal biomarker → gene → protein interactions → pathway → disease

    Uses lazy loading and LRU cache for efficient queries.
    """

    # Biomarker to gene mappings (common lab tests)
    BIOMARKER_GENES = {
        # Liver
        'alt': ['GPT', 'ALT'],
        'ast': ['GOT1', 'GOT2', 'AST'],
        'alp': ['ALPL', 'ALPI'],
        'ggt': ['GGT1'],
        'bilirubin': ['UGT1A1', 'SLCO1B1'],
        'albumin': ['ALB'],
        # Kidney
        'creatinine': ['CKB', 'CKM'],
        'bun': ['ARG1', 'CPS1'],
        'egfr': ['EGFR'],  # Also the gene
        # Cardiac
        'troponin': ['TNNT2', 'TNNI3'],
        'bnp': ['NPPB'],
        'nt_probnp': ['NPPB'],
        'ck_mb': ['CKM'],
        # Lipids
        'ldl': ['LDLR', 'APOB', 'PCSK9'],
        'hdl': ['APOA1', 'CETP', 'ABCA1'],
        'triglycerides': ['LPL', 'APOC3', 'APOA5'],
        'cholesterol': ['HMGCR', 'LDLR', 'NPC1L1'],
        # Glucose/Diabetes
        'glucose': ['GCK', 'SLC2A2', 'INS'],
        'hba1c': ['HBB', 'HBA1'],
        'insulin': ['INS', 'INSR'],
        # Thyroid
        'tsh': ['TSHR', 'TRH'],
        't4': ['DIO1', 'DIO2', 'TPO'],
        't3': ['DIO1', 'DIO2'],
        # Inflammatory
        'crp': ['CRP'],
        'esr': ['FGA', 'FGB', 'FGG'],
        'ferritin': ['FTH1', 'FTL'],
        # Blood
        'hemoglobin': ['HBB', 'HBA1', 'HBA2'],
        'hematocrit': ['EPO', 'EPOR'],
        'wbc': ['CSF3', 'IL6'],
        'platelets': ['THPO', 'MPL'],
        # Coagulation
        'inr': ['F2', 'F7', 'VKORC1'],
        'ptt': ['F8', 'F9', 'F11'],
        'd_dimer': ['FGA', 'PLAT'],
        # Electrolytes (ion channels)
        'sodium': ['SCNN1A', 'SLC9A3'],
        'potassium': ['KCNJ1', 'KCNQ1'],
        'calcium': ['CASR', 'PTH'],
        'magnesium': ['TRPM6', 'TRPM7'],
    }

    def __init__(self):
        """Initialize mapper. Data loads lazily."""
        self._loaded = False

    @property
    def available(self) -> bool:
        """Check if PrimeKG is available."""
        try:
            from app.data import get_primekg
            return get_primekg().available
        except:
            return False

    def _ensure_loaded(self):
        """Ensure PrimeKG is loaded."""
        if not self._loaded:
            pkg = _get_primekg()
            self._loaded = pkg is not None and pkg._loaded

    @lru_cache(maxsize=500)
    def _cached_gene_diseases(self, gene_symbol: str) -> tuple:
        """Cached lookup of diseases for a gene."""
        pkg = _get_primekg()
        if not pkg or not pkg._loaded:
            return ()

        diseases = pkg.get_gene_diseases(gene_symbol)
        return tuple(
            (d['disease_id'], d['disease_name'], d.get('relation', 'associated'))
            for d in diseases[:20]  # Limit per gene
        )

    @lru_cache(maxsize=500)
    def _cached_protein_interactions(self, gene_symbol: str) -> tuple:
        """Cached lookup of protein interactions."""
        pkg = _get_primekg()
        if not pkg or not pkg._loaded:
            return ()

        interactions = pkg.get_protein_interactions(gene_symbol, limit=10)
        return tuple(
            (i['partner_id'], i['partner_name'])
            for i in interactions
        )

    def _normalize_biomarker(self, name: str) -> str:
        """Normalize biomarker name for lookup."""
        name = name.lower().strip()
        # Common normalizations
        name = name.replace('-', '_').replace(' ', '_')
        name = name.replace('(', '').replace(')', '')
        return name

    def extract_abnormal_biomarkers(
        self,
        axis_scores: Dict[str, Any],
        features: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Extract abnormal biomarkers from axis scores."""
        abnormal = []

        for axis_name, axis_data in axis_scores.items():
            if not isinstance(axis_data, dict):
                continue

            # Check for abnormal status
            status = axis_data.get('status', '')
            if status not in ['abnormal', 'critical', 'borderline']:
                continue

            # Get abnormal markers
            for marker_data in axis_data.get('abnormal_markers', []):
                if isinstance(marker_data, dict):
                    marker_name = marker_data.get('marker', '')
                    if marker_name:
                        abnormal.append({
                            'name': marker_name,
                            'value': marker_data.get('value'),
                            'status': marker_data.get('status', status),
                            'axis': axis_name
                        })

        return abnormal

    def find_mechanism_paths(
        self,
        biomarker: str
    ) -> List[MechanismPath]:
        """
        Find mechanism paths from biomarker to diseases.

        Path: biomarker → gene → (protein interactions) → disease
        """
        self._ensure_loaded()

        normalized = self._normalize_biomarker(biomarker)
        genes = self.BIOMARKER_GENES.get(normalized, [])

        if not genes:
            # Try the biomarker name as a gene symbol
            genes = [biomarker.upper()]

        paths = []

        for gene in genes:
            # Direct gene-disease associations
            disease_tuples = self._cached_gene_diseases(gene)

            for disease_id, disease_name, relation in disease_tuples:
                path = MechanismPath(
                    biomarker=biomarker,
                    gene=gene,
                    pathway=f"{gene} → {relation}",
                    disease=disease_id,
                    disease_name=disease_name,
                    path_steps=[
                        f"Abnormal {biomarker}",
                        f"Affects {gene}",
                        f"{relation} {disease_name}"
                    ],
                    confidence=0.6,
                    evidence=[f"{biomarker} → {gene} → {disease_name}"]
                )
                paths.append(path)

            # Through protein interactions (for stronger evidence)
            interactions = self._cached_protein_interactions(gene)
            for partner_id, partner_name in interactions[:5]:
                partner_diseases = self._cached_gene_diseases(partner_name)

                for disease_id, disease_name, relation in partner_diseases[:3]:
                    path = MechanismPath(
                        biomarker=biomarker,
                        gene=gene,
                        pathway=f"{gene} ⟷ {partner_name} → {relation}",
                        disease=disease_id,
                        disease_name=disease_name,
                        path_steps=[
                            f"Abnormal {biomarker}",
                            f"Affects {gene}",
                            f"Interacts with {partner_name}",
                            f"{relation} {disease_name}"
                        ],
                        confidence=0.5,  # Lower for indirect paths
                        evidence=[f"{biomarker} → {gene} ⟷ {partner_name} → {disease_name}"]
                    )
                    paths.append(path)

        return paths

    def analyze(
        self,
        raw_data: Dict[str, Any],
        features: Dict[str, Any] = None,
        axis_scores: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Main analysis for Layer 4i.

        Only loads PrimeKG if abnormal biomarkers need explanation.
        """
        if not axis_scores:
            return []

        # Extract abnormal biomarkers
        abnormal = self.extract_abnormal_biomarkers(axis_scores, features)

        if not abnormal:
            logger.debug("[PrimeKGMapper] No abnormal biomarkers - skipping")
            return []

        logger.info(f"[PrimeKGMapper] Found {len(abnormal)} abnormal biomarkers, loading PrimeKG...")

        # Find mechanism paths for each biomarker
        all_paths: List[MechanismPath] = []
        for marker in abnormal[:10]:  # Limit to top 10 abnormal
            paths = self.find_mechanism_paths(marker['name'])
            all_paths.extend(paths)

        if not all_paths:
            return []

        # Group paths by disease
        disease_groups: Dict[str, Dict] = {}

        for path in all_paths:
            if path.disease not in disease_groups:
                disease_groups[path.disease] = {
                    'disease_id': path.disease,
                    'disease_name': path.disease_name,
                    'paths': [],
                    'biomarkers': set(),
                    'pathways': set(),
                    'total_confidence': 0.0
                }

            disease_groups[path.disease]['paths'].append(asdict(path))
            disease_groups[path.disease]['biomarkers'].add(path.biomarker)
            disease_groups[path.disease]['pathways'].add(path.pathway)
            disease_groups[path.disease]['total_confidence'] += path.confidence

        # Convert to diagnoses
        diagnoses = []

        for disease_id, data in disease_groups.items():
            path_count = len(data['paths'])
            biomarker_count = len(data['biomarkers'])

            # Confidence based on path diversity
            confidence = min(1.0, data['total_confidence'] / path_count * (1 + 0.1 * biomarker_count))

            if confidence >= 0.7:
                conf_label = 'high'
            elif confidence >= 0.4:
                conf_label = 'moderate'
            else:
                conf_label = 'low'

            evidence = [
                f"{p['biomarker']} → {p['gene']} mechanism"
                for p in data['paths'][:5]
            ]

            diagnoses.append(MechanismDiagnosis(
                disease_id=disease_id,
                disease_name=data['disease_name'],
                confidence=round(confidence, 3),
                confidence_label=conf_label,
                mechanism_paths=data['paths'][:5],
                biomarkers_explained=list(data['biomarkers']),
                pathways_involved=list(data['pathways'])[:5],
                evidence=evidence
            ))

        # Sort by confidence and path count
        diagnoses.sort(key=lambda d: (d.confidence, len(d.mechanism_paths)), reverse=True)

        logger.info(f"[PrimeKGMapper] Found {len(diagnoses)} diseases with mechanism paths")

        return [asdict(d) for d in diagnoses[:15]]

    def get_stats(self) -> Dict[str, Any]:
        """Get mapper statistics."""
        pkg = _get_primekg()
        return {
            'available': self.available,
            'loaded': self._loaded,
            'primekg_edges': len(pkg._edges) if pkg and pkg._loaded else 0,
            'primekg_nodes': len(pkg._nodes) if pkg and pkg._loaded else 0,
            'biomarker_mappings': len(self.BIOMARKER_GENES),
            'gene_cache_info': self._cached_gene_diseases.cache_info()._asdict() if self._loaded else {},
            'ppi_cache_info': self._cached_protein_interactions.cache_info()._asdict() if self._loaded else {}
        }


# Singleton
_primekg_mapper: Optional[PrimeKGMapper] = None


def get_primekg_mapper() -> PrimeKGMapper:
    """Get singleton PrimeKG mapper."""
    global _primekg_mapper
    if _primekg_mapper is None:
        _primekg_mapper = PrimeKGMapper()
    return _primekg_mapper
