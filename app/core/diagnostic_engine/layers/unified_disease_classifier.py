"""
Layer 4 Unified: Unified Disease Classification

Combines all disease detection sources:
- Rule-based patterns (disease_ontology.json) - 21 core diseases
- ClinVar genetic conditions - 209K+ diseases
- ML models trained on MIMIC-IV - 15 trained models
- ICD-10 code mapping - 97K+ codes

Provides intelligent merging with:
- Deduplication of same disease from multiple sources
- Confidence boosting when multiple sources agree
- Source attribution tracking
- Evidence combination
"""

from typing import Dict, List, Any, Optional, Set
import logging
from collections import defaultdict

from .disease_classifier import DiseaseClassifier

logger = logging.getLogger(__name__)


class UnifiedDiseaseClassifier:
    """
    Unified disease detection combining all available sources.

    Sources integrated:
    1. Rules: Pattern matching against disease_ontology.json
    2. ClinVar: Genetic variant → disease associations
    3. ML: MIMIC-IV trained disease prediction models
    4. ICD-10: Code validation and disease name mapping

    Confidence aggregation:
    - Base confidence from each source
    - Boost when multiple sources agree (+15% per additional source)
    - Maximum confidence capped at 0.98
    """

    # Confidence boost when multiple sources agree
    MULTI_SOURCE_BOOST = 0.15

    # Maximum confidence score
    MAX_CONFIDENCE = 0.98

    # Minimum confidence to include a disease
    MIN_CONFIDENCE = 0.3

    def __init__(
        self,
        disease_ontology: Dict = None,
        reference_ranges: Dict = None,
        clinvar_loader=None,
        ml_model_manager=None,
        icd10_loader=None
    ):
        """
        Initialize unified classifier with all available sources.

        Args:
            disease_ontology: Disease patterns config
            reference_ranges: Lab reference ranges
            clinvar_loader: ClinVar data loader instance
            ml_model_manager: ML model manager instance
            icd10_loader: ICD-10 loader instance
        """
        # Core rule-based classifier
        self.rule_classifier = DiseaseClassifier(disease_ontology, reference_ranges)

        # External data sources (optional)
        self.clinvar = clinvar_loader
        self.ml_models = ml_model_manager
        self.icd10 = icd10_loader

        # Track source availability
        self.sources_available = {
            'rules': True,
            'clinvar': clinvar_loader is not None and getattr(clinvar_loader, '_loaded', False),
            'ml': ml_model_manager is not None and len(getattr(ml_model_manager, 'models', {})) > 0,
            'icd10': icd10_loader is not None and getattr(icd10_loader, '_loaded', False)
        }

        logger.info(f"[UnifiedClassifier] Sources: {self.sources_available}")

    def classify(
        self,
        features: Dict,
        axis_scores: Dict,
        raw_data: Dict = None
    ) -> List[Dict]:
        """
        Run all classifiers and merge results.

        Args:
            features: Engineered features from Layer 2
            axis_scores: Axis scores from Layer 3
            raw_data: Original raw patient data

        Returns:
            Merged, deduplicated list of detected diseases
        """
        raw_data = raw_data or features.get('raw_features', {})

        # Collect all detections from each source
        all_detections = []

        # 1. Rule-based detection (always available)
        rule_diseases = self._classify_rules(features, axis_scores)
        all_detections.extend(rule_diseases)

        # 2. ClinVar genetic conditions
        if self.sources_available['clinvar']:
            clinvar_diseases = self._classify_clinvar(features, raw_data)
            all_detections.extend(clinvar_diseases)

        # 3. ML model predictions
        if self.sources_available['ml']:
            ml_diseases = self._classify_ml(features, raw_data)
            all_detections.extend(ml_diseases)

        # 4. Merge and deduplicate
        merged = self._merge_results(all_detections)

        # 5. Enrich with ICD-10
        if self.sources_available['icd10']:
            merged = self._enrich_icd10(merged)

        # Sort by confidence
        merged.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return merged

    def _classify_rules(self, features: Dict, axis_scores: Dict) -> List[Dict]:
        """Run rule-based classification."""
        diseases = self.rule_classifier.classify(features, axis_scores)

        # Add source attribution
        for d in diseases:
            d['source'] = 'rules'
            d['sources'] = ['rules']

        return diseases

    def _classify_clinvar(self, features: Dict, raw_data: Dict) -> List[Dict]:
        """Check ClinVar for genetic conditions."""
        if not self.clinvar:
            return []

        diseases = []
        genes_affected = self._extract_genes(features, raw_data)

        if not genes_affected:
            return []

        for gene in genes_affected:
            clinvar_diseases = self.clinvar.get_diseases_for_gene(gene)

            for cv_disease in clinvar_diseases:
                disease_name = cv_disease.get('disease', '')
                significance = cv_disease.get('significance', '')

                # Only include pathogenic/likely pathogenic
                if 'Pathogenic' not in significance:
                    continue

                confidence = 0.85 if significance == 'Pathogenic' else 0.70

                diseases.append({
                    'disease_id': f"clinvar_{gene}_{disease_name[:20]}",
                    'disease_name': disease_name,
                    'icd10': None,
                    'category': 'genetic',
                    'detected': True,
                    'confidence': confidence,
                    'confidence_label': self._get_confidence_label(confidence),
                    'severity': None,
                    'stage': None,
                    'evidence': [
                        f"Pathogenic variant in {gene}",
                        f"ClinVar significance: {significance}",
                        f"Review status: {cv_disease.get('review_status', 'unknown')}"
                    ],
                    'missing_data': [],
                    'exclusions_triggered': [],
                    'organ_systems': [],
                    'recommended_followup': [
                        'genetic_counseling',
                        'specialist_referral',
                        'family_screening'
                    ],
                    'source': 'clinvar',
                    'sources': ['clinvar'],
                    'gene': gene,
                    'omim_ids': cv_disease.get('omim_ids', '')
                })

        return diseases

    def _classify_ml(self, features: Dict, raw_data: Dict) -> List[Dict]:
        """Run ML model predictions."""
        if not self.ml_models:
            return []

        diseases = []

        # Extract lab values for ML
        raw_features = features.get('raw_features', raw_data)
        lab_values = {}

        for key, value in raw_features.items():
            if isinstance(value, (int, float)):
                lab_values[key.lower()] = float(value)
            elif isinstance(value, dict) and 'value' in value:
                lab_values[key.lower()] = float(value['value'])

        if not lab_values:
            return []

        # Get predictions
        try:
            predictions = self.ml_models.predict(lab_values)
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return []

        for pred in predictions:
            if pred.get('probability', 0) < self.MIN_CONFIDENCE:
                continue

            confidence = pred['probability']

            diseases.append({
                'disease_id': f"ml_{pred.get('disease_icd', '')}",
                'disease_name': pred.get('disease_name', ''),
                'icd10': pred.get('disease_icd'),
                'category': 'ml_predicted',
                'detected': True,
                'confidence': confidence,
                'confidence_label': self._get_confidence_label(confidence),
                'severity': None,
                'stage': None,
                'evidence': [
                    f"ML model prediction: {confidence*100:.0f}%",
                    f"Model trained on MIMIC-IV ICU data",
                    f"Threshold: {pred.get('threshold', 0.5):.2f}"
                ],
                'missing_data': [],
                'exclusions_triggered': [],
                'organ_systems': [],
                'recommended_followup': [
                    'clinical_correlation',
                    'confirmatory_testing'
                ],
                'source': 'ml',
                'sources': ['ml']
            })

        return diseases

    def _merge_results(self, all_diseases: List[Dict]) -> List[Dict]:
        """
        Merge diseases from multiple sources.

        - Same disease from multiple sources: combine evidence, boost confidence
        - Different diseases: keep separate
        """
        # Group by normalized disease name
        by_name: Dict[str, List[Dict]] = defaultdict(list)

        for disease in all_diseases:
            name = self._normalize_disease_name(disease.get('disease_name', ''))
            if name:
                by_name[name].append(disease)

        merged = []

        for name, entries in by_name.items():
            if len(entries) == 1:
                # Single source - just add it
                merged.append(entries[0])
            else:
                # Multiple sources - merge them
                merged_disease = self._merge_disease_entries(entries)
                merged.append(merged_disease)

        return merged

    def _merge_disease_entries(self, entries: List[Dict]) -> Dict:
        """
        Merge multiple entries for the same disease.

        Strategy:
        - Take base from highest confidence entry
        - Combine evidence from all sources
        - Boost confidence for multi-source agreement
        - Track all sources
        """
        # Sort by confidence to get best entry as base
        entries.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        best = entries[0].copy()

        # Collect all sources
        all_sources = set()
        for e in entries:
            sources = e.get('sources', [e.get('source', 'unknown')])
            all_sources.update(sources)

        best['sources'] = list(all_sources)
        best['source'] = 'unified'

        # Combine evidence (deduplicated)
        all_evidence = []
        seen_evidence = set()

        for e in entries:
            for ev in e.get('evidence', []):
                if ev not in seen_evidence:
                    all_evidence.append(ev)
                    seen_evidence.add(ev)

        best['evidence'] = all_evidence

        # Boost confidence for multi-source agreement
        num_sources = len(all_sources)
        if num_sources > 1:
            boost = (num_sources - 1) * self.MULTI_SOURCE_BOOST
            new_confidence = min(best['confidence'] + boost, self.MAX_CONFIDENCE)

            # Add evidence of multi-source agreement
            best['evidence'].insert(0, f"Detected by {num_sources} sources: {', '.join(all_sources)}")
            best['confidence'] = new_confidence
            best['confidence_label'] = self._get_confidence_label(new_confidence)
            best['multi_source'] = True
            best['source_count'] = num_sources

        # Combine ICD-10 codes (prefer explicit ones)
        icd_codes = set()
        for e in entries:
            if e.get('icd10'):
                icd_codes.add(e['icd10'])

        if icd_codes:
            best['icd10'] = list(icd_codes)[0]  # Primary code
            best['icd10_all'] = list(icd_codes)

        # Combine recommended followup
        all_followup = set()
        for e in entries:
            for f in e.get('recommended_followup', []):
                all_followup.add(f)

        best['recommended_followup'] = list(all_followup)

        return best

    def _enrich_icd10(self, diseases: List[Dict]) -> List[Dict]:
        """Enrich diseases with ICD-10 information."""
        if not self.icd10:
            return diseases

        enriched = []

        for disease in diseases:
            disease_copy = disease.copy()

            existing_icd = disease_copy.get('icd10')

            if existing_icd:
                # Validate existing code
                code_info = self.icd10.get_code(existing_icd)
                if code_info:
                    disease_copy['icd10_validated'] = True
                    disease_copy['icd10_title'] = code_info['title']
                    disease_copy['icd10_category'] = code_info['category_name']
            else:
                # Try to map disease name
                disease_name = disease_copy.get('disease_name', '')
                if disease_name:
                    mapped_code = self.icd10.map_disease_to_icd(disease_name)
                    if mapped_code:
                        code_info = self.icd10.get_code(mapped_code)
                        if code_info:
                            disease_copy['icd10'] = mapped_code
                            disease_copy['icd10_mapped'] = True
                            disease_copy['icd10_title'] = code_info['title']
                            disease_copy['icd10_category'] = code_info['category_name']

            enriched.append(disease_copy)

        return enriched

    def _extract_genes(self, features: Dict, raw_data: Dict) -> List[str]:
        """Extract gene symbols from patient data."""
        genes = []
        raw_features = features.get('raw_features', raw_data)

        # Check explicit gene fields
        if 'genes' in raw_features:
            genes.extend(raw_features['genes'])
        elif 'genetic_variants' in raw_features:
            for variant in raw_features['genetic_variants']:
                if isinstance(variant, dict) and 'gene' in variant:
                    genes.append(variant['gene'])
                elif isinstance(variant, str) and ':' in variant:
                    genes.append(variant.split(':')[0])

        # Check gene columns in raw data
        for key, value in raw_data.items():
            key_lower = key.lower()
            if 'gene' in key_lower and value:
                if isinstance(value, list):
                    genes.extend(value)
                elif isinstance(value, str):
                    genes.append(value)

        return list(set(genes))

    def _normalize_disease_name(self, name: str) -> str:
        """Normalize disease name for comparison."""
        if not name:
            return ''

        # Lowercase
        normalized = name.lower()

        # Remove common suffixes/prefixes
        for remove in [', type 2', ', type 1', ', unspecified', ', nos']:
            normalized = normalized.replace(remove, '')

        # Replace common variations
        replacements = {
            'diabetes mellitus': 'diabetes',
            'mellitus type 2': 'type 2',
            'mellitus type 1': 'type 1',
            'acute kidney injury': 'aki',
            'chronic kidney disease': 'ckd',
            'congestive heart failure': 'heart failure',
            'myocardial infarction': 'mi',
        }

        for old, new in replacements.items():
            if old in normalized:
                normalized = normalized.replace(old, new)

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        return normalized.strip()

    def _get_confidence_label(self, confidence: float) -> str:
        """Get confidence label from score."""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'moderate'
        elif confidence >= 0.4:
            return 'low'
        return 'weak'

    def get_stats(self) -> Dict:
        """Get unified classifier statistics."""
        stats = {
            'sources_available': self.sources_available,
            'rules': {
                'disease_count': len(self.rule_classifier.diseases)
            }
        }

        if self.clinvar and hasattr(self.clinvar, 'get_stats'):
            stats['clinvar'] = self.clinvar.get_stats()

        if self.ml_models and hasattr(self.ml_models, 'list_models'):
            models = self.ml_models.list_models()
            stats['ml'] = {
                'model_count': len(models),
                'models': [m['disease_icd'] for m in models]
            }

        if self.icd10 and hasattr(self.icd10, 'get_stats'):
            stats['icd10'] = self.icd10.get_stats()

        return stats
