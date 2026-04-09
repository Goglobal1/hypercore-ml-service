"""
Master Diagnostic Engine Pipeline
Orchestrates all 9 layers for signals-first disease detection.

Data integrations:
- ClinVar: Genetic disease detection (209K+ diseases)
- ML Models: MIMIC-trained disease classifiers
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os
import logging

from ..layers.input_normalization import InputNormalizer
from ..layers.feature_engineering import FeatureEngineer
from ..layers.axis_scoring import AxisScorer
from ..layers.disease_classifier import DiseaseClassifier
from ..layers.anomaly_detection import AnomalyDetector
from ..layers.convergence_engine import ConvergenceEngine
from ..layers.explainability import ExplainabilityEngine
from ..layers.recommendations import RecommendationEngine
from ..layers.audit import AuditLogger

# ClinVar integration (optional - loads if available)
try:
    from ..data_sources.clinvar_loader import ClinVarLoader, get_clinvar_loader
    CLINVAR_AVAILABLE = True
except ImportError:
    CLINVAR_AVAILABLE = False

# ML Models integration (optional - loads if models exist)
try:
    from ..ml.disease_models import DiseaseModelManager, get_model_manager
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DiagnosticEngine:
    """
    Master pipeline that orchestrates all 9 layers.

    Architecture:
    1. Input Normalization - Standardize raw data
    2. Feature Engineering - Create derived/temporal features
    3. Axis Scoring - Score biologic organ systems
    4. Disease Classification - Pattern match known diseases
    5. Anomaly Detection - Find unknown patterns
    6. Convergence Analysis - Multi-system reasoning
    7. Explainability - Generate explanations
    8. Recommendations - Actionable next steps
    9. Audit - Regulatory trace
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the diagnostic engine with all layers.

        Args:
            config_path: Optional path to config directory. If None, uses default.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

        # Load configurations
        self.ontology = self._load_json(os.path.join(config_path, 'biomarker_ontology.json'))
        self.axes_config = self._load_json(os.path.join(config_path, 'biologic_axes.json'))
        self.reference_ranges = self._load_json(os.path.join(config_path, 'reference_ranges.json'))
        self.disease_ontology = self._load_json(os.path.join(config_path, 'disease_ontology.json'))

        # Initialize layers
        self.normalizer = InputNormalizer(self.ontology)
        self.feature_engineer = FeatureEngineer()
        self.axis_scorer = AxisScorer(self.axes_config, self.reference_ranges)
        self.disease_classifier = DiseaseClassifier(self.disease_ontology, self.reference_ranges)
        self.anomaly_detector = AnomalyDetector()
        self.convergence_engine = ConvergenceEngine()
        self.explainability = ExplainabilityEngine()
        self.recommender = RecommendationEngine()
        self.auditor = AuditLogger()

        # Initialize ClinVar integration (optional)
        self.clinvar = None
        self.clinvar_loaded = False
        if CLINVAR_AVAILABLE:
            try:
                self.clinvar = get_clinvar_loader()
                self.clinvar_loaded = self.clinvar.load()
                if self.clinvar_loaded:
                    stats = self.clinvar.get_stats()
                    logger.info(f"[DiagnosticEngine] ClinVar: {stats['disease_count']:,} diseases, {stats['gene_count']:,} genes")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] ClinVar not loaded: {e}")

        # Initialize ML Models integration (optional)
        self.ml_models = None
        self.ml_models_loaded = False
        if ML_MODELS_AVAILABLE:
            try:
                self.ml_models = get_model_manager()
                model_count = self.ml_models.load_models()
                if model_count > 0:
                    self.ml_models_loaded = True
                    logger.info(f"[DiagnosticEngine] ML Models: {model_count} disease models loaded")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] ML Models not loaded: {e}")

    def _load_json(self, path: str) -> Dict:
        """Load JSON configuration file."""
        with open(path, 'r') as f:
            return json.load(f)

    def analyze(self, raw_patient_data: Dict, history: List[Dict] = None) -> Dict:
        """
        Run full diagnostic pipeline on patient data.

        Args:
            raw_patient_data: Raw patient lab values (any format)
            history: Optional list of previous normalized data points

        Returns:
            Comprehensive diagnostic result
        """

        # Layer 1: Normalize input
        normalized = self.normalizer.normalize(raw_patient_data)

        # Layer 2: Engineer features
        features = self.feature_engineer.engineer_features(normalized, history)

        # Layer 3: Score biologic axes
        axis_scores = self.axis_scorer.score_all_axes(features)

        # Layer 4: Classify known diseases
        diseases = self.disease_classifier.classify(features, axis_scores)

        # Layer 4b: Enhance with ClinVar genetic conditions
        if self.clinvar_loaded:
            diseases = self._enhance_with_clinvar(diseases, features, raw_patient_data)

        # Layer 4c: Enhance with ML model predictions
        if self.ml_models_loaded:
            diseases = self._enhance_with_ml(diseases, features, raw_patient_data)

        # Layer 5: Detect unknown anomalies
        anomalies = self.anomaly_detector.detect_anomalies(features, axis_scores, diseases)

        # Layer 6: Analyze convergence
        convergence = self.convergence_engine.analyze_convergence(axis_scores, diseases, anomalies)

        # Compile intermediate result
        intermediate_result = {
            'axis_scores': axis_scores,
            'diseases': diseases,
            'anomalies': anomalies,
            'convergence': convergence
        }

        # Layer 7: Generate explanations
        explanation = self.explainability.generate_explanation(intermediate_result)

        # Layer 8: Generate recommendations
        recommendations = self.recommender.generate_recommendations(intermediate_result)

        # Layer 9: Create audit record
        audit = self.auditor.create_audit_record(normalized, intermediate_result)

        # Compile final output
        clinical_state = self._determine_state(axis_scores, diseases, convergence)
        risk_score = self._calculate_risk(axis_scores, diseases, convergence)

        return {
            # Identification
            'patient_id': normalized.get('patient_id'),
            'timestamp': datetime.utcnow().isoformat(),
            'success': True,

            # Axis-level results
            'axis_scores': axis_scores,

            # Disease-level results (Layer 4)
            'identified_diseases': diseases,
            'conditions': [
                {
                    'disease': d['disease_name'],
                    'confidence': d['confidence'],
                    'confidence_label': d.get('confidence_label', 'moderate'),
                    'icd10_codes': [d['icd10']] if d.get('icd10') else [],
                    'stage': d.get('stage', ''),
                    'evidence': d.get('evidence', [])
                }
                for d in diseases
            ],

            # Unknown patterns (Layer 5)
            'anomalies': anomalies,
            'unknown_patterns': anomalies,

            # Multi-system (Layer 6)
            'convergence': convergence,

            # Explanations (Layer 7)
            'explanation': explanation,

            # Actions (Layer 8)
            'recommendations': recommendations,
            'immediate_actions': self.recommender.get_immediate_actions(recommendations),

            # Summary
            'summary': self._build_summary(axis_scores, diseases, anomalies, convergence),

            # Clinical state (CSE integration)
            'clinical_state': clinical_state,
            'state_label': self._get_state_label(clinical_state),
            'risk_score': risk_score,

            # Audit (Layer 9)
            'audit': audit,

            # Metadata
            'data_completeness': normalized.get('metadata', {}).get('completeness', 0),
            'engine_version': '2.0.0'
        }

    def analyze_batch(self, patients: List[Dict], history_map: Dict[str, List] = None) -> List[Dict]:
        """
        Analyze multiple patients.

        Args:
            patients: List of raw patient data dicts
            history_map: Optional dict mapping patient_id to history list

        Returns:
            List of diagnostic results
        """
        results = []
        history_map = history_map or {}

        for patient in patients:
            patient_id = patient.get('patient_id', 'unknown')
            history = history_map.get(patient_id)
            result = self.analyze(patient, history)
            results.append(result)

        return results

    def _determine_state(self, axes: Dict, diseases: List, convergence: Dict) -> str:
        """Determine overall clinical state (S0-S3)."""

        # S3: Critical
        if convergence.get('type') == 'critical':
            return 'S3'

        # Check for critical axis scores
        critical_count = sum(1 for a in axes.values() if a.get('status') == 'critical')
        if critical_count >= 2:
            return 'S3'

        # S2: Escalating
        if convergence.get('type') == 'severe':
            return 'S2'
        if critical_count == 1:
            return 'S2'
        if any(d.get('confidence', 0) >= 0.8 for d in diseases):
            return 'S2'

        # S1: Watch
        abnormal_count = sum(1 for a in axes.values() if a.get('status') in ['abnormal', 'borderline'])
        if abnormal_count > 0 or len(diseases) > 0:
            return 'S1'

        # S0: Stable
        return 'S0'

    def _get_state_label(self, state: str) -> str:
        """Get human-readable state label."""
        labels = {
            'S0': 'STABLE',
            'S1': 'WATCH',
            'S2': 'ESCALATING',
            'S3': 'CRITICAL'
        }
        return labels.get(state, 'UNKNOWN')

    def _calculate_risk(self, axes: Dict, diseases: List, convergence: Dict) -> float:
        """Calculate overall risk score 0-100."""

        risk = 0.0

        # From diseases (max 40 points)
        if diseases:
            max_disease_conf = max(d.get('confidence', 0) for d in diseases)
            risk += max_disease_conf * 40

        # From axis disturbances (max 30 points)
        axis_scores = [a.get('score', 0) for a in axes.values()]
        if axis_scores:
            avg_axis = sum(axis_scores) / len(axis_scores)
            risk += avg_axis * 30

        # From convergence (max 30 points)
        conv_score = convergence.get('score', 0)
        risk += conv_score * 30

        return min(risk, 100)

    def _build_summary(self, axes: Dict, diseases: List, anomalies: List, convergence: Dict) -> Dict:
        """Build summary object for easy consumption."""

        # Categorize systems
        critical_systems = [n for n, a in axes.items() if a.get('status') == 'critical']
        warning_systems = [n for n, a in axes.items() if a.get('status') == 'abnormal']
        watch_systems = [n for n, a in axes.items() if a.get('status') == 'borderline']

        # Categorize diseases
        high_conf = [d['disease_name'] for d in diseases if d.get('confidence', 0) >= 0.7]
        possible = [d['disease_name'] for d in diseases if 0.4 <= d.get('confidence', 0) < 0.7]

        # Determine overall risk
        if critical_systems or convergence.get('type') == 'critical':
            overall_risk = 'critical'
        elif warning_systems or convergence.get('type') == 'severe':
            overall_risk = 'high'
        elif watch_systems or diseases:
            overall_risk = 'moderate'
        else:
            overall_risk = 'low'

        return {
            'overall_risk': overall_risk,
            'critical_systems': critical_systems,
            'warning_systems': warning_systems,
            'watch_systems': watch_systems,
            'high_confidence_diseases': high_conf,
            'possible_diseases': possible,
            'total_diseases_matched': len(diseases),
            'unexplained_abnormalities': sum(a.get('marker_count', 0) for a in anomalies),
            'convergence_type': convergence.get('type'),
            'convergence_score': convergence.get('convergence_score', 0)
        }

    def _enhance_with_clinvar(self, diseases: List, features: Dict, raw_data: Dict) -> List:
        """
        Enhance disease detection with ClinVar genetic data.

        If patient has genetic data (genes, variants), check ClinVar
        for associated pathogenic conditions.

        Args:
            diseases: Current detected diseases from rule-based classifier
            features: Engineered features
            raw_data: Raw patient data (may contain genetic info)

        Returns:
            Enhanced disease list with ClinVar conditions
        """
        if not self.clinvar or not self.clinvar_loaded:
            return diseases

        enhanced = diseases.copy()

        # Check for genetic data in patient record
        genes_affected = []

        # Look for gene data in various formats
        raw_features = features.get('raw_features', raw_data)

        # Check for explicit gene fields
        if 'genes' in raw_features:
            genes_affected = raw_features['genes']
        elif 'genetic_variants' in raw_features:
            # Extract genes from variants
            for variant in raw_features['genetic_variants']:
                if isinstance(variant, dict) and 'gene' in variant:
                    genes_affected.append(variant['gene'])
                elif isinstance(variant, str) and ':' in variant:
                    # Format: GENE:variant
                    genes_affected.append(variant.split(':')[0])

        # Check for gene columns in raw data
        for key, value in raw_data.items():
            key_lower = key.lower()
            if 'gene' in key_lower and value:
                if isinstance(value, list):
                    genes_affected.extend(value)
                elif isinstance(value, str):
                    genes_affected.append(value)

        if not genes_affected:
            return diseases

        # Look up ClinVar conditions for each gene
        seen_diseases = {d.get('disease_name', '').lower() for d in diseases}

        for gene in genes_affected:
            clinvar_diseases = self.clinvar.get_diseases_for_gene(gene)

            for cv_disease in clinvar_diseases:
                disease_name = cv_disease.get('disease', '')

                # Skip if already detected
                if disease_name.lower() in seen_diseases:
                    continue

                significance = cv_disease.get('significance', '')

                # Only add pathogenic/likely pathogenic
                if 'Pathogenic' in significance:
                    confidence = 0.85 if significance == 'Pathogenic' else 0.70

                    enhanced.append({
                        'disease_id': f"clinvar_{gene}_{disease_name[:20]}",
                        'disease_name': disease_name,
                        'icd10': None,  # Could map via OMIM
                        'category': 'genetic',
                        'detected': True,
                        'confidence': confidence,
                        'confidence_label': 'high' if confidence >= 0.8 else 'moderate',
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
                        'source': 'ClinVar',
                        'gene': gene,
                        'omim_ids': cv_disease.get('omim_ids', '')
                    })

                    seen_diseases.add(disease_name.lower())

        # Re-sort by confidence
        enhanced.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return enhanced

    def get_clinvar_stats(self) -> Optional[Dict]:
        """Get ClinVar loader statistics."""
        if self.clinvar:
            return self.clinvar.get_stats()
        return None

    def _enhance_with_ml(self, diseases: List, features: Dict, raw_data: Dict) -> List:
        """
        Enhance disease detection with ML model predictions.

        Runs trained MIMIC models against patient lab values to predict
        disease probabilities.

        Args:
            diseases: Current detected diseases
            features: Engineered features
            raw_data: Raw patient data

        Returns:
            Enhanced disease list with ML predictions
        """
        if not self.ml_models or not self.ml_models_loaded:
            return diseases

        enhanced = diseases.copy()

        # Get raw features as dict for ML prediction
        raw_features = features.get('raw_features', raw_data)

        # Convert to simple dict of lab values
        lab_values = {}
        for key, value in raw_features.items():
            if isinstance(value, (int, float)):
                lab_values[key.lower()] = float(value)
            elif isinstance(value, dict) and 'value' in value:
                lab_values[key.lower()] = float(value['value'])

        if not lab_values:
            return diseases

        # Get ML predictions
        try:
            predictions = self.ml_models.predict(lab_values)
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return diseases

        # Already detected diseases
        seen_diseases = {d.get('disease_name', '').lower() for d in diseases}

        # Add ML predictions
        for pred in predictions:
            disease_name = pred.get('disease_name', '')

            # Skip if already detected with higher confidence
            if disease_name.lower() in seen_diseases:
                continue

            # Only add if probability is significant
            if pred.get('probability', 0) >= 0.4:
                confidence = pred['probability']

                enhanced.append({
                    'disease_id': f"ml_{pred.get('disease_icd', '')}",
                    'disease_name': disease_name,
                    'icd10': pred.get('disease_icd'),
                    'category': 'ml_predicted',
                    'detected': True,
                    'confidence': confidence,
                    'confidence_label': 'high' if confidence >= 0.7 else 'moderate' if confidence >= 0.5 else 'low',
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
                    'source': 'ML_MIMIC'
                })

                seen_diseases.add(disease_name.lower())

        # Re-sort by confidence
        enhanced.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return enhanced

    def get_ml_model_stats(self) -> Optional[List[Dict]]:
        """Get ML model statistics."""
        if self.ml_models:
            return self.ml_models.list_models()
        return None


# Convenience function for direct use
def analyze_patient(patient_data: Dict, history: List[Dict] = None) -> Dict:
    """
    Analyze a single patient using the diagnostic engine.

    This is a convenience function that creates an engine instance
    and runs analysis.
    """
    engine = DiagnosticEngine()
    return engine.analyze(patient_data, history)


def analyze_patients(patients: List[Dict]) -> List[Dict]:
    """
    Analyze multiple patients using the diagnostic engine.
    """
    engine = DiagnosticEngine()
    return engine.analyze_batch(patients)
