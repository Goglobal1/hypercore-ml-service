"""
Master Diagnostic Engine Pipeline
Orchestrates all 9 layers for signals-first disease detection.

Data integrations (Phase 5 Unified):
- Rule-based patterns: 21 core disease definitions
- ClinVar: 209K+ genetic disease conditions
- ML Models: 5 MIMIC-trained disease classifiers
- ICD-10: 97K+ diagnosis codes

Utility Gate (Phase 6):
- Handler/Feied framework for clinical decision support
- Surface/suppress/escalate decisions
- Shadow mode for validation
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os
import logging

from ..layers.input_normalization import InputNormalizer
from ..layers.feature_engineering import FeatureEngineer
from ..layers.axis_scoring import AxisScorer
from ..layers.unified_disease_classifier import UnifiedDiseaseClassifier
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

# ICD-10 integration (optional - loads if available)
try:
    from ..data_sources.icd10_loader import ICD10Loader, get_icd10_loader
    ICD10_AVAILABLE = True
except ImportError:
    ICD10_AVAILABLE = False

# ML Models integration (optional - loads if models exist)
try:
    from ..ml.disease_models import DiseaseModelManager, get_model_manager
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

# Utility Gate integration (Phase 6)
try:
    from ...utility_engine import (
        UtilityGate, DeploymentMode, UtilityInput, EvidenceItem
    )
    UTILITY_GATE_AVAILABLE = True
except ImportError:
    UTILITY_GATE_AVAILABLE = False


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

        # Initialize core layers
        self.normalizer = InputNormalizer(self.ontology)
        self.feature_engineer = FeatureEngineer()
        self.axis_scorer = AxisScorer(self.axes_config, self.reference_ranges)
        self.anomaly_detector = AnomalyDetector()
        self.convergence_engine = ConvergenceEngine()
        self.explainability = ExplainabilityEngine()
        self.recommender = RecommendationEngine()
        self.auditor = AuditLogger()

        # Initialize data sources for unified classifier
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

        self.icd10 = None
        self.icd10_loaded = False
        if ICD10_AVAILABLE:
            try:
                self.icd10 = get_icd10_loader()
                self.icd10_loaded = self.icd10.load()
                if self.icd10_loaded:
                    stats = self.icd10.get_stats()
                    logger.info(f"[DiagnosticEngine] ICD-10: {stats['icd10_codes']:,} codes in {stats['categories']} categories")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] ICD-10 not loaded: {e}")

        # Initialize Layer 4: Unified Disease Classifier (Phase 5)
        # Combines: Rules + ClinVar + ML Models + ICD-10
        self.unified_classifier = UnifiedDiseaseClassifier(
            disease_ontology=self.disease_ontology,
            reference_ranges=self.reference_ranges,
            clinvar_loader=self.clinvar if self.clinvar_loaded else None,
            ml_model_manager=self.ml_models if self.ml_models_loaded else None,
            icd10_loader=self.icd10 if self.icd10_loaded else None
        )

        # Log unified classifier status
        classifier_stats = self.unified_classifier.get_stats()
        sources = [k for k, v in classifier_stats['sources_available'].items() if v]
        logger.info(f"[DiagnosticEngine] Unified Classifier: {len(sources)} sources active ({', '.join(sources)})")

        # Initialize Utility Gate (Phase 6 - Shadow Mode)
        self.shadow_mode = True  # Shadow mode: compute but don't suppress
        self.utility_gate = None
        self.utility_gate_available = False

        if UTILITY_GATE_AVAILABLE:
            try:
                self.utility_gate = UtilityGate(mode=DeploymentMode.HOSPITAL)
                self.utility_gate_available = True
                logger.info(f"[DiagnosticEngine] Utility Gate: hospital mode (SHADOW)")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] Utility Gate not initialized: {e}")

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

        # Layer 4: Unified Disease Classification (Phase 5)
        # Combines: Rules + ClinVar + ML Models + ICD-10
        diseases = self.unified_classifier.classify(
            features=features,
            axis_scores=axis_scores,
            raw_data=raw_patient_data
        )

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
            'engine_version': '2.2.0-utility-gate'
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

    def get_clinvar_stats(self) -> Optional[Dict]:
        """Get ClinVar loader statistics."""
        if self.clinvar:
            return self.clinvar.get_stats()
        return None

    def get_ml_model_stats(self) -> Optional[List[Dict]]:
        """Get ML model statistics."""
        if self.ml_models:
            return self.ml_models.list_models()
        return None

    def get_icd10_stats(self) -> Optional[Dict]:
        """Get ICD-10 loader statistics."""
        if self.icd10:
            return self.icd10.get_stats()
        return None

    def search_icd10(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search ICD-10 codes by description."""
        if self.icd10 and self.icd10_loaded:
            return self.icd10.search(query, max_results)
        return []

    def get_icd10_code(self, code: str) -> Optional[Dict]:
        """Get details for a specific ICD-10 code."""
        if self.icd10 and self.icd10_loaded:
            return self.icd10.get_code(code)
        return None

    def get_unified_classifier_stats(self) -> Dict:
        """Get unified classifier statistics including all sources."""
        return self.unified_classifier.get_stats()

    def get_utility_gate_stats(self) -> Optional[Dict]:
        """Get Utility Gate statistics."""
        if self.utility_gate_available:
            return {
                'available': True,
                'mode': self.utility_gate.mode.value,
                'shadow_mode': self.shadow_mode,
                'policy': {
                    'rightness_weight': self.utility_gate.policy.rightness_weight,
                    'novelty_weight': self.utility_gate.policy.novelty_weight,
                    'convincing_weight': self.utility_gate.policy.convincing_weight,
                    'min_handler_score_surface': self.utility_gate.policy.min_handler_score_surface,
                    'min_handler_score_escalate': self.utility_gate.policy.min_handler_score_escalate,
                }
            }
        return {'available': False}

    def _enhance_with_clinvar(self, diseases: List, features: Dict, raw_data: Dict) -> List:
        """
        DEPRECATED: Use unified_classifier.classify() instead.
        Kept for backward compatibility.
        """
        logger.warning("_enhance_with_clinvar is deprecated, use unified_classifier")
        return diseases

    def _enhance_with_ml(self, diseases: List, features: Dict, raw_data: Dict) -> List:
        """
        DEPRECATED: Use unified_classifier.classify() instead.
        Kept for backward compatibility.
        """
        logger.warning("_enhance_with_ml is deprecated, use unified_classifier")
        return diseases

    def _enrich_with_icd10(self, diseases: List) -> List:
        """
        DEPRECATED: Use unified_classifier.classify() instead.
        Kept for backward compatibility.
        """
        logger.warning("_enrich_with_icd10 is deprecated, use unified_classifier")
        return diseases


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
