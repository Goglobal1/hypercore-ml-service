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

# LLM Medical Reasoning integration (Layer 4e)
try:
    from ..layers.llm_medical_reasoner import LLMMedicalReasoner, get_llm_reasoner
    LLM_REASONER_AVAILABLE = True
except ImportError:
    LLM_REASONER_AVAILABLE = False

# HPO Phenotype Mapping integration (Layer 4f)
try:
    from ..layers.hpo_mapper import HPOMapper, get_hpo_mapper
    HPO_MAPPER_AVAILABLE = True
except ImportError:
    HPO_MAPPER_AVAILABLE = False

# DisGeNET Genetic Mapping integration (Layer 4g)
try:
    from ..layers.disgenet_mapper import DisGeNETMapper, get_disgenet_mapper
    DISGENET_MAPPER_AVAILABLE = True
except ImportError:
    DISGENET_MAPPER_AVAILABLE = False

# Hetionet Gene-Disease Mapping (Layer 4h)
try:
    from ..layers.hetionet_mapper import HetionetMapper, get_hetionet_mapper
    HETIONET_MAPPER_AVAILABLE = True
except ImportError:
    HETIONET_MAPPER_AVAILABLE = False

# PrimeKG Disease Mechanism Paths (Layer 4i)
try:
    from ..layers.primekg_mapper import PrimeKGMapper, get_primekg_mapper
    PRIMEKG_MAPPER_AVAILABLE = True
except ImportError:
    PRIMEKG_MAPPER_AVAILABLE = False

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

        # Initialize LLM Medical Reasoner (Layer 4e)
        self.llm_reasoner = None
        self.llm_reasoner_available = False

        if LLM_REASONER_AVAILABLE:
            try:
                self.llm_reasoner = get_llm_reasoner()
                self.llm_reasoner_available = self.llm_reasoner.available
                if self.llm_reasoner_available:
                    logger.info(f"[DiagnosticEngine] LLM Reasoner: {self.llm_reasoner.model}")
                else:
                    logger.info("[DiagnosticEngine] LLM Reasoner: not configured (missing API key)")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] LLM Reasoner not initialized: {e}")

        # Initialize HPO Phenotype Mapper (Layer 4f)
        self.hpo_mapper = None
        self.hpo_mapper_available = False

        if HPO_MAPPER_AVAILABLE:
            try:
                self.hpo_mapper = get_hpo_mapper()
                self.hpo_mapper_available = self.hpo_mapper.available
                stats = self.hpo_mapper.get_stats()
                logger.info(f"[DiagnosticEngine] HPO Mapper: {stats['builtin_mappings']} mappings, "
                           f"obo={stats['obo_loaded']}, hpoa={stats['hpoa_loaded']}")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] HPO Mapper not initialized: {e}")

        # Initialize DisGeNET Genetic Mapper (Layer 4g)
        self.disgenet_mapper = None
        self.disgenet_mapper_available = False

        if DISGENET_MAPPER_AVAILABLE:
            try:
                self.disgenet_mapper = get_disgenet_mapper()
                self.disgenet_mapper_available = self.disgenet_mapper.available
                stats = self.disgenet_mapper.get_stats()
                logger.info(f"[DiagnosticEngine] DisGeNET Mapper: {stats['genes_count']} genes, "
                           f"{stats['associations_count']} associations")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] DisGeNET Mapper not initialized: {e}")

        # Initialize Hetionet Mapper (Layer 4h) - lazy loading
        self.hetionet_mapper = None
        self.hetionet_mapper_available = False

        if HETIONET_MAPPER_AVAILABLE:
            try:
                self.hetionet_mapper = get_hetionet_mapper()
                self.hetionet_mapper_available = self.hetionet_mapper.available
                logger.info(f"[DiagnosticEngine] Hetionet Mapper: available (lazy load)")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] Hetionet Mapper not initialized: {e}")

        # Initialize PrimeKG Mapper (Layer 4i) - lazy loading
        self.primekg_mapper = None
        self.primekg_mapper_available = False

        if PRIMEKG_MAPPER_AVAILABLE:
            try:
                self.primekg_mapper = get_primekg_mapper()
                self.primekg_mapper_available = self.primekg_mapper.available
                logger.info(f"[DiagnosticEngine] PrimeKG Mapper: available (lazy load)")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] PrimeKG Mapper not initialized: {e}")

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

        # Layer 4e: LLM Medical Reasoning
        # Send unexplained abnormalities to Claude for diagnosis
        llm_diagnoses = []
        if self.llm_reasoner_available and LLM_REASONER_AVAILABLE:
            try:
                unexplained = self._get_unexplained_abnormalities(
                    features=features,
                    axis_scores=axis_scores,
                    diseases=diseases
                )
                if unexplained:
                    patient_context = {
                        'age': raw_patient_data.get('age'),
                        'sex': raw_patient_data.get('sex', raw_patient_data.get('gender')),
                        'history': raw_patient_data.get('medical_history', raw_patient_data.get('history')),
                        'medications': raw_patient_data.get('medications')
                    }
                    llm_diagnoses = self.llm_reasoner.reason_about_abnormalities(
                        abnormal_labs=unexplained,
                        patient_context=patient_context,
                        axis_scores=axis_scores
                    )
                    logger.info(f"[DiagnosticEngine] Layer 4e: {len(llm_diagnoses)} LLM diagnoses")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] LLM reasoning error: {e}")

        # Layer 4f: HPO Phenotype Mapping
        # Map abnormalities to HPO terms, then to diseases
        hpo_diagnoses = []
        if self.hpo_mapper_available and HPO_MAPPER_AVAILABLE:
            try:
                hpo_diagnoses = self.hpo_mapper.analyze(
                    axis_scores=axis_scores,
                    features=features,
                    raw_data=raw_patient_data
                )
                if hpo_diagnoses:
                    logger.info(f"[DiagnosticEngine] Layer 4f: {len(hpo_diagnoses)} HPO diagnoses")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] HPO mapping error: {e}")

        # Layer 4g: DisGeNET Genetic Mapping
        # Look up gene-disease associations if genetic markers present
        genetic_diagnoses = []
        if self.disgenet_mapper_available and DISGENET_MAPPER_AVAILABLE:
            try:
                genetic_diagnoses = self.disgenet_mapper.analyze(
                    raw_data=raw_patient_data,
                    features=features,
                    axis_scores=axis_scores
                )
                if genetic_diagnoses:
                    logger.info(f"[DiagnosticEngine] Layer 4g: {len(genetic_diagnoses)} genetic diagnoses")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] DisGeNET mapping error: {e}")

        # Layer 4h: Hetionet Gene-Disease Mapping (supplements 4g)
        # Additional gene-disease associations from Hetionet knowledge graph
        hetionet_diagnoses = []
        if self.hetionet_mapper_available and HETIONET_MAPPER_AVAILABLE:
            try:
                hetionet_diagnoses = self.hetionet_mapper.analyze(
                    raw_data=raw_patient_data,
                    features=features,
                    axis_scores=axis_scores
                )
                if hetionet_diagnoses:
                    logger.info(f"[DiagnosticEngine] Layer 4h: {len(hetionet_diagnoses)} Hetionet diagnoses")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] Hetionet mapping error: {e}")

        # Layer 4i: PrimeKG Disease Mechanism Paths
        # Explains WHY abnormal biomarkers suggest specific diseases
        mechanism_diagnoses = []
        if self.primekg_mapper_available and PRIMEKG_MAPPER_AVAILABLE:
            try:
                mechanism_diagnoses = self.primekg_mapper.analyze(
                    raw_data=raw_patient_data,
                    features=features,
                    axis_scores=axis_scores
                )
                if mechanism_diagnoses:
                    logger.info(f"[DiagnosticEngine] Layer 4i: {len(mechanism_diagnoses)} mechanism diagnoses")
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] PrimeKG mapping error: {e}")

        # Layer 5: Detect unknown anomalies
        anomalies = self.anomaly_detector.detect_anomalies(features, axis_scores, diseases)

        # Layer 6: Analyze convergence
        convergence = self.convergence_engine.analyze_convergence(axis_scores, diseases, anomalies)

        # UTILITY GATE (Phase 6) - Shadow Mode Evaluation
        utility_gate_results = None
        if self.utility_gate_available and UTILITY_GATE_AVAILABLE and diseases:
            try:
                patient_id = normalized.get('patient_id', 'unknown')
                gated_results = []
                for disease in diseases:
                    # Build evidence items
                    evidence_items = [
                        EvidenceItem(kind="lab", label=str(ev), value=ev, weight=0.8)
                        for ev in disease.get('evidence', [])
                    ]
                    for source in disease.get('sources', []):
                        evidence_items.append(EvidenceItem(
                            kind="source", label=source, value=source, weight=0.6
                        ))
                    
                    # Create UtilityInput
                    candidate = UtilityInput(
                        entity_id=f"{patient_id}_{disease.get('disease_name', 'unknown')}",
                        entity_type="patient_alert",
                        mode=self.utility_gate.mode,
                        title=disease.get('disease_name', 'Unknown'),
                        summary=disease.get('description', ''),
                        risk_probability=disease.get('confidence', 0.5),
                        ppv_estimate=min(0.95, disease.get('confidence', 0.5) + len(disease.get('sources', [])) * 0.05),
                        confidence_score=disease.get('confidence', 0.5),
                        novelty_score=max(0.2, 1.0 - disease.get('confidence', 0.5) * 0.5),
                        explainability_score=min(1.0, 0.4 + len(disease.get('evidence', [])) * 0.1),
                        actionability_score=min(1.0, disease.get('confidence', 0.5) * 0.7 + (0.3 if disease.get('icd10') else 0)),
                        evidence=evidence_items,
                        metadata={"icd10": disease.get('icd10'), "sources": disease.get('sources', [])}
                    )
                    
                    # Evaluate
                    decision = self.utility_gate.evaluate(candidate)
                    gated_results.append({
                        "disease": disease.get('disease_name'),
                        "decision": {
                            "action": decision.action.value,
                            "should_surface": decision.should_surface,
                            "should_escalate": decision.should_escalate,
                            "priority": decision.priority,
                            "breakdown": {
                                "rightness": round(decision.breakdown.rightness, 3),
                                "novelty": round(decision.breakdown.novelty, 3),
                                "convincing": round(decision.breakdown.convincing, 3),
                                "handler_score": round(decision.breakdown.handler_score, 3),
                                "net_utility": round(decision.breakdown.net_utility, 1),
                            }
                        }
                    })
                
                utility_gate_results = {
                    'mode': 'shadow',
                    'deployment_mode': self.utility_gate.mode.value,
                    'total_candidates': len(gated_results),
                    'surfaced_count': sum(1 for r in gated_results if r['decision']['should_surface']),
                    'suppressed_count': sum(1 for r in gated_results if not r['decision']['should_surface']),
                    'escalated_count': sum(1 for r in gated_results if r['decision']['should_escalate']),
                    'decisions': gated_results
                }
            except Exception as e:
                logger.warning(f"[DiagnosticEngine] Utility Gate evaluation error: {e}")

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
            'engine_version': '2.5.0-kg-integrated',

            # Utility Gate (Phase 6)
            'utility_gate': utility_gate_results,

            # LLM Medical Reasoning (Layer 4e)
            'layer_4e_llm_diagnoses': llm_diagnoses,

            # HPO Phenotype Mapping (Layer 4f)
            'layer_4f_hpo_diagnoses': hpo_diagnoses,

            # DisGeNET Genetic Mapping (Layer 4g)
            'layer_4g_genetic_diagnoses': genetic_diagnoses,

            # Hetionet Gene-Disease Mapping (Layer 4h)
            'layer_4h_hetionet_diagnoses': hetionet_diagnoses,

            # PrimeKG Disease Mechanism Paths (Layer 4i)
            'layer_4i_mechanism_diagnoses': mechanism_diagnoses
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

    def _get_unexplained_abnormalities(
        self,
        features: Dict,
        axis_scores: Dict,
        diseases: List
    ) -> Dict[str, Any]:
        """
        Identify abnormal values not explained by existing disease matches.

        Args:
            features: Engineered features from Layer 2
            axis_scores: Axis scores from Layer 3
            diseases: Disease matches from Layer 4

        Returns:
            Dict of unexplained abnormal labs with their values and references
        """
        unexplained = {}

        # Get all evidence markers used by matched diseases
        explained_markers = set()
        for disease in diseases:
            for evidence in disease.get('evidence', []):
                if isinstance(evidence, str):
                    # Extract marker name from evidence string
                    marker = evidence.split(':')[0].strip().lower()
                    explained_markers.add(marker)
                elif isinstance(evidence, dict):
                    marker = evidence.get('marker', '').lower()
                    explained_markers.add(marker)

        # Get abnormal markers from axis scores
        for axis_name, axis_data in axis_scores.items():
            if not isinstance(axis_data, dict):
                continue

            # Check for abnormal markers in this axis
            for marker_data in axis_data.get('abnormal_markers', []):
                if isinstance(marker_data, dict):
                    marker_name = marker_data.get('marker', '').lower()
                    if marker_name and marker_name not in explained_markers:
                        unexplained[marker_name] = {
                            'value': marker_data.get('value'),
                            'unit': marker_data.get('unit', ''),
                            'reference_range': marker_data.get('reference', ''),
                            'status': marker_data.get('status', 'abnormal'),
                            'axis': axis_name,
                            'score': marker_data.get('score', 0)
                        }

        # Also check raw features for abnormal values not in axis scores
        raw_features = features.get('raw_features', {})
        for marker, value in raw_features.items():
            marker_lower = marker.lower()
            if marker_lower not in explained_markers and marker_lower not in unexplained:
                # Check if this value is abnormal based on reference ranges
                if isinstance(value, (int, float)):
                    ref_info = self.reference_ranges.get(marker, {})
                    if ref_info:
                        low = ref_info.get('low', float('-inf'))
                        high = ref_info.get('high', float('inf'))
                        if value < low or value > high:
                            unexplained[marker_lower] = {
                                'value': value,
                                'unit': ref_info.get('unit', ''),
                                'reference_range': f"{low}-{high}",
                                'status': 'low' if value < low else 'high'
                            }

        return unexplained

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
