"""
HyperCore Diagnostic Engine
A signals-first, layered inference system for disease detection.

Architecture:
    Layer 1: Input Normalization - Standardize raw data
    Layer 2: Feature Engineering - Create derived/temporal features
    Layer 3: Axis Scoring - Score biologic organ systems
    Layer 4: Disease Classification - Pattern match known diseases
    Layer 5: Anomaly Detection - Find unknown patterns
    Layer 6: Convergence Analysis - Multi-system reasoning
    Layer 7: Explainability - Generate explanations
    Layer 8: Recommendations - Actionable next steps
    Layer 9: Audit - Regulatory trace
"""

from .pipeline.diagnostic_engine import (
    DiagnosticEngine,
    analyze_patient,
    analyze_patients
)

from .layers.input_normalization import InputNormalizer
from .layers.feature_engineering import FeatureEngineer
from .layers.axis_scoring import AxisScorer
from .layers.disease_classifier import DiseaseClassifier
from .layers.anomaly_detection import AnomalyDetector
from .layers.convergence_engine import ConvergenceEngine
from .layers.explainability import ExplainabilityEngine
from .layers.recommendations import RecommendationEngine
from .layers.audit import AuditLogger

__all__ = [
    'DiagnosticEngine',
    'analyze_patient',
    'analyze_patients',
    'InputNormalizer',
    'FeatureEngineer',
    'AxisScorer',
    'DiseaseClassifier',
    'AnomalyDetector',
    'ConvergenceEngine',
    'ExplainabilityEngine',
    'RecommendationEngine',
    'AuditLogger'
]

__version__ = '2.0.0'
