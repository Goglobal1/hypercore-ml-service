"""
HyperCore Diagnostic Engine
A signals-first, layered inference system for disease detection.

Architecture:
    Layer 1: Input Normalization - Standardize raw data
    Layer 2: Feature Engineering - Create derived/temporal features
    Layer 3: Axis Scoring - Score biologic organ systems
    Layer 4: Disease Classification - Pattern match known diseases
    Layer 4b: ClinVar Integration - Genetic disease detection (NEW)
    Layer 5: Anomaly Detection - Find unknown patterns
    Layer 6: Convergence Analysis - Multi-system reasoning
    Layer 7: Explainability - Generate explanations
    Layer 8: Recommendations - Actionable next steps
    Layer 9: Audit - Regulatory trace

Data Sources:
    - ClinVar: 209K+ genetic diseases, 11K genes
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

# Data Sources (optional - loads if available)
try:
    from .data_sources.clinvar_loader import ClinVarLoader, get_clinvar_loader
    CLINVAR_AVAILABLE = True
except ImportError:
    CLINVAR_AVAILABLE = False
    ClinVarLoader = None
    get_clinvar_loader = None

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
    'AuditLogger',
    'ClinVarLoader',
    'get_clinvar_loader',
    'CLINVAR_AVAILABLE'
]

__version__ = '2.1.0'  # ClinVar integration
