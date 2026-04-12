"""
Diagnostic Engine Layers
Each layer performs a specific transformation in the diagnostic pipeline.

Phase 5: UnifiedDiseaseClassifier combines all disease detection sources.
Layer 4e: LLM Medical Reasoning (Claude API)
Layer 4f: HPO Phenotype Mapping (Human Phenotype Ontology)
Layer 4g: DisGeNET Genetic Mapping (Gene-Disease Associations)
"""

from .input_normalization import InputNormalizer
from .feature_engineering import FeatureEngineer
from .axis_scoring import AxisScorer
from .disease_classifier import DiseaseClassifier
from .unified_disease_classifier import UnifiedDiseaseClassifier
from .anomaly_detection import AnomalyDetector
from .convergence_engine import ConvergenceEngine
from .explainability import ExplainabilityEngine
from .recommendations import RecommendationEngine
from .audit import AuditLogger
from .hpo_mapper import HPOMapper, get_hpo_mapper
from .disgenet_mapper import DisGeNETMapper, get_disgenet_mapper

__all__ = [
    "InputNormalizer",
    "FeatureEngineer",
    "AxisScorer",
    "DiseaseClassifier",
    "UnifiedDiseaseClassifier",
    "AnomalyDetector",
    "ConvergenceEngine",
    "ExplainabilityEngine",
    "RecommendationEngine",
    "AuditLogger",
    "HPOMapper",
    "get_hpo_mapper",
    "DisGeNETMapper",
    "get_disgenet_mapper"
]
