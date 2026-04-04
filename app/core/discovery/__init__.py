"""
DiviScan Discovery Engine
=========================

A discovery engine that analyzes ANY patient data for ANY disease.
Core principle: Analyze what you have. Suggest what you don't. NEVER block.

Layers:
1. Universal Data Ingestion - Accept any format
2. 24-Endpoint Analysis - Threshold, pattern, trend
3. Cross-System Convergence - Multi-organ detection
4. Disease Identification - Match known + flag unknown
5. Anomaly Detection - Statistical + clinical
6. Output - Always return results
"""

from .main import DiscoveryEngine, get_discovery_engine
from .ingestion import UniversalIngestion, BIOMARKER_MAPPINGS
from .endpoint_analysis import EndpointAnalyzer, MultiEndpointAnalyzer, EndpointResult
from .convergence import ConvergenceDetector, ConvergenceResult
from .disease_identification import DiseaseIdentifier, DiseaseMatch
from .anomaly_detection import AnomalyDetector, Anomaly
from .output import OutputBuilder, DiscoveryOutput
from .hospital_aggregate import HospitalAggregator, HospitalAggregate, get_hospital_aggregator

__all__ = [
    'DiscoveryEngine', 'get_discovery_engine',
    'UniversalIngestion', 'BIOMARKER_MAPPINGS',
    'EndpointAnalyzer', 'MultiEndpointAnalyzer', 'EndpointResult',
    'ConvergenceDetector', 'ConvergenceResult',
    'DiseaseIdentifier', 'DiseaseMatch',
    'AnomalyDetector', 'Anomaly',
    'OutputBuilder', 'DiscoveryOutput',
    'HospitalAggregator', 'HospitalAggregate', 'get_hospital_aggregator'
]
