"""
Discovery Engine Orchestrator
Coordinates all 6 layers of the discovery system
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

from .ingestion import UniversalIngestion
from .endpoint_analysis import MultiEndpointAnalyzer
from .convergence import ConvergenceDetector
from .disease_identification import DiseaseIdentifier
from .anomaly_detection import AnomalyDetector
from .output import OutputBuilder, DiscoveryOutput


class DiscoveryEngine:
    """
    The HyperCore Discovery Engine.

    Takes ANY patient data and discovers:
    - Which of 24 body systems show abnormalities
    - Cross-system convergence patterns
    - Known disease matches
    - Unknown patterns requiring investigation
    - Statistical anomalies
    """

    def __init__(self):
        self.ingestion = UniversalIngestion()
        self.convergence_detector = ConvergenceDetector()
        self.disease_identifier = DiseaseIdentifier()
        self.anomaly_detector = AnomalyDetector()
        self.output_builder = OutputBuilder()

    def discover(self, data: Union[pd.DataFrame, Dict, List]) -> Dict[str, Any]:
        """
        Run full discovery on patient data.

        Args:
            data: Patient data as DataFrame, dict, or list of records

        Returns:
            Comprehensive discovery results
        """
        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                return self._error_response(f"Unsupported data type: {type(data)}")

            ingestion_result = self.ingestion.ingest(df)

            if not ingestion_result.get('success', False):
                return self._error_response(
                    ingestion_result.get('error', 'Ingestion failed')
                )

            endpoint_data = ingestion_result.get('endpoint_data', {})
            reference_ranges = ingestion_result.get('reference_ranges', {})

            multi_analyzer = MultiEndpointAnalyzer(reference_ranges)
            endpoint_results = multi_analyzer.analyze_all(endpoint_data)

            endpoint_results_dict = {}
            for ep, result in endpoint_results.items():
                if hasattr(result, 'to_dict'):
                    endpoint_results_dict[ep] = result.to_dict()
                else:
                    endpoint_results_dict[ep] = result

            convergence_result = self.convergence_detector.detect(endpoint_results_dict)

            disease_result = self.disease_identifier.identify(
                endpoint_results_dict, endpoint_data
            )

            anomalies = self.anomaly_detector.detect(endpoint_data)

            output = self.output_builder.build(
                endpoint_results=endpoint_results,
                convergence_result=convergence_result,
                disease_result=disease_result,
                anomalies=anomalies,
                ingestion_result=ingestion_result
            )

            return output.to_dict()

        except Exception as e:
            return self._error_response(str(e), traceback.format_exc())

    def discover_from_json(self, json_data: Dict) -> Dict[str, Any]:
        """
        Discover from JSON payload.
        Handles both single patient and batch.
        """
        if 'patients' in json_data:
            return self.discover(json_data['patients'])
        elif 'data' in json_data:
            return self.discover(json_data['data'])
        else:
            return self.discover(json_data)

    def discover_single(self, patient_data: Dict) -> Dict[str, Any]:
        """
        Analyze a SINGLE patient.

        This is the real-time analysis that happens when a nurse/doctor
        inputs one patient.

        Args:
            patient_data: Dict with patient biomarker values

        Returns:
            Analysis result for this one patient
        """
        # Convert single patient to DataFrame
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = pd.DataFrame(patient_data)

        # Run full discovery
        result = self.discover(df)

        # Add single-patient specific fields
        result['patient_id'] = patient_data.get(
            'patient_id',
            patient_data.get('id', 'Unknown')
        )
        result['analysis_type'] = 'individual'

        # Determine overall risk level from summary
        summary = result.get('summary', {})
        result['risk_level'] = summary.get('overall_risk', 'unknown')

        return result

    def quick_scan(self, data: Union[pd.DataFrame, Dict, List]) -> Dict[str, Any]:
        """
        Quick scan - just convergence and critical issues.
        Faster than full discovery.
        """
        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data

            ingestion_result = self.ingestion.ingest(df)
            endpoint_data = ingestion_result.get('endpoint_data', {})
            reference_ranges = ingestion_result.get('reference_ranges', {})

            multi_analyzer = MultiEndpointAnalyzer(reference_ranges)
            endpoint_results = multi_analyzer.analyze_all(endpoint_data)

            endpoint_results_dict = {}
            critical_endpoints = []
            for ep, result in endpoint_results.items():
                if hasattr(result, 'to_dict'):
                    d = result.to_dict()
                else:
                    d = result
                endpoint_results_dict[ep] = d
                if d.get('risk_level') in ['critical', 'warning']:
                    critical_endpoints.append(ep)

            convergence_result = self.convergence_detector.detect(endpoint_results_dict)

            return {
                'success': True,
                'scan_type': 'quick',
                'timestamp': datetime.utcnow().isoformat(),
                'convergence': convergence_result.to_dict() if hasattr(convergence_result, 'to_dict') else convergence_result,
                'critical_endpoints': critical_endpoints,
                'endpoints_scanned': len(endpoint_results),
                'needs_full_discovery': convergence_result.convergence_type.value != 'none' or len(critical_endpoints) > 0
            }

        except Exception as e:
            return self._error_response(str(e))

    def _error_response(self, error: str, trace: str = None) -> Dict[str, Any]:
        """Build error response."""
        response = {
            'success': False,
            'timestamp': datetime.utcnow().isoformat(),
            'error': error,
            'patient_count': 0,
            'endpoints_analyzed': [],
            'endpoint_results': {},
            'convergence': {'convergence_type': 'none', 'convergence_score': 0},
            'identified_diseases': [],
            'unknown_patterns': [],
            'anomalies': [],
            'recommendations': [],
            'summary': {'overall_risk': 'unknown'}
        }
        if trace:
            response['traceback'] = trace
        return response


_engine_instance = None


def get_discovery_engine() -> DiscoveryEngine:
    """Get singleton discovery engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DiscoveryEngine()
    return _engine_instance
