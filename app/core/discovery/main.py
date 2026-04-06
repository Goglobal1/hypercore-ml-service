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

    def discover(self, data: Union[pd.DataFrame, Dict, List], per_patient: bool = True) -> Dict[str, Any]:
        """
        Run full discovery on patient data.

        When per_patient=True (default), analyzes each patient individually
        and returns both individual results and aggregate statistics.

        Args:
            data: Patient data as DataFrame, dict, or list of records
            per_patient: If True, analyze each patient individually (default: True)

        Returns:
            Comprehensive discovery results with per-patient analysis
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

            # Check if we have multiple patients and per_patient analysis is enabled
            patient_col = self._find_patient_id_column(df)

            if per_patient and patient_col is not None:
                unique_patients = df[patient_col].nunique()
                if unique_patients > 1:
                    # Multiple patients - use per-patient batch analysis
                    return self.discover_batch(df)

            # Single patient or per_patient disabled - use original analysis
            return self._analyze_dataframe(df)

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
        result = self._analyze_dataframe(df)

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

    def _analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Core analysis logic for a DataFrame.
        Used by both single-patient and batch analysis.
        """
        try:
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

    def _find_patient_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the patient ID column in a DataFrame."""
        patient_id_patterns = [
            'patient_id', 'patientid', 'patient', 'id', 'mrn', 'subject_id',
            'subjectid', 'hadm_id', 'encounter_id', 'encounterid', 'stay_id',
            'icustay_id', 'record_id', 'recordid', 'case_id', 'caseid'
        ]

        for col in df.columns:
            col_lower = str(col).lower().strip()
            for pattern in patient_id_patterns:
                if pattern == col_lower or pattern in col_lower:
                    return col

        return None

    def discover_batch(self, data: Union[pd.DataFrame, List[Dict]]) -> Dict[str, Any]:
        """
        Analyze multiple patients individually, then aggregate results.

        This is the KEY method for proper per-patient analysis:
        1. Identifies each unique patient
        2. Analyzes each patient INDIVIDUALLY
        3. Stores individual results in patient_results array
        4. Calculates aggregate statistics FROM individual results
        5. Returns BOTH individual and aggregate views

        Args:
            data: DataFrame or list of patient records

        Returns:
            Dict with:
                - patient_results: Array of individual patient analyses
                - aggregate: Summary statistics across all patients
                - summary: Overall risk assessment
        """
        try:
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                return self._error_response(f"Unsupported data type: {type(data)}")

            # Find patient ID column
            patient_col = self._find_patient_id_column(df)

            if patient_col is None:
                # No patient ID - treat entire dataset as one "patient"
                result = self._analyze_dataframe(df)
                result['analysis_type'] = 'batch_no_patient_id'
                result['patient_results'] = []
                result['aggregate'] = result.get('summary', {})
                return result

            # Get unique patients
            unique_patients = df[patient_col].unique()
            total_patients = len(unique_patients)

            # Analyze each patient individually
            patient_results = []
            risk_distribution = {'critical': 0, 'high': 0, 'moderate': 0, 'watch': 0, 'low': 0, 'unknown': 0}
            all_critical_systems = []
            all_warning_systems = []
            all_diseases = []
            all_anomalies = []
            convergence_scores = []
            endpoints_seen = set()

            for idx, patient_id in enumerate(unique_patients):
                # Get this patient's data
                patient_df = df[df[patient_col] == patient_id]

                # Run analysis on this patient
                patient_result = self._analyze_dataframe(patient_df)

                # Add patient identifier
                patient_result['patient_id'] = str(patient_id)
                patient_result['patient_index'] = idx

                # Extract summary info
                summary = patient_result.get('summary', {})
                risk_level = summary.get('overall_risk', 'unknown')
                patient_result['risk_level'] = risk_level

                # Track risk distribution
                if risk_level in risk_distribution:
                    risk_distribution[risk_level] += 1
                else:
                    risk_distribution['unknown'] += 1

                # Collect critical/warning systems
                all_critical_systems.extend(summary.get('critical_systems', []))
                all_warning_systems.extend(summary.get('warning_systems', []))

                # Collect diseases
                all_diseases.extend(patient_result.get('identified_diseases', []))

                # Collect anomalies (limit per patient to avoid explosion)
                patient_anomalies = patient_result.get('anomalies', [])[:5]
                for a in patient_anomalies:
                    a['patient_id'] = str(patient_id)
                all_anomalies.extend(patient_anomalies)

                # Track convergence scores
                conv = patient_result.get('convergence', {})
                if conv.get('convergence_score'):
                    convergence_scores.append(conv['convergence_score'])

                # Track endpoints
                endpoints_seen.update(patient_result.get('endpoints_analyzed', []))

                # Store individual result
                patient_results.append(patient_result)

            # Build aggregate statistics FROM individual results
            aggregate = self._build_aggregate(
                patient_results=patient_results,
                risk_distribution=risk_distribution,
                all_critical_systems=all_critical_systems,
                all_warning_systems=all_warning_systems,
                all_diseases=all_diseases,
                convergence_scores=convergence_scores,
                total_patients=total_patients
            )

            # Build final response with BOTH views
            return {
                'success': True,
                'timestamp': datetime.utcnow().isoformat(),
                'analysis_type': 'per_patient_batch',
                'patient_count': total_patients,
                'patient_id_column': patient_col,

                # INDIVIDUAL RESULTS - Every patient's analysis
                'patient_results': patient_results,

                # AGGREGATE - Calculated from individual results
                'aggregate': aggregate,

                # Summary for quick overview
                'summary': {
                    'overall_risk': aggregate['overall_risk'],
                    'critical_patient_count': risk_distribution['critical'],
                    'high_risk_patient_count': risk_distribution['high'],
                    'total_patients_analyzed': total_patients,
                    'risk_distribution': risk_distribution,
                    'endpoints_analyzed': list(endpoints_seen),
                    'unique_diseases_found': len(set(d.get('disease_name', d.get('disease', '')) for d in all_diseases)),
                    'total_anomalies': len(all_anomalies)
                },

                # Top-level fields for compatibility
                'endpoints_analyzed': list(endpoints_seen),
                'identified_diseases': all_diseases[:20],  # Top 20
                'anomalies': all_anomalies[:50],  # Top 50
                'recommendations': aggregate.get('recommendations', []),

                # Raw metrics
                'raw_metrics': {
                    'patients_analyzed': total_patients,
                    'endpoints_with_data': len(endpoints_seen),
                    'analysis_method': 'per_patient_individual'
                }
            }

        except Exception as e:
            return self._error_response(str(e), traceback.format_exc())

    def _build_aggregate(
        self,
        patient_results: List[Dict],
        risk_distribution: Dict[str, int],
        all_critical_systems: List[str],
        all_warning_systems: List[str],
        all_diseases: List[Dict],
        convergence_scores: List[float],
        total_patients: int
    ) -> Dict[str, Any]:
        """Build aggregate statistics from individual patient results."""

        # Count system frequencies
        from collections import Counter
        critical_system_counts = Counter(all_critical_systems)
        warning_system_counts = Counter(all_warning_systems)

        # Disease frequencies
        disease_names = [d.get('disease_name', d.get('disease', 'unknown')) for d in all_diseases]
        disease_counts = Counter(disease_names)

        # Calculate average convergence
        avg_convergence = np.mean(convergence_scores) if convergence_scores else 0

        # Determine overall risk based on patient distribution
        critical_pct = risk_distribution['critical'] / total_patients if total_patients > 0 else 0
        high_pct = risk_distribution['high'] / total_patients if total_patients > 0 else 0

        if critical_pct >= 0.1:  # 10%+ critical
            overall_risk = 'critical'
        elif critical_pct >= 0.05 or high_pct >= 0.2:  # 5%+ critical or 20%+ high
            overall_risk = 'high'
        elif high_pct >= 0.1:  # 10%+ high
            overall_risk = 'moderate'
        else:
            overall_risk = 'low'

        # Build recommendations based on aggregate data
        recommendations = []
        priority = 1

        if risk_distribution['critical'] > 0:
            recommendations.append({
                'priority': priority,
                'category': 'critical_patients',
                'action': f"Immediate review of {risk_distribution['critical']} critical patients",
                'reason': f"{risk_distribution['critical']} patients have critical risk levels",
                'urgency': 'immediate',
                'patient_count': risk_distribution['critical']
            })
            priority += 1

        if critical_system_counts:
            top_critical = critical_system_counts.most_common(3)
            for system, count in top_critical:
                recommendations.append({
                    'priority': priority,
                    'category': 'system_alert',
                    'action': f"Address widespread {system} abnormalities",
                    'reason': f"{count} patients have critical {system} status",
                    'urgency': 'high',
                    'affected_patients': count
                })
                priority += 1

        if disease_counts:
            top_diseases = disease_counts.most_common(5)
            for disease, count in top_diseases:
                recommendations.append({
                    'priority': priority,
                    'category': 'disease_cluster',
                    'action': f"Investigate {disease} cluster",
                    'reason': f"{count} patients show signs of {disease}",
                    'urgency': 'moderate',
                    'affected_patients': count
                })
                priority += 1

        return {
            'overall_risk': overall_risk,
            'total_patients': total_patients,
            'risk_distribution': risk_distribution,
            'risk_percentages': {
                'critical': round(critical_pct * 100, 1),
                'high': round(high_pct * 100, 1),
                'moderate': round(risk_distribution['moderate'] / total_patients * 100, 1) if total_patients > 0 else 0,
                'low': round(risk_distribution['low'] / total_patients * 100, 1) if total_patients > 0 else 0
            },
            'critical_systems': dict(critical_system_counts.most_common(10)),
            'warning_systems': dict(warning_system_counts.most_common(10)),
            'disease_distribution': dict(disease_counts.most_common(10)),
            'average_convergence_score': round(avg_convergence, 2),
            'recommendations': recommendations,
            'analysis_summary': {
                'patients_with_critical_risk': risk_distribution['critical'],
                'patients_with_high_risk': risk_distribution['high'],
                'patients_with_any_disease': len(set(d.get('patient_id', i) for i, d in enumerate(all_diseases))),
                'most_common_critical_system': critical_system_counts.most_common(1)[0] if critical_system_counts else None,
                'most_common_disease': disease_counts.most_common(1)[0] if disease_counts else None
            }
        }

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
