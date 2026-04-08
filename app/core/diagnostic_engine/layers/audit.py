"""
Layer 9: Audit + Regulatory Trace
Tracks what was measured vs inferred vs predicted for regulatory credibility.
"""

from typing import Dict, List, Any
from datetime import datetime


class AuditLogger:
    """
    Layer 9: Regulatory audit trail.
    """

    VERSION = "2.0.0"

    def create_audit_record(self, input_data: Dict, result: Dict) -> Dict:
        """
        Create comprehensive audit record.
        """

        # Get features from normalized data
        features = input_data.get('features', {})

        # Build evidence sources
        evidence_sources = self._categorize_evidence(result)

        # Build processing trace
        processing = self._build_processing_trace(result)

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'version': self.VERSION,
            'engine': 'HyperCore Diagnostic Engine',

            'input': {
                'patient_id': input_data.get('patient_id', 'unknown'),
                'data_source': input_data.get('metadata', {}).get('source', 'unknown'),
                'biomarkers_received': list(features.keys()),
                'biomarkers_count': len(features),
                'data_completeness': input_data.get('metadata', {}).get('completeness', 0),
                'unmapped_fields': input_data.get('metadata', {}).get('unmapped_fields', [])
            },

            'processing': processing,

            'output': {
                'axes_scored': len(result.get('axis_scores', {})),
                'diseases_detected': len(result.get('diseases', [])),
                'anomalies_detected': len(result.get('anomalies', [])),
                'recommendations_generated': len(result.get('recommendations', [])),
                'clinical_state': result.get('clinical_state'),
                'risk_score': result.get('risk_score')
            },

            'evidence_sources': evidence_sources,

            'validation': {
                'all_layers_executed': processing.get('all_layers_complete', False),
                'data_quality_check': 'passed' if len(features) >= 5 else 'limited_data',
                'confidence_calibration': self._assess_calibration(result)
            },

            'regulatory': {
                'intended_use': 'clinical_decision_support',
                'disclaimer': 'For healthcare professional use only. Clinical correlation required.',
                'not_a_diagnosis': True,
                'requires_clinical_validation': True
            }
        }

    def _categorize_evidence(self, result: Dict) -> Dict:
        """Categorize all evidence by source type."""

        measured = []
        derived = []
        inferred = []
        pattern_matched = []

        # From axis scores - these are measured
        for axis_name, axis_data in result.get('axis_scores', {}).items():
            for marker in axis_data.get('abnormal_markers', []):
                measured.append({
                    'marker': marker.get('marker'),
                    'value': marker.get('value'),
                    'status': marker.get('status'),
                    'axis': axis_name
                })
            for marker in axis_data.get('normal_markers', []):
                measured.append({
                    'marker': marker,
                    'status': 'normal',
                    'axis': axis_name
                })

        # From diseases - these are pattern-matched/inferred
        for disease in result.get('diseases', []):
            pattern_matched.append({
                'finding': disease.get('disease_name'),
                'confidence': disease.get('confidence'),
                'evidence_count': len(disease.get('evidence', [])),
                'pattern_type': 'disease_signature'
            })

        # From anomalies - these are inferred
        for anomaly in result.get('anomalies', []):
            inferred.append({
                'finding': anomaly.get('description'),
                'novelty': anomaly.get('novelty_score'),
                'system': anomaly.get('organ_system'),
                'pattern_type': 'anomaly_detection'
            })

        return {
            'measured': measured,
            'derived': derived,
            'inferred': inferred,
            'pattern_matched': pattern_matched
        }

    def _build_processing_trace(self, result: Dict) -> Dict:
        """Build trace of processing steps."""

        layers = [
            {'layer': 1, 'name': 'input_normalization', 'status': 'complete'},
            {'layer': 2, 'name': 'feature_engineering', 'status': 'complete'},
            {'layer': 3, 'name': 'axis_scoring', 'status': 'complete' if result.get('axis_scores') else 'skipped'},
            {'layer': 4, 'name': 'disease_classification', 'status': 'complete'},
            {'layer': 5, 'name': 'anomaly_detection', 'status': 'complete'},
            {'layer': 6, 'name': 'convergence_analysis', 'status': 'complete' if result.get('convergence') else 'skipped'},
            {'layer': 7, 'name': 'explainability', 'status': 'complete' if result.get('explanation') else 'skipped'},
            {'layer': 8, 'name': 'recommendations', 'status': 'complete' if result.get('recommendations') else 'skipped'},
            {'layer': 9, 'name': 'audit', 'status': 'in_progress'}
        ]

        all_complete = all(l['status'] == 'complete' for l in layers[:-1])

        return {
            'layers_executed': layers,
            'all_layers_complete': all_complete,
            'models_used': [],  # No ML models in rule-based system
            'rules_triggered': self._count_rules_triggered(result)
        }

    def _count_rules_triggered(self, result: Dict) -> int:
        """Count how many detection rules were triggered."""
        count = 0

        # Axis abnormalities
        for axis_data in result.get('axis_scores', {}).values():
            count += len(axis_data.get('abnormal_markers', []))

        # Disease patterns
        count += len(result.get('diseases', []))

        # Anomalies
        count += len(result.get('anomalies', []))

        return count

    def _assess_calibration(self, result: Dict) -> str:
        """Assess confidence calibration quality."""

        diseases = result.get('diseases', [])
        if not diseases:
            return 'not_applicable'

        # Check if confidence scores are well-distributed
        confidences = [d.get('confidence', 0) for d in diseases]

        if all(c > 0.8 for c in confidences):
            return 'possibly_overconfident'
        elif all(c < 0.4 for c in confidences):
            return 'conservative'
        else:
            return 'well_calibrated'

    def format_for_export(self, audit_record: Dict) -> str:
        """Format audit record for export/logging."""
        import json
        return json.dumps(audit_record, indent=2, default=str)
