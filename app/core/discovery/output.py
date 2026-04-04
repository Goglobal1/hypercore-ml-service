"""
Discovery Output Layer
Builds the final comprehensive output
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class DiscoveryOutput:
    success: bool
    timestamp: str
    patient_count: int
    endpoints_analyzed: List[str]
    endpoint_results: Dict[str, Any]
    convergence: Dict[str, Any]
    identified_diseases: List[Dict]
    unknown_patterns: List[Dict]
    anomalies: List[Dict]
    recommendations: List[Dict]
    summary: Dict[str, Any]
    raw_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'timestamp': self.timestamp,
            'patient_count': self.patient_count,
            'endpoints_analyzed': self.endpoints_analyzed,
            'endpoint_results': self.endpoint_results,
            'convergence': self.convergence,
            'identified_diseases': self.identified_diseases,
            'unknown_patterns': self.unknown_patterns,
            'anomalies': self.anomalies,
            'recommendations': self.recommendations,
            'summary': self.summary,
            'raw_metrics': self.raw_metrics
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class OutputBuilder:
    """
    Builds the final discovery output from all layers.
    """

    def build(
        self,
        endpoint_results: Dict[str, Any],
        convergence_result: Any,
        disease_result: Dict[str, Any],
        anomalies: List[Any],
        ingestion_result: Dict[str, Any]
    ) -> DiscoveryOutput:
        """
        Build comprehensive output from all analysis layers.
        """
        endpoint_dicts = {}
        for endpoint, result in endpoint_results.items():
            if hasattr(result, 'to_dict'):
                endpoint_dicts[endpoint] = result.to_dict()
            else:
                endpoint_dicts[endpoint] = result

        if hasattr(convergence_result, 'to_dict'):
            convergence_dict = convergence_result.to_dict()
        else:
            convergence_dict = convergence_result

        diseases = disease_result.get('identified_diseases', [])
        disease_dicts = []
        for d in diseases:
            if hasattr(d, 'to_dict'):
                disease_dicts.append(d.to_dict())
            else:
                disease_dicts.append(d)

        unknown_patterns = disease_result.get('unknown_patterns', [])
        unknown_dicts = []
        for u in unknown_patterns:
            if hasattr(u, 'to_dict'):
                unknown_dicts.append(u.to_dict())
            else:
                unknown_dicts.append(u)

        anomaly_dicts = []
        for a in anomalies:
            if hasattr(a, 'to_dict'):
                anomaly_dicts.append(a.to_dict())
            else:
                anomaly_dicts.append(a)

        recommendations = self._generate_recommendations(
            endpoint_dicts, convergence_dict, disease_dicts, anomaly_dicts
        )

        summary = self._build_summary(
            endpoint_dicts, convergence_dict, disease_dicts, unknown_dicts, anomaly_dicts
        )

        return DiscoveryOutput(
            success=True,
            timestamp=datetime.utcnow().isoformat(),
            patient_count=ingestion_result.get('patient_count', 0),
            endpoints_analyzed=list(endpoint_results.keys()),
            endpoint_results=endpoint_dicts,
            convergence=convergence_dict,
            identified_diseases=disease_dicts,
            unknown_patterns=unknown_dicts,
            anomalies=anomaly_dicts,
            recommendations=recommendations,
            summary=summary,
            raw_metrics={
                'columns_mapped': ingestion_result.get('columns_mapped', 0),
                'total_columns': ingestion_result.get('total_columns', 0),
                'endpoints_with_data': len(endpoint_results)
            }
        )

    def _generate_recommendations(
        self,
        endpoint_results: Dict,
        convergence: Dict,
        diseases: List[Dict],
        anomalies: List[Dict]
    ) -> List[Dict]:
        """Generate prioritized recommendations."""
        recommendations = []
        priority = 1

        conv_type = convergence.get('convergence_type', 'none')
        if conv_type in ['critical', 'severe']:
            recommendations.append({
                'priority': priority,
                'category': 'convergence',
                'action': 'Immediate clinical review required',
                'reason': convergence.get('description', 'Multi-system convergence detected'),
                'urgency': 'immediate',
                'systems': convergence.get('systems_involved', [])
            })
            priority += 1

        for disease in diseases:
            if disease.get('confidence') == 'high':
                recommendations.append({
                    'priority': priority,
                    'category': 'disease',
                    'action': f"Evaluate for {disease.get('disease_name', 'unknown')}",
                    'reason': f"High confidence match: {disease.get('description', '')}",
                    'urgency': 'high',
                    'icd10_codes': disease.get('icd10_codes', [])
                })
                priority += 1

        for anomaly in anomalies:
            if anomaly.get('severity') == 'severe':
                recommendations.append({
                    'priority': priority,
                    'category': 'anomaly',
                    'action': anomaly.get('recommendation', 'Investigate anomaly'),
                    'reason': anomaly.get('description', 'Severe anomaly detected'),
                    'urgency': 'high'
                })
                priority += 1

        for endpoint, result in endpoint_results.items():
            if result.get('risk_level') == 'critical':
                recommendations.append({
                    'priority': priority,
                    'category': 'endpoint',
                    'action': f"Address {endpoint} abnormalities",
                    'reason': f"Critical risk level in {endpoint}",
                    'urgency': 'high',
                    'abnormal_count': len(result.get('abnormal_values', []))
                })
                priority += 1

        for disease in diseases:
            if disease.get('confidence') == 'moderate':
                recommendations.append({
                    'priority': priority,
                    'category': 'disease',
                    'action': f"Consider {disease.get('disease_name', 'unknown')}",
                    'reason': f"Moderate confidence match",
                    'urgency': 'moderate',
                    'missing_indicators': disease.get('missing_indicators', [])
                })
                priority += 1

        return recommendations

    def _build_summary(
        self,
        endpoint_results: Dict,
        convergence: Dict,
        diseases: List[Dict],
        unknown_patterns: List[Dict],
        anomalies: List[Dict]
    ) -> Dict:
        """Build executive summary."""
        critical_endpoints = []
        warning_endpoints = []
        watch_endpoints = []

        for endpoint, result in endpoint_results.items():
            level = result.get('risk_level', 'normal')
            if level == 'critical':
                critical_endpoints.append(endpoint)
            elif level == 'warning':
                warning_endpoints.append(endpoint)
            elif level == 'watch':
                watch_endpoints.append(endpoint)

        high_confidence_diseases = [d for d in diseases if d.get('confidence') == 'high']
        moderate_confidence_diseases = [d for d in diseases if d.get('confidence') == 'moderate']

        severe_anomalies = [a for a in anomalies if a.get('severity') == 'severe']

        if critical_endpoints or convergence.get('convergence_type') == 'critical':
            overall_risk = 'critical'
        elif warning_endpoints or convergence.get('convergence_type') == 'severe':
            overall_risk = 'high'
        elif watch_endpoints or convergence.get('convergence_type') == 'moderate':
            overall_risk = 'moderate'
        else:
            overall_risk = 'low'

        return {
            'overall_risk': overall_risk,
            'critical_systems': critical_endpoints,
            'warning_systems': warning_endpoints,
            'watch_systems': watch_endpoints,
            'high_confidence_diseases': [d.get('disease_name') for d in high_confidence_diseases],
            'possible_diseases': [d.get('disease_name') for d in moderate_confidence_diseases],
            'convergence_type': convergence.get('convergence_type', 'none'),
            'convergence_score': convergence.get('convergence_score', 0),
            'time_to_harm': convergence.get('estimated_time_to_harm'),
            'unexplained_abnormalities': len(unknown_patterns),
            'severe_anomalies': len(severe_anomalies),
            'total_anomalies': len(anomalies),
            'total_diseases_matched': len(diseases)
        }
