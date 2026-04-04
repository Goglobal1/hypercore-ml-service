"""
Hospital-Wide Aggregate Analysis
For CMO/Executive Dashboard
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


@dataclass
class PatientRiskSummary:
    patient_id: str
    risk_level: str
    risk_score: float
    primary_concerns: List[str]
    convergence_detected: bool
    time_in_system: Optional[str]


@dataclass
class EndpointAggregate:
    endpoint: str
    patients_affected: int
    percentage: float
    average_risk_score: float
    trend: str  # "increasing", "stable", "decreasing"
    critical_count: int


@dataclass
class HospitalAggregate:
    timestamp: str
    total_patients: int

    # Risk distribution
    critical_count: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int

    # Percentages
    critical_percentage: float
    high_risk_percentage: float

    # Endpoint breakdown
    endpoint_summary: List[Dict]
    most_concerning_endpoints: List[str]

    # Convergence (multi-system)
    patients_with_convergence: int
    convergence_percentage: float

    # Disease patterns
    top_identified_conditions: List[Dict]
    unknown_patterns_count: int

    # Clustering
    unit_breakdown: Dict[str, Dict]  # By unit/floor

    # Trends
    trends: Dict[str, str]  # What's getting better/worse

    # Alerts
    alerts_fired_last_hour: int
    alerts_suppressed_last_hour: int
    suppression_rate: float

    # Actionable summary
    executive_summary: str
    immediate_attention: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


class HospitalAggregator:
    """
    Aggregates patient data for executive dashboard.
    """

    def __init__(self, discovery_engine):
        self.discovery_engine = discovery_engine
        self.patient_cache = {}  # Store recent analyses
        self.historical_data = []  # For trend detection

    def analyze_hospital(
        self,
        patients: List[Dict],
        include_trends: bool = True,
        include_clustering: bool = True
    ) -> HospitalAggregate:
        """
        Analyze all patients for executive view.

        Args:
            patients: List of patient data dicts
            include_trends: Calculate trends over time
            include_clustering: Group by unit/floor

        Returns:
            HospitalAggregate with full analysis
        """
        if not patients:
            return self._empty_aggregate()

        # Analyze each patient
        patient_results = []
        for patient in patients:
            patient_id = patient.get('patient_id', patient.get('id', f'P{len(patient_results)}'))

            # Run individual analysis
            result = self.discovery_engine.discover_single(patient)
            result['patient_id'] = patient_id
            result['unit'] = patient.get('unit', patient.get('floor', 'Unknown'))
            result['nurse'] = patient.get('nurse', patient.get('assigned_nurse', 'Unknown'))
            result['physician'] = patient.get('physician', patient.get('doctor', 'Unknown'))

            patient_results.append(result)
            self.patient_cache[patient_id] = result

        # Aggregate results
        return self._aggregate_results(patient_results, include_trends, include_clustering)

    def analyze_single_patient(self, patient: Dict) -> Dict:
        """
        Analyze a single patient (for individual alerts).
        Also updates the hospital-wide cache.
        """
        patient_id = patient.get('patient_id', patient.get('id', 'Unknown'))

        # Run analysis
        result = self.discovery_engine.discover_single(patient)
        result['patient_id'] = patient_id
        result['unit'] = patient.get('unit', patient.get('floor', 'Unknown'))
        result['nurse'] = patient.get('nurse', patient.get('assigned_nurse', 'Unknown'))
        result['physician'] = patient.get('physician', patient.get('doctor', 'Unknown'))

        # Cache for aggregate
        self.patient_cache[patient_id] = result

        return result

    def _aggregate_results(
        self,
        results: List[Dict],
        include_trends: bool,
        include_clustering: bool
    ) -> HospitalAggregate:
        """
        Aggregate individual results into hospital view.
        """
        total = len(results)

        # Risk distribution
        risk_counts = {'critical': 0, 'high': 0, 'moderate': 0, 'low': 0, 'minimal': 0}
        for r in results:
            # Get risk level from summary or convergence
            summary = r.get('summary', {})
            level = summary.get('overall_risk', r.get('risk_level', 'low'))
            if level in risk_counts:
                risk_counts[level] += 1
            else:
                risk_counts['low'] += 1

        # Endpoint aggregation
        endpoint_data = defaultdict(lambda: {'count': 0, 'scores': [], 'critical': 0})
        for r in results:
            for endpoint in r.get('endpoints_analyzed', []):
                endpoint_data[endpoint]['count'] += 1

            # Get endpoint-specific scores if available
            for endpoint, ep_result in r.get('endpoint_results', {}).items():
                score = ep_result.get('risk_score', 0) if isinstance(ep_result, dict) else 0
                endpoint_data[endpoint]['scores'].append(score)
                if score >= 80:
                    endpoint_data[endpoint]['critical'] += 1

        endpoint_summary = []
        for endpoint, data in endpoint_data.items():
            avg_score = statistics.mean(data['scores']) if data['scores'] else 0
            endpoint_summary.append({
                'endpoint': endpoint,
                'patients_affected': data['count'],
                'percentage': (data['count'] / total * 100) if total > 0 else 0,
                'average_risk_score': round(avg_score, 1),
                'critical_count': data['critical']
            })

        # Sort by risk
        endpoint_summary.sort(key=lambda x: x['average_risk_score'], reverse=True)
        most_concerning = [e['endpoint'] for e in endpoint_summary[:3]]

        # Convergence count
        convergence_count = sum(
            1 for r in results
            if r.get('convergence', {}).get('convergence_type', 'none') != 'none'
        )

        # Disease patterns
        disease_counts = defaultdict(int)
        unknown_count = 0
        for r in results:
            for disease in r.get('identified_diseases', []):
                name = disease.get('disease_name', 'Unknown')
                disease_counts[name] += 1
            unknown_count += len(r.get('unknown_patterns', []))

        top_conditions = [
            {'condition': name, 'count': count, 'percentage': round(count/total*100, 1)}
            for name, count in sorted(disease_counts.items(), key=lambda x: -x[1])[:5]
        ]

        # Unit clustering
        unit_breakdown = {}
        if include_clustering:
            unit_data = defaultdict(lambda: {'total': 0, 'critical': 0, 'high': 0})
            for r in results:
                unit = r.get('unit', 'Unknown')
                unit_data[unit]['total'] += 1
                summary = r.get('summary', {})
                level = summary.get('overall_risk', r.get('risk_level', 'low'))
                if level == 'critical':
                    unit_data[unit]['critical'] += 1
                elif level == 'high':
                    unit_data[unit]['high'] += 1

            unit_breakdown = dict(unit_data)

        # Trends (compare to historical if available)
        trends = {}
        if include_trends and self.historical_data:
            # Compare to last snapshot
            last = self.historical_data[-1]

            # Risk trend
            if risk_counts['critical'] > last.get('critical', 0):
                trends['critical_patients'] = 'increasing'
            elif risk_counts['critical'] < last.get('critical', 0):
                trends['critical_patients'] = 'decreasing'
            else:
                trends['critical_patients'] = 'stable'

        # Build executive summary
        executive_summary = self._build_executive_summary(
            total, risk_counts, most_concerning, convergence_count, top_conditions
        )

        # Immediate attention items
        immediate_attention = []
        for r in results:
            summary = r.get('summary', {})
            level = summary.get('overall_risk', r.get('risk_level', 'low'))
            if level == 'critical':
                patient_id = r.get('patient_id', 'Unknown')
                recs = r.get('recommendations', [])
                concerns = [rec.get('action', str(rec)) if isinstance(rec, dict) else str(rec) for rec in recs[:2]]
                immediate_attention.append(f"Patient {patient_id}: {', '.join(concerns) if concerns else 'Multiple critical concerns'}")

        # Recommendations
        recommendations = self._build_recommendations(
            risk_counts, endpoint_summary, unit_breakdown
        )

        # Store for trend analysis
        self.historical_data.append({
            'timestamp': datetime.now().isoformat(),
            'critical': risk_counts['critical'],
            'high': risk_counts['high'],
            'total': total
        })

        # Keep only last 24 hours of history
        cutoff = datetime.now() - timedelta(hours=24)
        self.historical_data = [
            h for h in self.historical_data
            if datetime.fromisoformat(h['timestamp']) > cutoff
        ]

        return HospitalAggregate(
            timestamp=datetime.now().isoformat(),
            total_patients=total,
            critical_count=risk_counts['critical'],
            high_risk_count=risk_counts['high'],
            moderate_risk_count=risk_counts['moderate'],
            low_risk_count=risk_counts['low'] + risk_counts['minimal'],
            critical_percentage=round(risk_counts['critical']/total*100, 1) if total > 0 else 0,
            high_risk_percentage=round(risk_counts['high']/total*100, 1) if total > 0 else 0,
            endpoint_summary=endpoint_summary,
            most_concerning_endpoints=most_concerning,
            patients_with_convergence=convergence_count,
            convergence_percentage=round(convergence_count/total*100, 1) if total > 0 else 0,
            top_identified_conditions=top_conditions,
            unknown_patterns_count=unknown_count,
            unit_breakdown=unit_breakdown,
            trends=trends,
            alerts_fired_last_hour=0,  # Would come from alert system
            alerts_suppressed_last_hour=0,
            suppression_rate=0.0,
            executive_summary=executive_summary,
            immediate_attention=immediate_attention[:5],
            recommendations=recommendations
        )

    def _build_executive_summary(
        self,
        total: int,
        risk_counts: Dict,
        most_concerning: List[str],
        convergence_count: int,
        top_conditions: List[Dict]
    ) -> str:
        """Build human-readable executive summary."""
        critical = risk_counts['critical']
        high = risk_counts['high']

        summary_parts = [f"Monitoring {total} patients."]

        if critical > 0:
            summary_parts.append(f"{critical} critical ({critical/total*100:.0f}%).")

        if high > 0:
            summary_parts.append(f"{high} high-risk.")

        if convergence_count > 0:
            summary_parts.append(
                f"{convergence_count} showing multi-system involvement."
            )

        if most_concerning:
            summary_parts.append(
                f"Most active concerns: {', '.join(most_concerning)}."
            )

        if top_conditions:
            top = top_conditions[0]['condition']
            summary_parts.append(f"Most common: {top}.")

        return " ".join(summary_parts)

    def _build_recommendations(
        self,
        risk_counts: Dict,
        endpoint_summary: List,
        unit_breakdown: Dict
    ) -> List[str]:
        """Build actionable recommendations."""
        recs = []

        # Critical patient recommendations
        if risk_counts['critical'] > 0:
            recs.append(f"Review {risk_counts['critical']} critical patients immediately")

        # Endpoint-specific recommendations
        for ep in endpoint_summary[:2]:
            if ep['critical_count'] > 0:
                recs.append(
                    f"Focus on {ep['endpoint']}: {ep['critical_count']} patients critical"
                )

        # Unit clustering
        for unit, data in unit_breakdown.items():
            if data['critical'] >= 2:
                recs.append(
                    f"Unit {unit} has {data['critical']} critical patients - investigate clustering"
                )

        return recs[:5]

    def _empty_aggregate(self) -> HospitalAggregate:
        """Return empty aggregate when no patients."""
        return HospitalAggregate(
            timestamp=datetime.now().isoformat(),
            total_patients=0,
            critical_count=0,
            high_risk_count=0,
            moderate_risk_count=0,
            low_risk_count=0,
            critical_percentage=0,
            high_risk_percentage=0,
            endpoint_summary=[],
            most_concerning_endpoints=[],
            patients_with_convergence=0,
            convergence_percentage=0,
            top_identified_conditions=[],
            unknown_patterns_count=0,
            unit_breakdown={},
            trends={},
            alerts_fired_last_hour=0,
            alerts_suppressed_last_hour=0,
            suppression_rate=0,
            executive_summary="No patients currently in system.",
            immediate_attention=[],
            recommendations=["Add patients to begin monitoring"]
        )


# Singleton
_aggregator_instance = None


def get_hospital_aggregator(discovery_engine=None):
    """Get singleton aggregator instance."""
    global _aggregator_instance
    if _aggregator_instance is None:
        if discovery_engine is None:
            from .main import get_discovery_engine
            discovery_engine = get_discovery_engine()
        _aggregator_instance = HospitalAggregator(discovery_engine)
    return _aggregator_instance
