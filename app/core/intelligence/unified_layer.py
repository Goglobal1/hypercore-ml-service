"""
Unified Intelligence Layer - The Brain of HyperCore

This is the CENTRAL HUB that all modules feed into and query from.
Patterns from any module are stored here.
Correlations are computed here.
Any module can get unified insights from here.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from .patterns import (
    Pattern, PatternType, PatternSource,
    TrajectoryPattern, GenomicPattern, PharmaPattern,
    PathogenPattern, MultiomicPattern, AlertPattern,
    ClinicalPattern
)
from .pattern_store import PatternStore
from .correlator import CrossDomainCorrelator, Correlation
from .insights import InsightGenerator, UnifiedInsight, ViewFocus

# Import trajectory engine
try:
    from app.core.trajectory import EarlyWarningEngine
    TRAJECTORY_AVAILABLE = True
except ImportError:
    TRAJECTORY_AVAILABLE = False

logger = logging.getLogger(__name__)


class UnifiedIntelligenceLayer:
    """
    The Brain of HyperCore.

    - Stores patterns from ALL modules
    - Computes cross-domain correlations
    - Provides unified insights to ANY module
    - Trajectory analysis is the FOUNDATION
    """

    def __init__(self):
        self.store = PatternStore()
        self.correlator = CrossDomainCorrelator()
        self.insight_generator = InsightGenerator()

        if TRAJECTORY_AVAILABLE:
            self.trajectory_engine = EarlyWarningEngine()
        else:
            self.trajectory_engine = None
            logger.warning("Trajectory engine not available")

        self._correlation_cache: Dict[str, List[Correlation]] = {}
        self._insight_cache: Dict[str, UnifiedInsight] = {}

    # =========================================================================
    # PATTERN INGESTION - Modules report patterns here
    # =========================================================================

    def report_pattern(self, pattern: Pattern) -> str:
        """
        Any module can report a pattern.
        Pattern is stored and correlations are updated.
        """
        pattern_id = self.store.store(pattern)

        # Invalidate caches for this patient
        self._correlation_cache.pop(pattern.patient_id, None)
        self._insight_cache.pop(pattern.patient_id, None)

        logger.debug(f"Pattern reported: {pattern.pattern_type.value} for {pattern.patient_id}")

        return pattern_id

    def report_trajectory(
        self,
        patient_id: str,
        patient_data: Dict[str, List[float]],
        timestamps: List[float]
    ) -> List[str]:
        """
        Report longitudinal data for trajectory analysis.
        Creates multiple trajectory patterns from the analysis.
        """
        if not self.trajectory_engine:
            logger.warning("Trajectory engine not available")
            return []

        # Run trajectory analysis
        report = self.trajectory_engine.analyze_patient(patient_id, patient_data, timestamps)

        pattern_ids = []

        # Create patterns from rate changes
        for biomarker, rate_data in report.rate_changes.items():
            alert_level = rate_data.get('alert_level', 'normal')
            if alert_level != 'normal':
                severity = 0.9 if alert_level == 'critical' else 0.6 if alert_level == 'warning' else 0.4

                pattern = TrajectoryPattern(
                    patient_id=patient_id,
                    pattern_type=PatternType.RATE_OF_CHANGE,
                    source=PatternSource.TRAJECTORY,
                    confidence=rate_data.get('confidence', 0.5),
                    severity=severity,
                    biomarker=biomarker,
                    rate_of_change=rate_data.get('current_rate', 0),
                    z_score=rate_data.get('z_score', 0),
                    days_of_trend=rate_data.get('days_of_trend', 0),
                    trajectory_phase=alert_level,
                    related_biomarkers=[biomarker],
                    evidence={
                        'alert_level': alert_level,
                        'domain': report.primary_pattern
                    }
                )
                pattern_ids.append(self.report_pattern(pattern))

        # Create pattern from primary matched pattern
        if report.matched_patterns:
            top_match = report.matched_patterns[0]
            pattern = TrajectoryPattern(
                patient_id=patient_id,
                pattern_type=PatternType.TRAJECTORY_SHAPE,
                source=PatternSource.TRAJECTORY,
                confidence=top_match.get('confidence', 0.5),
                severity=0.7 if report.risk_level in ['high', 'critical'] else 0.4,
                onset_days_ago=report.earliest_signal_days_ago,
                predicted_days_to_event=report.estimated_days_to_event,
                related_biomarkers=list(report.rate_changes.keys()),
                recommendations=report.clinical_recommendations[:5],
                contributing_factors=report.signal_propagation_order[:5],
                evidence={
                    'pattern_name': top_match.get('name'),
                    'domain': top_match.get('name', '').lower().replace(' ', '_'),
                    'features': top_match.get('matched_features', [])
                }
            )
            pattern_ids.append(self.report_pattern(pattern))

        # Create forecast patterns
        for biomarker, forecast in report.forecasts.items():
            crossing_day = forecast.get('predicted_crossing_day', 0)
            if crossing_day > 0:
                pattern = TrajectoryPattern(
                    patient_id=patient_id,
                    pattern_type=PatternType.FORECAST,
                    source=PatternSource.TRAJECTORY,
                    confidence=forecast.get('prediction_confidence', 0.5),
                    severity=0.8 if crossing_day < 7 else 0.5,
                    biomarker=biomarker,
                    forecast_threshold_crossing=crossing_day,
                    related_biomarkers=[biomarker],
                    predicted_days_to_event=crossing_day,
                    evidence={
                        'threshold': forecast.get('threshold'),
                        'model': forecast.get('trajectory_model')
                    }
                )
                pattern_ids.append(self.report_pattern(pattern))

        return pattern_ids

    def report_genomic(
        self,
        patient_id: str,
        gene: str,
        variant: str,
        classification: str,
        clinical_significance: str = "",
        conditions: List[str] = None,
        drug_implications: List[str] = None
    ) -> str:
        """Report a genomic finding."""
        pattern = GenomicPattern(
            patient_id=patient_id,
            pattern_type=PatternType.PHARMACOGENOMIC if drug_implications else PatternType.PATHOGENIC_VARIANT,
            source=PatternSource.GENOMICS,
            confidence=0.95 if classification in ['pathogenic', 'likely_pathogenic'] else 0.7,
            severity=0.8 if classification == 'pathogenic' else 0.5,
            gene=gene,
            variant=variant,
            classification=classification,
            clinical_significance=clinical_significance,
            associated_conditions=conditions or [],
            drug_implications=drug_implications or [],
            related_genes=[gene],
            related_drugs=drug_implications or []
        )
        return self.report_pattern(pattern)

    def report_drug_interaction(
        self,
        patient_id: str,
        drug_a: str,
        drug_b: str,
        interaction_type: str,
        effect: str,
        management: str,
        affected_gene: str = None
    ) -> str:
        """Report a drug interaction."""
        pattern = PharmaPattern(
            patient_id=patient_id,
            pattern_type=PatternType.DRUG_INTERACTION,
            source=PatternSource.PHARMA,
            confidence=0.9,
            severity=0.9 if interaction_type == 'major' else 0.6 if interaction_type == 'moderate' else 0.3,
            drug_a=drug_a,
            drug_b=drug_b,
            interaction_type=interaction_type,
            effect=effect,
            management=management,
            affected_gene=affected_gene,
            related_drugs=[drug_a, drug_b],
            related_genes=[affected_gene] if affected_gene else []
        )
        return self.report_pattern(pattern)

    def report_pathogen(
        self,
        patient_id: str,
        pathogen: str = None,
        infection_type: str = "",
        resistance_genes: List[str] = None,
        outbreak_cluster: str = None
    ) -> str:
        """Report a pathogen finding."""
        pattern = PathogenPattern(
            patient_id=patient_id,
            pattern_type=PatternType.INFECTION_SIGNAL if not outbreak_cluster else PatternType.OUTBREAK_CLUSTER,
            source=PatternSource.PATHOGEN,
            confidence=0.8,
            severity=0.7 if resistance_genes else 0.5,
            pathogen=pathogen,
            infection_type=infection_type,
            resistance_genes=resistance_genes or [],
            outbreak_cluster_id=outbreak_cluster,
            related_genes=resistance_genes or []
        )
        return self.report_pattern(pattern)

    def report_alert(
        self,
        patient_id: str,
        state: str,
        previous_state: str = None,
        duration_hours: float = 0,
        alert_type: str = "informational"
    ) -> str:
        """Report an alert state change."""
        severity_map = {'S3': 1.0, 'S2': 0.7, 'S1': 0.4, 'S0': 0.1}
        pattern = AlertPattern(
            patient_id=patient_id,
            pattern_type=PatternType.STATE_TRANSITION if previous_state else PatternType.ESCALATION,
            source=PatternSource.ALERT_SYSTEM,
            confidence=1.0,
            severity=severity_map.get(state, 0.5),
            alert_state=state,
            previous_state=previous_state,
            state_duration_hours=duration_hours,
            alert_type=alert_type
        )
        return self.report_pattern(pattern)

    def report_clinical_domain(
        self,
        patient_id: str,
        domain: str,
        confidence: float,
        primary_markers: List[str],
        secondary_markers: List[str] = None,
        missing_markers: List[str] = None
    ) -> str:
        """Report a clinical domain classification."""
        # Try to map domain to PatternType
        domain_upper = domain.upper()
        pattern_type = PatternType.MULTI_ORGAN
        if domain_upper in PatternType.__members__:
            pattern_type = PatternType[domain_upper]

        pattern = ClinicalPattern(
            patient_id=patient_id,
            pattern_type=pattern_type,
            source=PatternSource.DOMAIN_CLASSIFIER,
            confidence=confidence,
            severity=0.7 if confidence > 0.8 else 0.5,
            domain=domain,
            domain_confidence=confidence,
            primary_markers=primary_markers,
            secondary_markers=secondary_markers or [],
            missing_markers=missing_markers or [],
            related_biomarkers=primary_markers + (secondary_markers or [])
        )
        return self.report_pattern(pattern)

    # =========================================================================
    # INSIGHT RETRIEVAL - Modules query unified views here
    # =========================================================================

    def get_unified_insight(
        self,
        patient_id: str,
        focus: ViewFocus = ViewFocus.ALL,
        max_age_hours: float = 24
    ) -> UnifiedInsight:
        """
        Get unified insight for a patient.

        Any module can call this with their preferred focus:
        - TIMING: Early risk discovery
        - BIOMARKERS: Analysis
        - INTERVENTION: Trial rescue
        - POPULATION: Surveillance
        - ALERT: Alert system
        - ALL: Everything
        """
        cache_key = f"{patient_id}_{focus.value}"

        # Check cache (1 minute TTL)
        if cache_key in self._insight_cache:
            cached = self._insight_cache[cache_key]
            if (datetime.now() - cached.generated_at).total_seconds() < 60:
                return cached

        # Get patterns
        patterns = self.store.get_by_patient(patient_id, max_age_hours=max_age_hours)

        if not patterns:
            # Return empty insight
            return UnifiedInsight(
                patient_id=patient_id,
                generated_at=datetime.now(),
                focus=focus,
                unified_risk_score=0.0,
                risk_level="low",
                confidence=0.0,
                primary_concern="No patterns detected",
                primary_domain="unknown",
                earliest_signal_days_ago=0.0,
                earliest_signal_source="none",
                estimated_days_to_event=0.0,
                detection_improvement_days=0.0
            )

        # Compute correlations
        if patient_id not in self._correlation_cache:
            self._correlation_cache[patient_id] = self.correlator.correlate(patterns, patient_id)
        correlations = self._correlation_cache[patient_id]

        # Generate insight
        insight = self.insight_generator.generate(patient_id, patterns, correlations, focus)

        # Cache
        self._insight_cache[cache_key] = insight

        return insight

    def get_correlations(self, patient_id: str) -> List[Correlation]:
        """Get cross-domain correlations for a patient."""
        if patient_id not in self._correlation_cache:
            patterns = self.store.get_by_patient(patient_id)
            self._correlation_cache[patient_id] = self.correlator.correlate(patterns, patient_id)
        return self._correlation_cache[patient_id]

    def get_patterns(
        self,
        patient_id: str,
        sources: List[PatternSource] = None
    ) -> List[Pattern]:
        """Get patterns for a patient, optionally filtered by source."""
        return self.store.get_by_patient(patient_id, sources=sources)

    # =========================================================================
    # POPULATION / SURVEILLANCE
    # =========================================================================

    def get_high_risk_patients(
        self,
        min_severity: float = 0.7,
        hours: float = 24
    ) -> List[Dict]:
        """Get all high-risk patients across the system."""
        recent = self.store.get_recent(hours=hours, min_severity=min_severity)

        patient_risks: Dict[str, Dict] = {}
        for p in recent:
            if p.patient_id not in patient_risks:
                patient_risks[p.patient_id] = {
                    'severity': 0,
                    'patterns': 0,
                    'sources': set()
                }
            patient_risks[p.patient_id]['severity'] = max(
                patient_risks[p.patient_id]['severity'], p.severity
            )
            patient_risks[p.patient_id]['patterns'] += 1
            patient_risks[p.patient_id]['sources'].add(p.source.value)

        results = []
        for pid, data in patient_risks.items():
            results.append({
                'patient_id': pid,
                'max_severity': data['severity'],
                'pattern_count': data['patterns'],
                'domains': list(data['sources'])
            })

        return sorted(results, key=lambda x: x['max_severity'], reverse=True)

    def get_population_summary(self) -> Dict:
        """Get summary statistics across all patients."""
        stats = self.store.get_statistics()
        recent = self.store.get_recent(hours=24)

        return {
            **stats,
            "recent_24h": len(recent),
            "high_severity_24h": len([p for p in recent if p.severity > 0.7]),
            "critical_24h": len([p for p in recent if p.severity > 0.9])
        }

    # =========================================================================
    # MAINTENANCE
    # =========================================================================

    def cleanup(self) -> Dict:
        """Run maintenance tasks."""
        expired = self.store.cleanup_expired()
        self._correlation_cache.clear()
        self._insight_cache.clear()
        return {"patterns_expired": expired, "caches_cleared": True}

    def get_health(self) -> Dict:
        """Get health status of intelligence layer."""
        stats = self.store.get_statistics()
        return {
            "status": "healthy",
            "trajectory_engine": "available" if self.trajectory_engine else "unavailable",
            "patterns_stored": stats["total_patterns"],
            "patients_tracked": stats["unique_patients"],
            "correlation_cache_size": len(self._correlation_cache),
            "insight_cache_size": len(self._insight_cache)
        }
