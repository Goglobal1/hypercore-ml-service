"""
Insight Generation - Creates unified views for any module to consume.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

from .patterns import Pattern, PatternSource
from .correlator import Correlation


class ViewFocus(Enum):
    """Different views focus on different aspects of the same intelligence."""
    ALL = "all"
    TIMING = "timing"              # Early risk discovery focus
    BIOMARKERS = "biomarkers"      # Analysis focus
    INTERVENTION = "intervention"  # Trial rescue focus
    POPULATION = "population"      # Surveillance focus
    ALERT = "alert"                # Alert system focus


@dataclass
class UnifiedInsight:
    """
    The complete intelligence picture for a patient.
    Any module can request this and get the SAME underlying intelligence,
    just filtered/emphasized for their specific focus.
    """
    patient_id: str
    generated_at: datetime
    focus: ViewFocus

    # Overall assessment
    unified_risk_score: float
    risk_level: str  # low, moderate, high, critical
    confidence: float

    # Primary concern
    primary_concern: str
    primary_domain: str

    # Timing
    earliest_signal_days_ago: float
    earliest_signal_source: str
    estimated_days_to_event: float
    detection_improvement_days: float

    # Patterns by domain
    trajectory_patterns: List[Dict] = field(default_factory=list)
    genomic_patterns: List[Dict] = field(default_factory=list)
    pharma_patterns: List[Dict] = field(default_factory=list)
    pathogen_patterns: List[Dict] = field(default_factory=list)
    alert_patterns: List[Dict] = field(default_factory=list)

    # Cross-domain correlations
    correlations: List[Dict] = field(default_factory=list)
    correlation_summary: str = ""

    # Cascade detection
    cascade_detected: bool = False
    cascade_levels: List[str] = field(default_factory=list)
    cascade_timeline: Dict[str, float] = field(default_factory=dict)

    # Actionable output
    clinical_recommendations: List[str] = field(default_factory=list)
    monitoring_recommendations: Dict[str, str] = field(default_factory=dict)
    genetic_recommendations: List[str] = field(default_factory=list)

    # Contributing factors
    contributing_factors: List[str] = field(default_factory=list)

    # For specific views
    view_specific_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "patient_id": self.patient_id,
            "generated_at": self.generated_at.isoformat(),
            "focus": self.focus.value,
            "unified_risk_score": self.unified_risk_score,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "primary_concern": self.primary_concern,
            "primary_domain": self.primary_domain,
            "earliest_signal_days_ago": self.earliest_signal_days_ago,
            "earliest_signal_source": self.earliest_signal_source,
            "estimated_days_to_event": self.estimated_days_to_event,
            "detection_improvement_days": self.detection_improvement_days,
            "trajectory_patterns": self.trajectory_patterns,
            "genomic_patterns": self.genomic_patterns,
            "pharma_patterns": self.pharma_patterns,
            "pathogen_patterns": self.pathogen_patterns,
            "alert_patterns": self.alert_patterns,
            "correlations": self.correlations,
            "correlation_summary": self.correlation_summary,
            "cascade_detected": self.cascade_detected,
            "cascade_levels": self.cascade_levels,
            "cascade_timeline": self.cascade_timeline,
            "clinical_recommendations": self.clinical_recommendations,
            "monitoring_recommendations": self.monitoring_recommendations,
            "genetic_recommendations": self.genetic_recommendations,
            "contributing_factors": self.contributing_factors,
            "view_specific": self.view_specific_data
        }


class InsightGenerator:
    """Generates unified insights from patterns and correlations."""

    def generate(
        self,
        patient_id: str,
        patterns: List[Pattern],
        correlations: List[Correlation],
        focus: ViewFocus = ViewFocus.ALL
    ) -> UnifiedInsight:
        """Generate unified insight from patterns and correlations."""

        # Group patterns by source
        by_source: Dict[PatternSource, List[Pattern]] = {}
        for p in patterns:
            by_source.setdefault(p.source, []).append(p)

        # Calculate unified risk
        risk_score, risk_level = self._calculate_risk(patterns, correlations)

        # Find earliest signal
        earliest_days, earliest_source = self._find_earliest(patterns)

        # Estimate timing
        days_to_event = self._estimate_timing(patterns, correlations)

        # Primary concern
        primary_concern, primary_domain = self._identify_primary(patterns, correlations)

        # Check for cascade
        cascade, cascade_levels, cascade_timeline = self._analyze_cascade(correlations)

        # Generate recommendations
        clin_recs, mon_recs, gen_recs = self._generate_recommendations(
            patterns, correlations, risk_level
        )

        # Contributing factors
        factors = self._extract_factors(patterns, correlations)

        # Build insight
        insight = UnifiedInsight(
            patient_id=patient_id,
            generated_at=datetime.now(),
            focus=focus,
            unified_risk_score=risk_score,
            risk_level=risk_level,
            confidence=self._calculate_confidence(patterns, correlations),
            primary_concern=primary_concern,
            primary_domain=primary_domain,
            earliest_signal_days_ago=earliest_days,
            earliest_signal_source=earliest_source,
            estimated_days_to_event=days_to_event,
            detection_improvement_days=earliest_days,
            trajectory_patterns=[p.to_dict() for p in by_source.get(PatternSource.TRAJECTORY, [])],
            genomic_patterns=[p.to_dict() for p in by_source.get(PatternSource.GENOMICS, [])],
            pharma_patterns=[p.to_dict() for p in by_source.get(PatternSource.PHARMA, [])],
            pathogen_patterns=[p.to_dict() for p in by_source.get(PatternSource.PATHOGEN, [])],
            alert_patterns=[p.to_dict() for p in by_source.get(PatternSource.ALERT_SYSTEM, [])],
            correlations=[c.to_dict() for c in correlations],
            correlation_summary=self._summarize_correlations(correlations),
            cascade_detected=cascade,
            cascade_levels=cascade_levels,
            cascade_timeline=cascade_timeline,
            clinical_recommendations=clin_recs,
            monitoring_recommendations=mon_recs,
            genetic_recommendations=gen_recs,
            contributing_factors=factors
        )

        # Add focus-specific data
        insight.view_specific_data = self._add_focus_data(insight, focus, patterns, correlations)

        return insight

    def _calculate_risk(
        self,
        patterns: List[Pattern],
        correlations: List[Correlation]
    ) -> Tuple[float, str]:
        """Calculate unified risk score and level."""
        scores = [p.severity * p.confidence for p in patterns]

        # Correlations increase risk
        for c in correlations:
            scores.append(c.strength * c.risk_multiplier * 0.5)

        if not scores:
            return 0.3, "low"

        # Combined score: max contributes 60%, mean contributes 40%
        score = float(min(max(scores) * 0.6 + np.mean(scores) * 0.4, 1.0))

        if score >= 0.75:
            return score, "critical"
        elif score >= 0.55:
            return score, "high"
        elif score >= 0.35:
            return score, "moderate"
        return score, "low"

    def _find_earliest(self, patterns: List[Pattern]) -> Tuple[float, str]:
        """Find the earliest detected signal."""
        earliest_days, earliest_source = 0.0, "unknown"
        for p in patterns:
            if p.onset_days_ago and p.onset_days_ago > earliest_days:
                earliest_days = p.onset_days_ago
                earliest_source = p.source.value
        return earliest_days, earliest_source

    def _estimate_timing(
        self,
        patterns: List[Pattern],
        correlations: List[Correlation]
    ) -> float:
        """Estimate days to clinical event."""
        estimates = []
        for p in patterns:
            if p.predicted_days_to_event:
                estimates.append(p.predicted_days_to_event)
        for c in correlations:
            if c.detection_improvement_days:
                # Inverse: improvement means event is further away
                estimates.append(14 - c.detection_improvement_days)

        return float(np.mean(estimates)) if estimates else 14.0

    def _identify_primary(
        self,
        patterns: List[Pattern],
        correlations: List[Correlation]
    ) -> Tuple[str, str]:
        """Identify the primary clinical concern."""
        if not patterns:
            return "No significant patterns detected", "unknown"

        # Highest severity pattern
        top = max(patterns, key=lambda p: p.severity * p.confidence)
        domain = top.evidence.get('domain', top.pattern_type.value)

        # Check correlations for better description
        for c in correlations:
            if c.strength > 0.7:
                return c.clinical_significance, domain

        return f"{domain} trajectory detected", domain

    def _analyze_cascade(
        self,
        correlations: List[Correlation]
    ) -> Tuple[bool, List[str], Dict[str, float]]:
        """Check for temporal cascade pattern."""
        for c in correlations:
            if c.correlation_type.value == "temporal_cascade":
                levels = c.domains_involved
                sequence = c.evidence.get("sequence", [])
                timeline = {}
                for i, d in enumerate(levels[:5]):
                    if i < len(sequence):
                        timeline[d] = sequence[i][1]
                return True, levels, timeline
        return False, [], {}

    def _generate_recommendations(
        self,
        patterns: List[Pattern],
        correlations: List[Correlation],
        risk: str
    ) -> Tuple[List[str], Dict[str, str], List[str]]:
        """Generate clinical, monitoring, and genetic recommendations."""
        clin, mon, gen = [], {}, []

        # From risk level
        if risk == "critical":
            clin.extend(["IMMEDIATE: Alert care team - critical status", "ICU evaluation required"])
        elif risk == "high":
            clin.extend(["URGENT: Same-day clinical review", "Prepare escalation protocols"])
        elif risk == "moderate":
            clin.extend(["Increase monitoring frequency", "Clinical review within 24 hours"])

        # From correlations (highest priority insights)
        for c in correlations[:5]:
            if c.action_required:
                clin.append(c.action_required)
            clin.extend(c.recommendations[:2])

        # From patterns
        for p in patterns:
            clin.extend(p.recommendations[:2])
            for gene in p.related_genes[:2]:
                gen.append(f"Consider {gene} pharmacogenomics")
            for bio in p.related_biomarkers[:3]:
                if p.severity > 0.6:
                    mon[bio] = "every 4 hours"
                elif p.severity > 0.3:
                    mon[bio] = "every 8 hours"

        return list(set(clin))[:10], mon, list(set(gen))[:5]

    def _extract_factors(
        self,
        patterns: List[Pattern],
        correlations: List[Correlation]
    ) -> List[str]:
        """Extract contributing factors for the assessment."""
        factors = []
        for p in patterns:
            factors.extend(p.contributing_factors[:3])
        for c in correlations:
            factors.append(f"{c.correlation_type.value}: {c.clinical_significance[:50]}")
        return list(set(factors))[:15]

    def _calculate_confidence(
        self,
        patterns: List[Pattern],
        correlations: List[Correlation]
    ) -> float:
        """Calculate overall confidence in the assessment."""
        if not patterns:
            return 0.3
        confs = [p.confidence for p in patterns]
        confs.extend(c.strength for c in correlations)
        return float(np.mean(confs))

    def _summarize_correlations(self, correlations: List[Correlation]) -> str:
        """Generate human-readable correlation summary."""
        if not correlations:
            return "No cross-domain correlations detected"

        types = set(c.correlation_type.value for c in correlations)
        urgent = [c for c in correlations if c.urgency in ["immediate", "urgent"]]

        summary = f"{len(correlations)} correlations across {len(types)} types."
        if urgent:
            summary += f" {len(urgent)} require urgent attention."

        return summary

    def _add_focus_data(
        self,
        insight: UnifiedInsight,
        focus: ViewFocus,
        patterns: List[Pattern],
        correlations: List[Correlation]
    ) -> Dict:
        """Add focus-specific data to the insight."""
        data = {}

        if focus == ViewFocus.TIMING:
            data["detection_timeline"] = {
                "earliest_signal": insight.earliest_signal_days_ago,
                "threshold_crossing": insight.estimated_days_to_event,
                "improvement": insight.detection_improvement_days
            }
            data["inflection_points"] = [
                p.evidence.get("inflection") for p in patterns
                if p.evidence.get("inflection")
            ]

        elif focus == ViewFocus.BIOMARKERS:
            biomarkers = {}
            for p in patterns:
                for bio in p.related_biomarkers:
                    if bio not in biomarkers:
                        biomarkers[bio] = {"patterns": 0, "max_severity": 0}
                    biomarkers[bio]["patterns"] += 1
                    biomarkers[bio]["max_severity"] = max(
                        biomarkers[bio]["max_severity"], p.severity
                    )
            data["biomarker_summary"] = biomarkers

        elif focus == ViewFocus.INTERVENTION:
            data["intervention_windows"] = []
            for c in correlations:
                if c.detection_improvement_days > 0:
                    data["intervention_windows"].append({
                        "action": c.action_required,
                        "window_days": c.detection_improvement_days,
                        "urgency": c.urgency
                    })

        elif focus == ViewFocus.POPULATION:
            data["patient_count"] = 1
            data["similar_patterns"] = len(patterns)

        elif focus == ViewFocus.ALERT:
            alert_patterns = [p for p in patterns if p.source == PatternSource.ALERT_SYSTEM]
            data["alert_count"] = len(alert_patterns)
            data["escalations"] = sum(
                1 for p in alert_patterns
                if "escalat" in str(p.pattern_type.value).lower()
            )

        return data
