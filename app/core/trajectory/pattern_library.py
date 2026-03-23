"""
Pattern Library - Layer 3 of Trajectory System

Library of known disease trajectory patterns.
Each pattern describes HOW biomarkers behave during disease onset,
not just WHAT thresholds they cross.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class DiseasePattern(Enum):
    SEPSIS_EARLY = "sepsis_early"
    SEPSIS_RAPID = "sepsis_rapid"
    CARDIAC_ACS = "cardiac_acs"
    CARDIAC_CHF = "cardiac_chf"
    RENAL_AKI = "renal_aki"
    RENAL_CKD = "renal_ckd_progression"
    RESPIRATORY_ARDS = "respiratory_ards"
    MULTI_ORGAN = "multi_organ_failure"
    HEPATIC_ACUTE = "hepatic_acute"
    METABOLIC_DKA = "metabolic_dka"


@dataclass
class PatternSignature:
    pattern: DiseasePattern
    name: str
    description: str
    typical_timeline_days: Tuple[int, int]
    required_biomarkers: List[str]
    trajectory_shape: str
    biomarker_sequence: List[str]
    correlation_pattern: Dict[str, str] = field(default_factory=dict)


@dataclass
class PatternMatch:
    pattern: DiseasePattern
    pattern_name: str
    confidence: float
    matched_features: List[str]
    estimated_days_to_event: float
    recommended_actions: List[str]


class PatternLibrary:
    """Library of known disease trajectory patterns."""

    PATTERNS = {
        DiseasePattern.SEPSIS_EARLY: PatternSignature(
            pattern=DiseasePattern.SEPSIS_EARLY,
            name="Early Sepsis",
            description="Gradual onset bacterial infection progressing to sepsis",
            typical_timeline_days=(7, 21),
            required_biomarkers=['procalcitonin', 'wbc', 'lactate', 'crp'],
            trajectory_shape='exponential',
            biomarker_sequence=['crp', 'wbc', 'procalcitonin', 'lactate'],
            correlation_pattern={
                'procalcitonin': 'leads_lactate',
                'crp': 'leads_wbc',
            }
        ),
        DiseasePattern.SEPSIS_RAPID: PatternSignature(
            pattern=DiseasePattern.SEPSIS_RAPID,
            name="Rapid Sepsis",
            description="Rapid onset fulminant infection",
            typical_timeline_days=(1, 4),
            required_biomarkers=['procalcitonin', 'lactate'],
            trajectory_shape='exponential',
            biomarker_sequence=['procalcitonin', 'lactate'],
            correlation_pattern={'procalcitonin': 'correlates_lactate'}
        ),
        DiseasePattern.CARDIAC_ACS: PatternSignature(
            pattern=DiseasePattern.CARDIAC_ACS,
            name="Acute Coronary Syndrome",
            description="Acute myocardial injury pattern",
            typical_timeline_days=(0, 3),
            required_biomarkers=['troponin', 'bnp'],
            trajectory_shape='stepwise',
            biomarker_sequence=['troponin', 'bnp', 'crp'],
            correlation_pattern={'troponin': 'leads_bnp'}
        ),
        DiseasePattern.CARDIAC_CHF: PatternSignature(
            pattern=DiseasePattern.CARDIAC_CHF,
            name="Heart Failure Decompensation",
            description="Gradual worsening of heart failure",
            typical_timeline_days=(7, 30),
            required_biomarkers=['bnp', 'creatinine'],
            trajectory_shape='linear',
            biomarker_sequence=['bnp', 'creatinine', 'troponin'],
            correlation_pattern={'bnp': 'leads_creatinine'}
        ),
        DiseasePattern.RENAL_AKI: PatternSignature(
            pattern=DiseasePattern.RENAL_AKI,
            name="Acute Kidney Injury",
            description="Rapid decline in kidney function",
            typical_timeline_days=(1, 7),
            required_biomarkers=['creatinine', 'egfr'],
            trajectory_shape='exponential',
            biomarker_sequence=['egfr', 'creatinine', 'potassium'],
            correlation_pattern={'creatinine': 'inverse_egfr'}
        ),
        DiseasePattern.MULTI_ORGAN: PatternSignature(
            pattern=DiseasePattern.MULTI_ORGAN,
            name="Multi-Organ Dysfunction",
            description="Progressive failure of multiple organ systems",
            typical_timeline_days=(3, 14),
            required_biomarkers=['lactate', 'creatinine', 'bilirubin'],
            trajectory_shape='exponential',
            biomarker_sequence=['lactate', 'creatinine', 'bilirubin'],
            correlation_pattern={'lactate': 'drives_all'}
        ),
        DiseasePattern.HEPATIC_ACUTE: PatternSignature(
            pattern=DiseasePattern.HEPATIC_ACUTE,
            name="Acute Hepatic Injury",
            description="Acute liver dysfunction",
            typical_timeline_days=(2, 10),
            required_biomarkers=['alt', 'ast', 'bilirubin'],
            trajectory_shape='exponential',
            biomarker_sequence=['alt', 'ast', 'bilirubin', 'albumin'],
            correlation_pattern={'alt': 'correlates_ast'}
        ),
        DiseasePattern.METABOLIC_DKA: PatternSignature(
            pattern=DiseasePattern.METABOLIC_DKA,
            name="Diabetic Ketoacidosis",
            description="Metabolic decompensation in diabetes",
            typical_timeline_days=(1, 5),
            required_biomarkers=['glucose', 'potassium'],
            trajectory_shape='exponential',
            biomarker_sequence=['glucose', 'potassium'],
            correlation_pattern={}
        ),
    }

    def match_patterns(
        self,
        patient_trajectories: Dict[str, List[float]],
        inflection_points: Dict[str, List],
        rate_changes: Dict[str, any]
    ) -> List[PatternMatch]:
        """Match patient data against known disease patterns."""
        matches = []

        for pattern_type, signature in self.PATTERNS.items():
            match_result = self._evaluate_pattern_match(
                signature, patient_trajectories, inflection_points, rate_changes
            )
            if match_result.confidence > 0.3:
                matches.append(match_result)

        return sorted(matches, key=lambda x: x.confidence, reverse=True)

    def _evaluate_pattern_match(
        self,
        signature: PatternSignature,
        trajectories: Dict[str, List[float]],
        inflection_points: Dict[str, List],
        rate_changes: Dict[str, any]
    ) -> PatternMatch:
        """Evaluate how well patient data matches a specific pattern."""
        score = 0.0
        matched_features = []

        # Normalize biomarker names
        available = set(k.lower() for k in trajectories.keys())
        required = set(b.lower() for b in signature.required_biomarkers)
        overlap = available.intersection(required)

        # Check 1: Required biomarkers present (30%)
        biomarker_score = len(overlap) / len(required) if required else 0
        score += biomarker_score * 0.30
        if overlap:
            matched_features.append(f"biomarkers: {', '.join(overlap)}")

        # Check 2: Trajectory shape (25%)
        shape_score = self._evaluate_trajectory_shape(trajectories, signature.trajectory_shape)
        score += shape_score * 0.25
        if shape_score > 0.5:
            matched_features.append(f"shape: {signature.trajectory_shape}")

        # Check 3: Biomarker sequence (25%)
        sequence_score = self._evaluate_biomarker_sequence(inflection_points, signature.biomarker_sequence)
        score += sequence_score * 0.25
        if sequence_score > 0.5:
            matched_features.append("sequence_match")

        # Check 4: Rate of change severity (20%)
        rate_score = self._evaluate_rate_severity(rate_changes, signature.required_biomarkers)
        score += rate_score * 0.20
        if rate_score > 0.5:
            matched_features.append("elevated_rates")

        # Estimate days to event
        days_to_event = self._estimate_days_to_event(inflection_points, signature)

        # Generate recommendations
        recommendations = self._generate_recommendations(signature, score)

        return PatternMatch(
            pattern=signature.pattern,
            pattern_name=signature.name,
            confidence=min(score, 1.0),
            matched_features=matched_features,
            estimated_days_to_event=days_to_event,
            recommended_actions=recommendations
        )

    def _evaluate_trajectory_shape(
        self,
        trajectories: Dict[str, List[float]],
        expected_shape: str
    ) -> float:
        """Check if trajectory matches expected shape."""
        scores = []

        for biomarker, values in trajectories.items():
            if len(values) < 4:
                continue

            try:
                values = np.array([float(v) for v in values])
                x = np.arange(len(values))

                if expected_shape == 'exponential':
                    log_values = np.log(values + 0.01)
                    slope, intercept = np.polyfit(x, log_values, 1)
                    predicted = np.exp(intercept + slope * x)
                    ss_res = np.sum((values - predicted) ** 2)
                    ss_tot = np.sum((values - np.mean(values)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    scores.append(max(r_squared, 0))

                elif expected_shape == 'linear':
                    slope, intercept = np.polyfit(x, values, 1)
                    predicted = intercept + slope * x
                    ss_res = np.sum((values - predicted) ** 2)
                    ss_tot = np.sum((values - np.mean(values)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    scores.append(max(r_squared, 0))

                elif expected_shape == 'stepwise':
                    diffs = np.diff(values)
                    max_jump = np.max(np.abs(diffs))
                    avg_change = np.mean(np.abs(diffs)) or 1
                    jump_ratio = max_jump / avg_change
                    scores.append(min(jump_ratio / 5.0, 1.0))
            except:
                continue

        return float(np.mean(scores)) if scores else 0.5

    def _evaluate_biomarker_sequence(
        self,
        inflection_points: Dict[str, List],
        expected_sequence: List[str]
    ) -> float:
        """Check if biomarkers are rising in expected order."""
        if not inflection_points:
            return 0.5

        first_inflections = {}
        for biomarker, points in inflection_points.items():
            if points:
                first_inflections[biomarker.lower()] = max(p.days_ago for p in points)

        if len(first_inflections) < 2:
            return 0.5

        expected_lower = [b.lower() for b in expected_sequence]
        available = [b for b in expected_lower if b in first_inflections]

        if len(available) < 2:
            return 0.5

        actual_order = sorted(available, key=lambda b: first_inflections[b], reverse=True)
        expected_order = [b for b in expected_lower if b in available]

        matches = sum(1 for a, e in zip(actual_order, expected_order) if a == e)
        return matches / len(available)

    def _evaluate_rate_severity(
        self,
        rate_changes: Dict[str, any],
        required_biomarkers: List[str]
    ) -> float:
        """Check if relevant biomarkers have elevated rates."""
        if not rate_changes:
            return 0.5

        scores = []
        for biomarker in required_biomarkers:
            biomarker_lower = biomarker.lower()
            for key, result in rate_changes.items():
                if key.lower() == biomarker_lower:
                    if hasattr(result, 'alert_level'):
                        if result.alert_level == 'critical':
                            scores.append(1.0)
                        elif result.alert_level == 'warning':
                            scores.append(0.75)
                        elif result.alert_level == 'elevated':
                            scores.append(0.5)
                        else:
                            scores.append(0.25)

        return float(np.mean(scores)) if scores else 0.5

    def _estimate_days_to_event(
        self,
        inflection_points: Dict[str, List],
        signature: PatternSignature
    ) -> float:
        """Estimate days until predicted event."""
        min_days_ago = float('inf')

        for biomarker in signature.required_biomarkers:
            biomarker_lower = biomarker.lower()
            for key, points in inflection_points.items():
                if key.lower() == biomarker_lower and points:
                    for point in points:
                        if point.days_ago < min_days_ago:
                            min_days_ago = point.days_ago

        if min_days_ago == float('inf'):
            return float(signature.typical_timeline_days[1])

        min_timeline, max_timeline = signature.typical_timeline_days
        avg_timeline = (min_timeline + max_timeline) / 2

        return max(avg_timeline - min_days_ago, 0.5)

    def _generate_recommendations(
        self,
        signature: PatternSignature,
        confidence: float
    ) -> List[str]:
        """Generate clinical recommendations."""
        recommendations = []

        if confidence > 0.7:
            recommendations.append(f"HIGH ALERT: {signature.name} pattern detected (confidence: {confidence:.0%})")
            recommendations.append(f"Typical timeline: {signature.typical_timeline_days[0]}-{signature.typical_timeline_days[1]} days to event")
            recommendations.append("URGENT: Immediate clinical review recommended")
        elif confidence > 0.5:
            recommendations.append(f"WARNING: Possible {signature.name} (confidence: {confidence:.0%})")
            recommendations.append("Increase monitoring frequency")
            recommendations.append(f"Watch closely: {', '.join(signature.required_biomarkers)}")
        else:
            recommendations.append(f"Monitor for {signature.name}")
            recommendations.append("Continue standard surveillance")

        return recommendations
