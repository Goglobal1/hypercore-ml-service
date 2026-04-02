"""
HyperCore 24-Endpoint Cross-Loop Engine V2
==========================================

Analyzes ALL 24 endpoints simultaneously to detect:
1. Known disease pathways
2. Unknown/novel patterns
3. Multi-system convergence
4. Hidden correlations
5. Clinical validation metrics (Handler required)
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np

from app.core.endpoints.endpoint_definitions import (
    ENDPOINT_DEFINITIONS,
    ENDPOINT_CATEGORIES,
    ALL_ENDPOINTS,
    EndpointScorer,
)
from app.core.pathways.pathway_library import (
    PATHWAY_LIBRARY,
    PathwayMatcher,
)


@dataclass
class CrossLoopResult:
    """Result from cross-loop analysis."""
    endpoints_analyzed: int
    endpoints_with_data: int
    endpoints_alerting: List[str]
    n_endpoints_alerting: int
    detected_pathways: List[Dict]
    unknown_pattern: Optional[Dict]
    convergence_score: float
    convergence_type: str
    multi_system_failure: bool
    cross_domain_insights: List[str]
    confidence: float
    data_completeness: float
    category_summary: Dict[str, Dict]
    recommended_actions: List[str]
    urgency: str


class CrossLoopEngineV2:
    """
    24-Endpoint Cross-Loop Analysis Engine.

    This is the core of HyperCore's multi-system detection capability.
    """

    def __init__(self):
        self.endpoint_scorer = EndpointScorer()
        self.pathway_matcher = PathwayMatcher()
        self.endpoints = ENDPOINT_DEFINITIONS
        self.categories = ENDPOINT_CATEGORIES

    def analyze_patient(
        self,
        patient_data: Dict[str, Any],
        mode: str = "balanced"
    ) -> CrossLoopResult:
        """
        Perform full 24-endpoint analysis on patient data.

        Args:
            patient_data: Dict with biomarker values
            mode: Operating mode (screening, balanced, high_confidence)

        Returns:
            CrossLoopResult with full analysis
        """
        # Step 1: Score all 24 endpoints
        endpoint_results = self.endpoint_scorer.score_all_endpoints(patient_data)

        # Step 2: Identify alerting endpoints
        alerting_endpoints = self._get_alerting_endpoints(endpoint_results, mode)

        # Step 3: Match against pathway library
        detected_pathways = self.pathway_matcher.match_pathways(
            alerting_endpoints, endpoint_results
        )

        # Step 4: Detect unknown patterns
        unknown_pattern = self.pathway_matcher.detect_unknown_pattern(
            alerting_endpoints, detected_pathways
        )

        # Step 5: Calculate convergence
        convergence = self._calculate_convergence(alerting_endpoints, detected_pathways)

        # Step 6: Generate cross-domain insights
        insights = self._generate_insights(
            endpoint_results, alerting_endpoints, detected_pathways, unknown_pattern
        )

        # Step 7: Category summary
        category_summary = self._summarize_by_category(endpoint_results)

        # Step 8: Determine urgency
        urgency = self._determine_urgency(detected_pathways, alerting_endpoints)

        # Step 9: Generate recommendations
        recommendations = self._generate_recommendations(
            detected_pathways, unknown_pattern, alerting_endpoints
        )

        return CrossLoopResult(
            endpoints_analyzed=len(endpoint_results),
            endpoints_with_data=sum(1 for r in endpoint_results.values() if r.get("has_data")),
            endpoints_alerting=list(alerting_endpoints.keys()),
            n_endpoints_alerting=len(alerting_endpoints),
            detected_pathways=detected_pathways,
            unknown_pattern=unknown_pattern,
            convergence_score=convergence["score"],
            convergence_type=convergence["type"],
            multi_system_failure=len(alerting_endpoints) >= 4,
            cross_domain_insights=insights,
            confidence=self._calculate_confidence(endpoint_results, detected_pathways),
            data_completeness=self._calculate_data_completeness(endpoint_results),
            category_summary=category_summary,
            recommended_actions=recommendations,
            urgency=urgency,
        )

    def analyze_all_endpoints(
        self,
        vitals: Dict[str, float],
        labs: Dict[str, float]
    ) -> Tuple[Dict[str, Dict], Dict[str, Any]]:
        """
        Score all 24 endpoints and return cross-loop analysis.

        Args:
            vitals: Dict of vital signs
            labs: Dict of lab values

        Returns:
            Tuple of (endpoint_results, cross_loop_analysis)
        """
        # Combine vitals and labs
        patient_data = {**vitals, **labs}

        # Score all endpoints
        endpoint_results = self.endpoint_scorer.score_all_endpoints(patient_data)

        # Get alerting endpoints
        alerting = self._get_alerting_endpoints(endpoint_results, "balanced")

        # Match pathways
        pathways = self.pathway_matcher.match_pathways(alerting, endpoint_results)

        # Detect unknown patterns
        unknown = self.pathway_matcher.detect_unknown_pattern(alerting, pathways)

        # Calculate convergence
        convergence = self._calculate_convergence(alerting, pathways)

        cross_loop = {
            "endpoints_alerting": list(alerting.keys()),
            "n_endpoints_alerting": len(alerting),
            "cross_domain_patterns": [p["pathway_name"] for p in pathways[:5]],
            "detected_pathways": pathways,
            "unknown_pattern": unknown,
            "convergence_detected": convergence["score"] > 0.5,
            "convergence_score": convergence["score"],
            "convergence_type": convergence["type"],
            "multi_system_failure": len(alerting) >= 4,
        }

        return endpoint_results, cross_loop

    def _get_alerting_endpoints(
        self,
        endpoint_results: Dict[str, Dict],
        mode: str
    ) -> Dict[str, Dict]:
        """Get endpoints that are alerting based on mode."""
        thresholds = {
            "screening": {"elevated": True, "borderline": True, "critical": True},
            "balanced": {"elevated": True, "critical": True},
            "high_confidence": {"critical": True, "elevated": True},
        }

        mode_thresholds = thresholds.get(mode, thresholds["balanced"])
        alerting = {}

        for name, result in endpoint_results.items():
            status = result.get("status", "normal")

            # For high_confidence, require higher scores
            if mode == "high_confidence":
                if status == "critical" or (status == "elevated" and result.get("score", 0) > 0.6):
                    alerting[name] = result
            elif mode_thresholds.get(status, False):
                alerting[name] = result

        return alerting

    def _calculate_convergence(
        self,
        alerting_endpoints: Dict[str, Dict],
        detected_pathways: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate convergence score and type."""
        n_alerting = len(alerting_endpoints)

        if n_alerting == 0:
            return {"score": 0.0, "type": "none"}
        elif n_alerting == 1:
            return {"score": 0.2, "type": "single_system"}
        elif n_alerting == 2:
            return {"score": 0.4, "type": "dual_system"}
        elif n_alerting == 3:
            return {"score": 0.6, "type": "triple_system"}
        elif n_alerting >= 4:
            # Check if pathways explain the convergence
            if detected_pathways:
                return {"score": 0.9, "type": "multi_system_pathway"}
            else:
                return {"score": 0.8, "type": "multi_system_unknown"}

        return {"score": 0.5, "type": "moderate"}

    def _generate_insights(
        self,
        endpoint_results: Dict[str, Dict],
        alerting_endpoints: Dict[str, Dict],
        detected_pathways: List[Dict],
        unknown_pattern: Optional[Dict]
    ) -> List[str]:
        """Generate cross-domain insights."""
        insights = []

        # Check for specific cross-domain patterns
        alerting_names = set(alerting_endpoints.keys())

        # Cardiorenal
        if {"cardiac", "renal"}.issubset(alerting_names):
            insights.append("Cardiorenal interaction detected - heart and kidney dysfunction linked")

        # Hepatorenal
        if {"hepatic", "renal"}.issubset(alerting_names):
            insights.append("Hepatorenal pattern - liver dysfunction affecting kidney function")

        # Sepsis signature
        if {"inflammatory", "hemodynamic", "metabolic"}.issubset(alerting_names):
            insights.append("Sepsis-like pattern - inflammation with hemodynamic and metabolic impact")

        # Metabolic syndrome
        if {"metabolic", "lipid_atherogenic", "endocrine"}.issubset(alerting_names):
            insights.append("Metabolic syndrome pattern - glucose, lipid, and hormone dysregulation")

        # Multi-organ failure
        if len(alerting_names) >= 4:
            categories_affected = set()
            for ep in alerting_names:
                for cat, eps in self.categories.items():
                    if ep in eps:
                        categories_affected.add(cat)
            if len(categories_affected) >= 3:
                insights.append(f"Multi-organ involvement across {len(categories_affected)} system categories")

        # Add pathway-specific insights
        for pathway in detected_pathways[:3]:
            insights.append(f"Pathway match: {pathway['pathway_name']} ({pathway['urgency']} urgency)")

        # Unknown pattern insight
        if unknown_pattern:
            insights.append(f"Novel pattern: {len(unknown_pattern.get('unexplained_endpoints', []))} unexplained endpoints")

        return insights

    def _summarize_by_category(
        self,
        endpoint_results: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Summarize results by endpoint category."""
        summary = {}

        for category, endpoints in self.categories.items():
            category_results = {
                ep: endpoint_results.get(ep, {})
                for ep in endpoints
            }

            n_alerting = sum(
                1 for r in category_results.values()
                if r.get("status") in ["elevated", "critical"]
            )

            avg_score = np.mean([
                r.get("score", 0) for r in category_results.values()
                if r.get("has_data")
            ]) if any(r.get("has_data") for r in category_results.values()) else 0

            max_score = max(
                (r.get("score", 0) for r in category_results.values()),
                default=0
            )

            summary[category] = {
                "n_endpoints": len(endpoints),
                "n_alerting": n_alerting,
                "avg_score": round(float(avg_score), 3),
                "max_score": round(float(max_score), 3),
                "status": "critical" if max_score > 0.8 else "elevated" if max_score > 0.5 else "normal",
            }

        return summary

    def _determine_urgency(
        self,
        detected_pathways: List[Dict],
        alerting_endpoints: Dict[str, Dict]
    ) -> str:
        """Determine overall urgency level."""
        # Check pathway urgencies
        for pathway in detected_pathways:
            if pathway.get("urgency") == "critical":
                return "critical"

        # Check endpoint statuses
        critical_count = sum(
            1 for r in alerting_endpoints.values()
            if r.get("status") == "critical"
        )

        if critical_count >= 2:
            return "critical"
        elif critical_count == 1 or len(alerting_endpoints) >= 4:
            return "high"
        elif len(alerting_endpoints) >= 2:
            return "moderate"
        elif len(alerting_endpoints) == 1:
            return "low"
        else:
            return "none"

    def _generate_recommendations(
        self,
        detected_pathways: List[Dict],
        unknown_pattern: Optional[Dict],
        alerting_endpoints: Dict[str, Dict]
    ) -> List[str]:
        """Generate clinical recommendations."""
        recommendations = []

        # Add pathway-specific recommendations
        for pathway in detected_pathways[:3]:
            action = pathway.get("recommended_action", "")
            if action:
                recommendations.append(action)

        # Add unknown pattern recommendation
        if unknown_pattern:
            recommendations.append(unknown_pattern.get("recommendation", "Comprehensive workup recommended"))

        # Add general recommendations based on alerting patterns
        if len(alerting_endpoints) >= 4:
            recommendations.append("Multi-system involvement: Consider ICU evaluation and multi-specialty consultation")

        # Deduplicate
        seen = set()
        unique = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique.append(r)

        return unique[:5]  # Top 5 recommendations

    def _calculate_confidence(
        self,
        endpoint_results: Dict[str, Dict],
        detected_pathways: List[Dict]
    ) -> float:
        """Calculate overall confidence in the analysis."""
        scores = []

        # Data availability (40%)
        data_score = self._calculate_data_completeness(endpoint_results)
        scores.append(data_score * 0.4)

        # Pathway match confidence (30%)
        if detected_pathways:
            pathway_conf = max(p.get("confidence", 0) for p in detected_pathways)
            scores.append(pathway_conf * 0.3)
        else:
            scores.append(0.1)

        # Consistency (30%)
        # Check if endpoint results are internally consistent
        consistency_score = self._calculate_consistency(endpoint_results)
        scores.append(consistency_score * 0.3)

        return round(sum(scores), 3)

    def _calculate_data_completeness(
        self,
        endpoint_results: Dict[str, Dict]
    ) -> float:
        """Calculate data completeness across endpoints."""
        total_biomarkers = 0
        available_biomarkers = 0

        for result in endpoint_results.values():
            total_biomarkers += result.get("biomarkers_total", 0)
            available_biomarkers += result.get("biomarkers_available", 0)

        return available_biomarkers / total_biomarkers if total_biomarkers > 0 else 0

    def _calculate_consistency(
        self,
        endpoint_results: Dict[str, Dict]
    ) -> float:
        """Check internal consistency of results."""
        # Higher consistency when related endpoints agree
        scores = []

        # Check cardiac-hemodynamic consistency
        cardiac = endpoint_results.get("cardiac", {})
        hemodynamic = endpoint_results.get("hemodynamic", {})
        if cardiac.get("has_data") and hemodynamic.get("has_data"):
            diff = abs(cardiac.get("score", 0) - hemodynamic.get("score", 0))
            scores.append(1 - min(diff, 1))

        # Check renal-metabolic consistency
        renal = endpoint_results.get("renal", {})
        metabolic = endpoint_results.get("metabolic", {})
        if renal.get("has_data") and metabolic.get("has_data"):
            diff = abs(renal.get("score", 0) - metabolic.get("score", 0))
            scores.append(1 - min(diff, 1))

        # Check inflammatory-infectious consistency
        inflammatory = endpoint_results.get("inflammatory", {})
        infectious = endpoint_results.get("infectious_pathogenic", {})
        if inflammatory.get("has_data") and infectious.get("has_data"):
            diff = abs(inflammatory.get("score", 0) - infectious.get("score", 0))
            scores.append(1 - min(diff, 1))

        return np.mean(scores) if scores else 0.7


def get_cross_loop_engine() -> CrossLoopEngineV2:
    """Get singleton instance of cross-loop engine."""
    return CrossLoopEngineV2()
