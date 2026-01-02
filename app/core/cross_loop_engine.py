# cross_loop_engine.py
# Location: app/core/cross_loop_engine.py
"""
HyperCore Cross-Loop Engine - Meta-analysis across all endpoints.
"""

from typing import Dict, Any, List, Set
from datetime import datetime


class CrossLoopEngine:
    """Meta-analysis engine that cross-references all endpoint results."""

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.original_data: Any = None
        self.execution_timestamp: str = ""

    def ingest_results(self, endpoint_results: Dict[str, Any], original_data: Any = None):
        """Store all endpoint results for cross-referencing."""
        self.results = endpoint_results
        self.original_data = original_data
        self.execution_timestamp = datetime.utcnow().isoformat()

    def execute_cross_loop(self) -> Dict[str, Any]:
        """Run all cross-reference analyses."""
        return {
            "cross_loop_id": f"cl_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "timestamp": self.execution_timestamp,
            "endpoints_analyzed": list(self.results.keys()),
            "cross_validated_findings": self._cross_validate_findings(),
            "emergent_patterns": self._find_emergent_patterns(),
            "contradictions": self._find_contradictions(),
            "coverage_gaps": self._identify_gaps(),
            "confidence_assessment": self._assess_confidence(),
            "super_insights": self._generate_super_insights(),
            "executive_summary": self._generate_executive_summary(),
            "recommended_actions": self._generate_recommendations(),
            "alerts": self._generate_alerts()
        }

    def _cross_validate_findings(self) -> List[Dict]:
        """Find findings that appear in multiple endpoints."""
        validated = []
        endpoint_variables = {}

        if "responder_prediction" in self.results:
            resp = self.results["responder_prediction"]
            biomarkers = resp.get("key_biomarkers", {})
            endpoint_variables["responder_prediction"] = set(biomarkers.keys())

        if "confounder_detection" in self.results:
            conf = self.results["confounder_detection"]
            if isinstance(conf, list):
                vars_found = {c.get("variable") for c in conf if c.get("variable")}
            else:
                vars_found = set()
            endpoint_variables["confounder_detection"] = vars_found

        if "analyze" in self.results:
            analyze = self.results["analyze"]
            features = analyze.get("feature_importance", [])
            if isinstance(features, list):
                top_features = {f.get("feature", "").split("__")[-1] for f in features[:10]}
            else:
                top_features = set()
            endpoint_variables["analyze"] = top_features

        all_endpoints = list(endpoint_variables.keys())
        for i, ep1 in enumerate(all_endpoints):
            for ep2 in all_endpoints[i+1:]:
                vars1 = endpoint_variables.get(ep1, set())
                vars2 = endpoint_variables.get(ep2, set())
                overlap = vars1 & vars2
                for var in overlap:
                    if var and var not in ["", "None", None]:
                        validated.append({
                            "finding": f"Variable '{var}' identified as significant",
                            "sources": [ep1, ep2],
                            "confidence": "HIGH",
                            "interpretation": f"'{var}' was independently identified by multiple analyses"
                        })

        return validated

    def _find_emergent_patterns(self) -> List[Dict]:
        """Find patterns only visible when combining results."""
        patterns = []

        outbreak = self.results.get("outbreak_detection", {})
        early_risk = self.results.get("early_risk_discovery", {})
        if outbreak.get("outbreak_regions") and early_risk.get("clinical_impact"):
            patterns.append({
                "pattern": "Geographic-Clinical Correlation",
                "description": "Outbreak regions correlate with early risk signals",
                "insight": "Geographic clustering may explain patient deterioration",
                "action": "Focus surveillance on affected regions",
                "confidence": "MEDIUM"
            })

        multi_omic = self.results.get("multi_omic_fusion", {})
        responder = self.results.get("responder_prediction", {})
        if multi_omic.get("primary_driver") and responder.get("key_biomarkers"):
            driver = multi_omic["primary_driver"]
            patterns.append({
                "pattern": "Multi-Omic Response Correlation",
                "description": f"Primary omic driver '{driver}' correlates with response biomarkers",
                "insight": f"The {driver} axis may explain treatment response",
                "action": f"Consider {driver}-based patient stratification",
                "confidence": "HIGH"
            })

        confounders = self.results.get("confounder_detection", [])
        rescue = self.results.get("trial_rescue", {})
        if confounders and rescue.get("rescue_score", 0) > 50:
            patterns.append({
                "pattern": "Confounded Rescue Opportunity",
                "description": "High rescue potential alongside confounding variables",
                "insight": "Rescue subgroups may be artifacts of confounding",
                "action": "Stratify rescue analysis by confounders before acting",
                "confidence": "HIGH"
            })

        return patterns

    def _find_contradictions(self) -> List[Dict]:
        """Find where endpoints disagree."""
        contradictions = []

        rescue = self.results.get("trial_rescue", {})
        confounders = self.results.get("confounder_detection", [])
        rescue_score = rescue.get("rescue_score", 0)
        n_confounders = len(confounders) if isinstance(confounders, list) else 0

        if rescue_score > 70 and n_confounders > 2:
            contradictions.append({
                "type": "rescue_vs_confounder",
                "severity": "HIGH",
                "concern": f"High rescue score ({rescue_score}%) but {n_confounders} confounders detected",
                "interpretation": "Rescue opportunity may be driven by confounding",
                "recommendation": "Stratify analysis by confounders before proceeding"
            })

        return contradictions

    def _identify_gaps(self) -> List[Dict]:
        """Identify coverage gaps."""
        gaps = []
        core_endpoints = {"analyze": "Core analysis", "trial_rescue": "Trial rescue", "responder_prediction": "Response prediction", "confounder_detection": "Confounder detection", "early_risk_discovery": "Early risk detection"}

        for endpoint, description in core_endpoints.items():
            if endpoint not in self.results:
                gaps.append({"gap_type": "missing_endpoint", "endpoint": endpoint, "description": description, "impact": f"Missing {description.lower()}", "recommendation": f"Run /{endpoint} for complete analysis"})
            elif not self.results[endpoint]:
                gaps.append({"gap_type": "empty_result", "endpoint": endpoint, "description": description, "impact": f"{endpoint} returned empty", "recommendation": "Check data format and re-run"})

        return gaps

    def _assess_confidence(self) -> Dict:
        """Calculate overall confidence."""
        base_confidence = 0.5
        n_endpoints = len([r for r in self.results.values() if r])
        endpoint_boost = min(0.2, n_endpoints * 0.02)
        cross_validated = len(self._cross_validate_findings())
        cross_boost = min(0.15, cross_validated * 0.05)
        contradictions = len(self._find_contradictions())
        contradiction_penalty = min(0.2, contradictions * 0.05)
        gaps = len(self._identify_gaps())
        gap_penalty = min(0.15, gaps * 0.03)

        final_confidence = base_confidence + endpoint_boost + cross_boost - contradiction_penalty - gap_penalty
        final_confidence = max(0.1, min(0.95, final_confidence))

        if final_confidence >= 0.8:
            interpretation = "HIGH - Multiple analyses agree"
        elif final_confidence >= 0.6:
            interpretation = "MEDIUM - Reasonable support with some gaps"
        elif final_confidence >= 0.4:
            interpretation = "LOW - Limited support; significant gaps"
        else:
            interpretation = "VERY LOW - Major issues; interpret with caution"

        return {"overall_confidence": round(final_confidence, 2), "interpretation": interpretation}

    def _generate_super_insights(self) -> List[Dict]:
        """Generate high-level insights."""
        insights = []

        rescue = self.results.get("trial_rescue", {})
        if rescue.get("rescue_score", 0) > 70:
            insights.append({"type": "opportunity", "title": "Strong Trial Rescue Potential", "detail": f"Rescue score of {rescue.get('rescue_score')}%", "priority": "HIGH"})

        early = self.results.get("early_risk_discovery", {})
        if early.get("risk_timing_delta", {}).get("detection_window_days", 0) > 3:
            days = early["risk_timing_delta"]["detection_window_days"]
            insights.append({"type": "capability", "title": "Early Detection Advantage", "detail": f"Risk detected {days} days before standard systems", "priority": "HIGH"})

        return insights

    def _generate_executive_summary(self) -> str:
        """Generate executive summary."""
        parts = []
        n_endpoints = len([r for r in self.results.values() if r])
        parts.append(f"Cross-loop analysis completed across {n_endpoints} endpoints.")

        rescue = self.results.get("trial_rescue", {})
        if rescue.get("rescue_score"):
            parts.append(f"Trial rescue potential: {rescue['rescue_score']}%.")

        early = self.results.get("early_risk_discovery", {})
        if early.get("risk_timing_delta", {}).get("detection_window_days"):
            days = early["risk_timing_delta"]["detection_window_days"]
            parts.append(f"Early detection: {days} days advantage.")

        confidence = self._assess_confidence()
        parts.append(f"Overall confidence: {confidence['overall_confidence']:.0%}.")

        return " ".join(parts)

    def _generate_recommendations(self) -> List[Dict]:
        """Generate recommendations."""
        recommendations = []

        rescue = self.results.get("trial_rescue", {})
        if rescue.get("rescue_score", 0) > 50:
            recommendations.append({"priority": 1, "action": "Pursue subgroup analysis for trial rescue", "rationale": f"Rescue score of {rescue.get('rescue_score')}%"})

        confounders = self.results.get("confounder_detection", [])
        if confounders:
            recommendations.append({"priority": 2, "action": "Address confounding before final analysis", "rationale": f"{len(confounders)} confounders detected"})

        recommendations.sort(key=lambda x: x.get("priority", 99))
        return recommendations

    def _generate_alerts(self) -> List[Dict]:
        """Generate alerts."""
        alerts = []

        for contradiction in self._find_contradictions():
            if contradiction.get("severity") == "HIGH":
                alerts.append({"level": "WARNING", "type": contradiction["type"], "message": contradiction["concern"], "action_required": contradiction["recommendation"]})

        for insight in self._generate_super_insights():
            if insight.get("priority") == "HIGH" and insight.get("type") == "opportunity":
                alerts.append({"level": "OPPORTUNITY", "type": "actionable_finding", "message": insight["title"], "action_required": insight["detail"]})

        return alerts


def run_cross_loop_analysis(endpoint_results: Dict[str, Any], original_data: Any = None) -> Dict[str, Any]:
    """Convenience function to run cross-loop analysis."""
    engine = CrossLoopEngine()
    engine.ingest_results(endpoint_results, original_data)
    return engine.execute_cross_loop()
