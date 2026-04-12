"""
Rescue Report Builder - Steps 4-5 of Trial Rescue
==================================================

Generate rescue strategies and reports with asset value calculation.

CRITICAL: Ranks opportunities by UTILITY (handler_score + net_utility + asset_value)
NOT by p-value alone!

Asset Value Formula:
    asset_value = base_value * subgroup_mult * (1 + efficacy_mult) * plausibility_mult * confounder_penalty

Hard Escalation: Triggered when estimated_asset_value_usd >= $1,000,000,000

Reference: HyperCore Implementation Guide - Appendix E, Section E.7.4-5
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import uuid

from .subgroup_discovery import ResponderSubgroup
from .confounder_detector import Confounder
from .endpoint_reinterpreter import AlternativeEndpoint

logger = logging.getLogger(__name__)


@dataclass
class RescueOpportunity:
    """A ranked rescue opportunity from trial analysis."""
    opportunity_id: str
    opportunity_type: str  # "subgroup", "confounder_adjustment", "alternative_endpoint", "composite"
    title: str
    description: str

    # Core metrics (for ranking)
    confidence: float  # 0-1
    effect_size: float
    plausibility_score: float  # 0-1

    # Utility metrics (set by Utility Gate)
    handler_score: float = 0.0
    net_utility: float = 0.0

    # Asset value calculation
    estimated_asset_value_usd: float = 0.0
    asset_value_breakdown: Dict[str, float] = field(default_factory=dict)

    # Escalation
    hard_escalation_flag: bool = False

    # Utility Gate decision (filled in by engine)
    utility_decision: Optional[str] = None
    utility_breakdown: Dict[str, float] = field(default_factory=dict)

    # Source references
    subgroup_name: Optional[str] = None
    subgroup_response_rate: Optional[float] = None
    confounder_name: Optional[str] = None
    endpoint_name: Optional[str] = None

    # Actionable recommendations
    recommended_actions: List[str] = field(default_factory=list)
    regulatory_pathway: Optional[str] = None
    timeline_estimate: Optional[str] = None

    # Evidence
    evidence: List[str] = field(default_factory=list)
    pvalue: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialRescueReport:
    """Complete trial rescue report."""
    report_id: str
    trial_name: str
    generated_at: str

    # Summary
    executive_summary: str
    total_opportunities: int
    surfaced_opportunities: int
    suppressed_opportunities: int

    # Asset valuation
    total_estimated_value_usd: float
    hard_escalation_triggered: bool

    # Top opportunities
    top_opportunities: List[RescueOpportunity]

    # Detailed findings
    subgroup_summary: Dict[str, Any]
    confounder_summary: Dict[str, Any]
    endpoint_summary: Dict[str, Any]

    # Recommendations
    primary_recommendation: str
    secondary_recommendations: List[str]
    regulatory_considerations: List[str]

    # Metadata
    sponsor: Optional[str] = None
    phase: Optional[str] = None
    indication: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RescueReportBuilder:
    """
    Builds rescue opportunities and reports.

    CRITICAL: Ranks by UTILITY not p-value!

    Asset Value Formula:
        asset_value = base_value * subgroup_mult * (1 + efficacy_mult) * plausibility_mult * confounder_penalty

    Multipliers:
        - subgroup_mult: Based on subgroup size (larger = higher value)
        - efficacy_mult: Based on effect size (larger effect = higher value)
        - plausibility_mult: Based on biological/clinical plausibility
        - confounder_penalty: Reduces value if confounders are severe
    """

    # Asset value multiplier ranges
    SUBGROUP_SIZE_MULTIPLIERS = {
        (0, 0.05): 0.3,    # <5% of patients: low value
        (0.05, 0.15): 0.6,  # 5-15%: moderate
        (0.15, 0.30): 0.8,  # 15-30%: good
        (0.30, 1.0): 1.0,   # >30%: high value
    }

    EFFICACY_MULTIPLIERS = {
        (0, 0.2): 0.0,     # Negligible effect
        (0.2, 0.5): 0.2,   # Small effect
        (0.5, 0.8): 0.5,   # Medium effect
        (0.8, 1.2): 0.8,   # Large effect
        (1.2, float('inf')): 1.0,  # Very large effect
    }

    BILLION_THRESHOLD = 1_000_000_000

    def __init__(self):
        """Initialize report builder."""
        pass

    def generate_opportunities(
        self,
        subgroups: List[ResponderSubgroup],
        confounders: List[Confounder],
        alternative_endpoints: List[AlternativeEndpoint],
        base_asset_value: float,
        trial_name: str,
    ) -> List[RescueOpportunity]:
        """
        Generate all rescue opportunities with asset values.

        Args:
            subgroups: Discovered responder subgroups
            confounders: Detected confounders
            alternative_endpoints: Alternative endpoints found
            base_asset_value: Base asset value in USD
            trial_name: Name of the trial

        Returns:
            List of RescueOpportunity objects (NOT yet filtered by Utility Gate)
        """
        opportunities = []

        # 1. Generate subgroup opportunities
        for sg in subgroups:
            try:
                opp = self._create_subgroup_opportunity(sg, base_asset_value, trial_name)
                opportunities.append(opp)
            except Exception as e:
                logger.warning(f"Failed to create subgroup opportunity: {e}")

        # 2. Generate confounder adjustment opportunities
        for conf in confounders:
            try:
                opp = self._create_confounder_opportunity(conf, base_asset_value, trial_name)
                opportunities.append(opp)
            except Exception as e:
                logger.warning(f"Failed to create confounder opportunity: {e}")

        # 3. Generate alternative endpoint opportunities
        for ep in alternative_endpoints:
            try:
                opp = self._create_endpoint_opportunity(ep, base_asset_value, trial_name)
                opportunities.append(opp)
            except Exception as e:
                logger.warning(f"Failed to create endpoint opportunity: {e}")

        # Sort by estimated asset value (before Utility Gate filtering)
        opportunities.sort(key=lambda x: x.estimated_asset_value_usd, reverse=True)

        logger.info(f"Generated {len(opportunities)} rescue opportunities")
        return opportunities

    def _create_subgroup_opportunity(
        self,
        subgroup: ResponderSubgroup,
        base_value: float,
        trial_name: str,
    ) -> RescueOpportunity:
        """Create opportunity from subgroup finding."""
        # Calculate asset value
        asset_value, breakdown = self._calculate_asset_value(
            base_value=base_value,
            subgroup_percentage=subgroup.percentage_of_total / 100,
            effect_size=subgroup.effect_size,
            plausibility=subgroup.biological_plausibility,
            confounder_severity=0.0,  # No confounder penalty for subgroups
        )

        # Determine regulatory pathway
        if subgroup.percentage_of_total >= 20 and subgroup.effect_size >= 0.5:
            regulatory = "Potential label expansion or enrichment strategy"
            timeline = "18-24 months to supplemental approval"
        elif subgroup.percentage_of_total >= 10:
            regulatory = "Orphan drug or precision medicine pathway"
            timeline = "24-36 months"
        else:
            regulatory = "Academic publication, possible Phase 2 re-design"
            timeline = "36+ months"

        return RescueOpportunity(
            opportunity_id=f"subgroup_{subgroup.subgroup_id}_{uuid.uuid4().hex[:8]}",
            opportunity_type="subgroup",
            title=f"Responder Subgroup: {subgroup.subgroup_name}",
            description=subgroup.description,
            confidence=1.0 - subgroup.pvalue,
            effect_size=subgroup.effect_size,
            plausibility_score=subgroup.biological_plausibility,
            estimated_asset_value_usd=asset_value,
            asset_value_breakdown=breakdown,
            hard_escalation_flag=asset_value >= self.BILLION_THRESHOLD,
            subgroup_name=subgroup.subgroup_name,
            subgroup_response_rate=subgroup.response_rate,
            recommended_actions=[
                f"Validate subgroup in independent dataset",
                f"Confirm biological rationale for {subgroup.subgroup_name}",
                f"Design enrichment trial targeting this population",
                f"Engage regulatory affairs for pathway discussion",
            ],
            regulatory_pathway=regulatory,
            timeline_estimate=timeline,
            evidence=[
                f"Response rate: {subgroup.response_rate:.1%}",
                f"Relative improvement: {subgroup.relative_response_improvement:.1%}",
                f"Effect size (Cohen's d): {subgroup.effect_size:.3f}",
                f"P-value: {subgroup.pvalue:.4f}",
                f"N patients: {subgroup.n_patients} ({subgroup.percentage_of_total:.1f}%)",
            ],
            pvalue=subgroup.pvalue,
            metadata={
                "cluster_method": subgroup.cluster_method,
                "defining_features": subgroup.defining_features,
            }
        )

    def _create_confounder_opportunity(
        self,
        confounder: Confounder,
        base_value: float,
        trial_name: str,
    ) -> RescueOpportunity:
        """Create opportunity from confounder finding."""
        # Confounders reveal hidden effect - value depends on effect change
        if confounder.adjusted_effect > confounder.unadjusted_effect:
            # Positive finding: effect is larger after adjustment
            effect_boost = abs(confounder.effect_change)
            plausibility = confounder.confidence * 0.8
        else:
            # Effect decreases - less valuable
            effect_boost = 0.0
            plausibility = confounder.confidence * 0.5

        asset_value, breakdown = self._calculate_asset_value(
            base_value=base_value * 0.5,  # Lower base for confounder adjustments
            subgroup_percentage=1.0,  # Applies to all patients
            effect_size=effect_boost,
            plausibility=plausibility,
            confounder_severity=confounder.impact_score,
        )

        return RescueOpportunity(
            opportunity_id=f"confounder_{confounder.variable_name}_{uuid.uuid4().hex[:8]}",
            opportunity_type="confounder_adjustment",
            title=f"Confounder Adjustment: {confounder.variable_name}",
            description=confounder.recommendation,
            confidence=confounder.confidence,
            effect_size=abs(confounder.effect_change),
            plausibility_score=plausibility,
            estimated_asset_value_usd=asset_value,
            asset_value_breakdown=breakdown,
            hard_escalation_flag=asset_value >= self.BILLION_THRESHOLD,
            confounder_name=confounder.variable_name,
            recommended_actions=[
                f"Re-analyze with {confounder.adjustment_method} adjustment",
                f"Stratify results by {confounder.variable_name}",
                f"Include in FDA statistical analysis plan revision",
                f"Consider prospective stratification in follow-up trial",
            ],
            regulatory_pathway="Statistical re-analysis with pre-specified adjustment",
            timeline_estimate="3-6 months for re-analysis",
            evidence=[
                f"Unadjusted effect: {confounder.unadjusted_effect:.4f}",
                f"Adjusted effect: {confounder.adjusted_effect:.4f}",
                f"Effect change: {confounder.effect_change_percentage:.1f}%",
                f"Confounder type: {confounder.confounder_type}",
                f"Impact score: {confounder.impact_score:.2f}",
            ],
            pvalue=confounder.pvalue,
            metadata={
                "adjustment_method": confounder.adjustment_method,
                "imbalance_ratio": confounder.imbalance_ratio,
            }
        )

    def _create_endpoint_opportunity(
        self,
        endpoint: AlternativeEndpoint,
        base_value: float,
        trial_name: str,
    ) -> RescueOpportunity:
        """Create opportunity from alternative endpoint finding."""
        # Endpoint opportunities - value depends on regulatory acceptability
        modified_base = base_value * endpoint.regulatory_acceptability

        asset_value, breakdown = self._calculate_asset_value(
            base_value=modified_base,
            subgroup_percentage=1.0,  # Applies to all patients
            effect_size=endpoint.effect_size,
            plausibility=endpoint.clinical_relevance,
            confounder_severity=0.0,
        )

        # Determine pathway based on endpoint type
        if endpoint.endpoint_type == "secondary":
            pathway = "FDA Type C meeting to discuss endpoint acceptability"
        elif endpoint.endpoint_type == "composite":
            pathway = "Requires pre-specification; suitable for Phase 3 re-design"
        elif endpoint.endpoint_type == "biomarker":
            pathway = "Accelerated approval pathway with confirmatory requirement"
        else:
            pathway = "Post-hoc analysis; academic value"

        return RescueOpportunity(
            opportunity_id=f"endpoint_{endpoint.endpoint_id}_{uuid.uuid4().hex[:8]}",
            opportunity_type="alternative_endpoint",
            title=f"Alternative Endpoint: {endpoint.endpoint_name}",
            description=endpoint.description,
            confidence=1.0 - endpoint.pvalue,
            effect_size=endpoint.effect_size,
            plausibility_score=endpoint.clinical_relevance,
            estimated_asset_value_usd=asset_value,
            asset_value_breakdown=breakdown,
            hard_escalation_flag=asset_value >= self.BILLION_THRESHOLD,
            endpoint_name=endpoint.endpoint_name,
            recommended_actions=[
                f"Validate {endpoint.endpoint_name} in independent cohort",
                f"Establish clinical meaningfulness of endpoint",
                f"Discuss with FDA via pre-submission meeting",
                f"Consider co-primary or key secondary for future trials",
            ],
            regulatory_pathway=pathway,
            timeline_estimate="12-18 months to regulatory discussion",
            evidence=[
                f"Effect size: {endpoint.effect_size:.3f}",
                f"P-value: {endpoint.pvalue:.4f}",
                f"Treatment effect: {endpoint.treatment_effect:.4f}",
                f"Improvement over primary: {endpoint.improvement_over_primary:.1%}",
                f"Clinical relevance: {endpoint.clinical_relevance:.2f}",
                f"Regulatory acceptability: {endpoint.regulatory_acceptability:.2f}",
            ],
            pvalue=endpoint.pvalue,
            metadata={
                "endpoint_type": endpoint.endpoint_type,
                "components": endpoint.components,
                "power": endpoint.power_estimate,
            }
        )

    def _calculate_asset_value(
        self,
        base_value: float,
        subgroup_percentage: float,
        effect_size: float,
        plausibility: float,
        confounder_severity: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate asset value using the formula.

        Formula:
            asset_value = base_value * subgroup_mult * (1 + efficacy_mult) * plausibility_mult * confounder_penalty

        Args:
            base_value: Base asset value in USD
            subgroup_percentage: Fraction of patients in subgroup (0-1)
            effect_size: Cohen's d or equivalent
            plausibility: Biological/clinical plausibility (0-1)
            confounder_severity: Severity of confounding (0-1)

        Returns:
            Tuple of (asset_value, breakdown_dict)
        """
        # 1. Subgroup multiplier
        subgroup_mult = 0.3  # Default
        for (low, high), mult in self.SUBGROUP_SIZE_MULTIPLIERS.items():
            if low <= subgroup_percentage < high:
                subgroup_mult = mult
                break

        # 2. Efficacy multiplier
        efficacy_mult = 0.0  # Default
        abs_effect = abs(effect_size)
        for (low, high), mult in self.EFFICACY_MULTIPLIERS.items():
            if low <= abs_effect < high:
                efficacy_mult = mult
                break

        # 3. Plausibility multiplier (direct mapping)
        plausibility_mult = max(0.2, min(1.0, plausibility))

        # 4. Confounder penalty (reduces value if confounding is severe)
        confounder_penalty = max(0.5, 1.0 - confounder_severity * 0.5)

        # Calculate final value
        asset_value = (
            base_value *
            subgroup_mult *
            (1 + efficacy_mult) *
            plausibility_mult *
            confounder_penalty
        )

        breakdown = {
            "base_value": base_value,
            "subgroup_mult": subgroup_mult,
            "efficacy_mult": efficacy_mult,
            "plausibility_mult": plausibility_mult,
            "confounder_penalty": confounder_penalty,
            "final_value": asset_value,
        }

        return asset_value, breakdown

    def build_report(
        self,
        trial_name: str,
        surfaced_opportunities: List[RescueOpportunity],
        subgroups: List[ResponderSubgroup],
        confounders: List[Confounder],
        alternative_endpoints: List[AlternativeEndpoint],
        total_asset_value: float,
        sponsor: Optional[str] = None,
        phase: Optional[str] = None,
        indication: Optional[str] = None,
    ) -> TrialRescueReport:
        """
        Build the final rescue report.

        Args:
            trial_name: Name of the trial
            surfaced_opportunities: Utility Gate-approved opportunities
            subgroups: All discovered subgroups
            confounders: All detected confounders
            alternative_endpoints: All alternative endpoints
            total_asset_value: Total estimated value
            sponsor: Trial sponsor
            phase: Trial phase
            indication: Trial indication

        Returns:
            TrialRescueReport
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        hard_escalation = any(o.hard_escalation_flag for o in surfaced_opportunities)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            trial_name=trial_name,
            n_opportunities=len(surfaced_opportunities),
            top_opportunity=surfaced_opportunities[0] if surfaced_opportunities else None,
            total_value=total_asset_value,
            hard_escalation=hard_escalation,
        )

        # Generate primary recommendation
        primary_rec = self._generate_primary_recommendation(surfaced_opportunities)

        # Generate secondary recommendations
        secondary_recs = self._generate_secondary_recommendations(
            subgroups, confounders, alternative_endpoints
        )

        # Regulatory considerations
        regulatory = self._generate_regulatory_considerations(
            surfaced_opportunities, phase
        )

        return TrialRescueReport(
            report_id=f"rescue_{uuid.uuid4().hex}",
            trial_name=trial_name,
            generated_at=timestamp,
            executive_summary=executive_summary,
            total_opportunities=len(surfaced_opportunities),
            surfaced_opportunities=len(surfaced_opportunities),
            suppressed_opportunities=0,  # Filled in by engine
            total_estimated_value_usd=total_asset_value,
            hard_escalation_triggered=hard_escalation,
            top_opportunities=surfaced_opportunities[:5],
            subgroup_summary={
                "total_found": len(subgroups),
                "avg_improvement": sum(s.relative_response_improvement for s in subgroups) / len(subgroups) if subgroups else 0,
                "best_subgroup": subgroups[0].subgroup_name if subgroups else None,
            },
            confounder_summary={
                "total_found": len(confounders),
                "most_impactful": confounders[0].variable_name if confounders else None,
                "max_effect_change": max((c.effect_change_percentage for c in confounders), default=0),
            },
            endpoint_summary={
                "total_found": len(alternative_endpoints),
                "best_endpoint": alternative_endpoints[0].endpoint_name if alternative_endpoints else None,
                "best_effect_size": alternative_endpoints[0].effect_size if alternative_endpoints else 0,
            },
            primary_recommendation=primary_rec,
            secondary_recommendations=secondary_recs,
            regulatory_considerations=regulatory,
            sponsor=sponsor,
            phase=phase,
            indication=indication,
        )

    def _generate_executive_summary(
        self,
        trial_name: str,
        n_opportunities: int,
        top_opportunity: Optional[RescueOpportunity],
        total_value: float,
        hard_escalation: bool,
    ) -> str:
        """Generate executive summary."""
        if hard_escalation:
            urgency = "URGENT: "
        else:
            urgency = ""

        if n_opportunities == 0:
            return f"Trial rescue analysis of {trial_name} found no actionable opportunities."

        top_desc = ""
        if top_opportunity:
            top_desc = (
                f" The leading opportunity is '{top_opportunity.title}' with "
                f"estimated value of ${top_opportunity.estimated_asset_value_usd:,.0f}."
            )

        return (
            f"{urgency}Trial rescue analysis of {trial_name} identified {n_opportunities} "
            f"actionable rescue opportunities with combined estimated value of "
            f"${total_value:,.0f}.{top_desc}"
        )

    def _generate_primary_recommendation(
        self,
        opportunities: List[RescueOpportunity]
    ) -> str:
        """Generate primary recommendation."""
        if not opportunities:
            return "No actionable rescue strategies identified. Consider alternative indications or patient populations."

        top = opportunities[0]

        if top.opportunity_type == "subgroup":
            return (
                f"Pursue enrichment strategy targeting {top.subgroup_name}. "
                f"This subgroup shows {top.effect_size:.2f} effect size with "
                f"estimated value of ${top.estimated_asset_value_usd:,.0f}."
            )

        if top.opportunity_type == "confounder_adjustment":
            return (
                f"Re-analyze trial data controlling for {top.confounder_name}. "
                f"This may reveal a masked treatment effect worth "
                f"${top.estimated_asset_value_usd:,.0f}."
            )

        if top.opportunity_type == "alternative_endpoint":
            return (
                f"Consider alternative endpoint '{top.endpoint_name}' which shows "
                f"significant effect (d={top.effect_size:.2f}). "
                f"Engage FDA for endpoint acceptability discussion."
            )

        return f"Pursue top rescue opportunity: {top.title}"

    def _generate_secondary_recommendations(
        self,
        subgroups: List[ResponderSubgroup],
        confounders: List[Confounder],
        endpoints: List[AlternativeEndpoint],
    ) -> List[str]:
        """Generate secondary recommendations."""
        recs = []

        if len(subgroups) > 1:
            recs.append(
                f"Validate top {min(3, len(subgroups))} subgroups in independent datasets"
            )

        if confounders:
            recs.append(
                f"Conduct sensitivity analyses adjusting for {confounders[0].variable_name}"
            )

        if any(ep.endpoint_type == "biomarker" for ep in endpoints):
            recs.append(
                "Explore biomarker-based accelerated approval pathway"
            )

        if any(ep.endpoint_type == "composite" for ep in endpoints):
            recs.append(
                "Consider composite endpoint for Phase 3 re-design"
            )

        recs.append("Document all findings for regulatory submission package")

        return recs[:5]

    def _generate_regulatory_considerations(
        self,
        opportunities: List[RescueOpportunity],
        phase: Optional[str],
    ) -> List[str]:
        """Generate regulatory considerations."""
        considerations = []

        # Check for subgroup opportunities
        subgroup_opps = [o for o in opportunities if o.opportunity_type == "subgroup"]
        if subgroup_opps:
            considerations.append(
                "Subgroup findings are post-hoc; require prospective validation for label claims"
            )
            considerations.append(
                "Consider FDA guidance on enrichment strategies (2019)"
            )

        # Check for endpoint opportunities
        endpoint_opps = [o for o in opportunities if o.opportunity_type == "alternative_endpoint"]
        if endpoint_opps:
            considerations.append(
                "Alternative endpoints require pre-specification for confirmatory trials"
            )
            if any(o.endpoint_name and "biomarker" in o.endpoint_name.lower() for o in endpoint_opps):
                considerations.append(
                    "Biomarker endpoints may qualify for accelerated approval with confirmatory requirement"
                )

        # Phase-specific considerations
        if phase:
            phase_lower = phase.lower()
            if "2" in phase_lower:
                considerations.append(
                    "Phase 2 findings support adaptive Phase 3 design with enrichment"
                )
            elif "3" in phase_lower:
                considerations.append(
                    "Phase 3 rescue requires careful regulatory strategy - consider Type A meeting"
                )

        considerations.append(
            "All rescue strategies should be reviewed by regulatory affairs and biostatistics"
        )

        return considerations
