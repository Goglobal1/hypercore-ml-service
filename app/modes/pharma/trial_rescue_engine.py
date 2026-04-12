"""
Trial Rescue Engine - Pharma Mode
=================================

Main orchestrator for the 5-step trial rescue workflow.
Finds hidden value in "failed" clinical trials.

Reference: HyperCore Implementation Guide - Appendix E, Section E.7

CRITICAL RULES:
- Rank rescue opportunities by handler_score + net_utility + asset_value
- NOT by p-value alone
- Use Utility Gate with DeploymentMode.PHARMA
- Each rescue opportunity goes through Utility Gate before surfacing

The 5 Steps:
1. Subgroup Discovery - Find hidden responder subgroups
2. Confounder Detection - Identify masking variables
3. Endpoint Reinterpretation - Find alternative endpoints showing effect
4. Rescue Strategy Generation - Rank by utility, not p-value
5. Report Building - Calculate asset value and generate report
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import io

import numpy as np
import pandas as pd

from .subgroup_discovery import SubgroupDiscovery, ResponderSubgroup
from .confounder_detector import ConfounderDetector, Confounder
from .endpoint_reinterpreter import EndpointReinterpreter, AlternativeEndpoint
from .rescue_report_builder import RescueReportBuilder, RescueOpportunity, TrialRescueReport

# Import Utility Gate for PHARMA mode policy enforcement
try:
    from app.core.utility_engine.utility_gate import UtilityGate
    from app.core.utility_engine.schemas import (
        DeploymentMode, UtilityInput, UtilityDecision,
        DecisionAction, EvidenceItem,
    )
    UTILITY_GATE_AVAILABLE = True
except ImportError:
    UTILITY_GATE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrialRescueInput:
    """Input for trial rescue analysis."""
    trial_data: pd.DataFrame
    treatment_column: str = "treatment"
    outcome_column: str = "outcome"
    patient_id_column: str = "patient_id"
    covariate_columns: Optional[List[str]] = None
    trial_name: Optional[str] = None
    sponsor: Optional[str] = None
    phase: Optional[str] = None
    indication: Optional[str] = None
    original_pvalue: Optional[float] = None
    base_asset_value_usd: float = 100_000_000  # $100M base


@dataclass
class TrialRescueResult:
    """Complete result from trial rescue analysis."""
    trial_name: str
    timestamp: str
    success: bool

    # Step 1: Subgroups
    subgroups_found: int
    subgroups: List[ResponderSubgroup]

    # Step 2: Confounders
    confounders_found: int
    confounders: List[Confounder]

    # Step 3: Alternative endpoints
    alternative_endpoints_found: int
    alternative_endpoints: List[AlternativeEndpoint]

    # Step 4-5: Rescue opportunities (utility-ranked)
    rescue_opportunities: List[RescueOpportunity]
    surfaced_opportunities: List[RescueOpportunity]
    suppressed_opportunities: List[RescueOpportunity]

    # Final report
    report: Optional[TrialRescueReport] = None

    # Escalation flag
    hard_escalation_triggered: bool = False
    estimated_total_asset_value_usd: float = 0.0

    # Metadata
    engine_version: str = "1.0.0"
    utility_gate_mode: str = "PHARMA"
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrialRescueEngine:
    """
    Main orchestrator for trial rescue workflow.

    Implements the 5-step process to find hidden value in failed trials:
    1. Subgroup Discovery - Find patient clusters with better response
    2. Confounder Detection - Find variables masking treatment effect
    3. Endpoint Reinterpretation - Find endpoints where drug works
    4. Rescue Strategy Generation - Rank by UTILITY not p-value
    5. Report Building - Calculate asset value, flag escalations
    """

    BILLION_DOLLAR_THRESHOLD = 1_000_000_000  # $1B triggers hard escalation

    def __init__(self):
        """Initialize the Trial Rescue Engine with all components."""
        self.subgroup_discovery = SubgroupDiscovery()
        self.confounder_detector = ConfounderDetector()
        self.endpoint_reinterpreter = EndpointReinterpreter()
        self.report_builder = RescueReportBuilder()

        # Initialize Utility Gate in PHARMA mode
        if UTILITY_GATE_AVAILABLE:
            self.utility_gate = UtilityGate(DeploymentMode.PHARMA)
        else:
            self.utility_gate = None
            logger.warning("Utility Gate not available - rescue opportunities will not be filtered")

        self.version = "1.0.1"  # Added error handling for clustering failures

    def analyze(self, input_data: TrialRescueInput) -> TrialRescueResult:
        """
        Run the complete 5-step trial rescue analysis.

        Args:
            input_data: TrialRescueInput with trial data and configuration

        Returns:
            TrialRescueResult with all findings and rescue opportunities
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        trial_name = input_data.trial_name or "Unknown Trial"

        logger.info(f"Starting trial rescue analysis for: {trial_name}")

        try:
            # Prepare data
            df = input_data.trial_data.copy()
            treatment_col = input_data.treatment_column
            outcome_col = input_data.outcome_column
            patient_id_col = input_data.patient_id_column

            # Identify covariate columns (all except treatment, outcome, patient_id)
            covariate_cols = input_data.covariate_columns
            if covariate_cols is None:
                covariate_cols = [
                    c for c in df.columns
                    if c not in [treatment_col, outcome_col, patient_id_col]
                ]

            # ===============================================
            # STEP 1: Subgroup Discovery
            # ===============================================
            logger.info("Step 1: Running subgroup discovery...")
            subgroups = self.subgroup_discovery.find_subgroups(
                df=df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                covariate_cols=covariate_cols,
            )
            logger.info(f"Found {len(subgroups)} potential responder subgroups")

            # ===============================================
            # STEP 2: Confounder Detection
            # ===============================================
            logger.info("Step 2: Running confounder detection...")
            confounders = self.confounder_detector.detect_confounders(
                df=df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                covariate_cols=covariate_cols,
            )
            logger.info(f"Found {len(confounders)} potential confounders")

            # ===============================================
            # STEP 3: Endpoint Reinterpretation
            # ===============================================
            logger.info("Step 3: Running endpoint reinterpretation...")
            alternative_endpoints = self.endpoint_reinterpreter.find_alternatives(
                df=df,
                treatment_col=treatment_col,
                primary_outcome_col=outcome_col,
            )
            logger.info(f"Found {len(alternative_endpoints)} alternative endpoints")

            # ===============================================
            # STEP 4: Generate Rescue Opportunities
            # ===============================================
            logger.info("Step 4: Generating rescue opportunities...")
            rescue_opportunities = self.report_builder.generate_opportunities(
                subgroups=subgroups,
                confounders=confounders,
                alternative_endpoints=alternative_endpoints,
                base_asset_value=input_data.base_asset_value_usd,
                trial_name=trial_name,
            )
            logger.info(f"Generated {len(rescue_opportunities)} rescue opportunities")

            # ===============================================
            # STEP 5: Utility Gate Filtering & Report Building
            # ===============================================
            logger.info("Step 5: Applying Utility Gate and building report...")

            surfaced = []
            suppressed = []
            hard_escalation = False

            for opp in rescue_opportunities:
                # Check hard escalation threshold
                if opp.estimated_asset_value_usd >= self.BILLION_DOLLAR_THRESHOLD:
                    hard_escalation = True
                    opp.hard_escalation_flag = True

                # Apply Utility Gate
                if self.utility_gate:
                    decision = self._apply_utility_gate(opp)
                    opp.utility_decision = decision.action.value
                    opp.utility_breakdown = {
                        "handler_score": decision.breakdown.handler_score,
                        "net_utility": decision.breakdown.net_utility,
                        "rightness": decision.breakdown.rightness,
                        "novelty": decision.breakdown.novelty,
                        "convincing": decision.breakdown.convincing,
                    }

                    if decision.should_surface:
                        surfaced.append(opp)
                    else:
                        suppressed.append(opp)
                else:
                    # No utility gate - surface all
                    surfaced.append(opp)

            # Sort surfaced opportunities by combined score (NOT p-value!)
            surfaced.sort(
                key=lambda x: (
                    x.utility_breakdown.get("handler_score", 0) +
                    x.utility_breakdown.get("net_utility", 0) +
                    x.estimated_asset_value_usd / 1e9  # Normalize to billions
                ),
                reverse=True
            )

            # Calculate total estimated asset value
            total_asset_value = sum(o.estimated_asset_value_usd for o in surfaced)

            # Build final report
            report = self.report_builder.build_report(
                trial_name=trial_name,
                surfaced_opportunities=surfaced,
                subgroups=subgroups,
                confounders=confounders,
                alternative_endpoints=alternative_endpoints,
                total_asset_value=total_asset_value,
                sponsor=input_data.sponsor,
                phase=input_data.phase,
                indication=input_data.indication,
            )

            logger.info(f"Trial rescue analysis complete. Surfaced: {len(surfaced)}, Suppressed: {len(suppressed)}")

            return TrialRescueResult(
                trial_name=trial_name,
                timestamp=timestamp,
                success=True,
                subgroups_found=len(subgroups),
                subgroups=subgroups,
                confounders_found=len(confounders),
                confounders=confounders,
                alternative_endpoints_found=len(alternative_endpoints),
                alternative_endpoints=alternative_endpoints,
                rescue_opportunities=rescue_opportunities,
                surfaced_opportunities=surfaced,
                suppressed_opportunities=suppressed,
                report=report,
                hard_escalation_triggered=hard_escalation,
                estimated_total_asset_value_usd=total_asset_value,
                engine_version=self.version,
                utility_gate_mode="PHARMA",
                metadata={
                    "treatment_column": treatment_col,
                    "outcome_column": outcome_col,
                    "n_patients": len(df),
                    "n_covariates": len(covariate_cols),
                    "original_pvalue": input_data.original_pvalue,
                },
            )

        except Exception as e:
            logger.error(f"Trial rescue analysis failed: {e}", exc_info=True)
            return TrialRescueResult(
                trial_name=trial_name,
                timestamp=timestamp,
                success=False,
                subgroups_found=0,
                subgroups=[],
                confounders_found=0,
                confounders=[],
                alternative_endpoints_found=0,
                alternative_endpoints=[],
                rescue_opportunities=[],
                surfaced_opportunities=[],
                suppressed_opportunities=[],
                report=None,
                hard_escalation_triggered=False,
                estimated_total_asset_value_usd=0.0,
                engine_version=self.version,
                utility_gate_mode="PHARMA",
                metadata={"error": str(e)},
            )

    def _apply_utility_gate(self, opportunity: RescueOpportunity) -> UtilityDecision:
        """
        Apply Utility Gate to a rescue opportunity.

        Args:
            opportunity: RescueOpportunity to evaluate

        Returns:
            UtilityDecision with action and breakdown
        """
        # Build evidence items
        evidence = [
            EvidenceItem(
                kind="subgroup",
                label=f"Subgroup: {opportunity.subgroup_name or 'N/A'}",
                value=opportunity.subgroup_response_rate,
                weight=0.8,
            ),
            EvidenceItem(
                kind="statistical",
                label=f"Effect size: {opportunity.effect_size:.3f}",
                value=opportunity.effect_size,
                weight=0.7,
            ),
            EvidenceItem(
                kind="financial",
                label=f"Asset value: ${opportunity.estimated_asset_value_usd:,.0f}",
                value=opportunity.estimated_asset_value_usd,
                weight=0.6,
            ),
        ]

        # Create UtilityInput
        utility_input = UtilityInput(
            entity_id=opportunity.opportunity_id,
            entity_type="trial_opportunity",
            mode=DeploymentMode.PHARMA,
            title=opportunity.title,
            summary=opportunity.description,
            risk_probability=None,  # Not applicable for trial rescue
            severity=opportunity.effect_size,  # Use effect size as proxy
            calibration_score=opportunity.plausibility_score,
            ppv_estimate=opportunity.confidence,
            lead_time_hours=None,
            novelty_score=0.7 if opportunity.opportunity_type == "subgroup" else 0.5,
            explainability_score=opportunity.plausibility_score,
            actionability_score=0.8,  # Pharma trials are highly actionable
            confidence_score=opportunity.confidence,
            evidence=evidence,
            metadata={
                "hard_escalation_flag": opportunity.hard_escalation_flag,
                "asset_value_usd": opportunity.estimated_asset_value_usd,
                "opportunity_type": opportunity.opportunity_type,
            },
        )

        return self.utility_gate.evaluate(utility_input)

    def analyze_from_csv(
        self,
        csv_data: str,
        treatment_column: str = "treatment",
        outcome_column: str = "outcome",
        patient_id_column: str = "patient_id",
        trial_name: Optional[str] = None,
        **kwargs
    ) -> TrialRescueResult:
        """
        Convenience method to analyze from CSV string.

        Args:
            csv_data: CSV string with trial data
            treatment_column: Name of treatment column
            outcome_column: Name of outcome column
            patient_id_column: Name of patient ID column
            trial_name: Optional trial name
            **kwargs: Additional arguments for TrialRescueInput

        Returns:
            TrialRescueResult
        """
        df = pd.read_csv(io.StringIO(csv_data))

        input_data = TrialRescueInput(
            trial_data=df,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            patient_id_column=patient_id_column,
            trial_name=trial_name,
            **kwargs
        )

        return self.analyze(input_data)


# Singleton instance
_engine_instance: Optional[TrialRescueEngine] = None


def get_trial_rescue_engine() -> TrialRescueEngine:
    """Get or create the singleton Trial Rescue Engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TrialRescueEngine()
    return _engine_instance
