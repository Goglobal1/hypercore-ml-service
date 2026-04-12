"""
Pharma Mode - Trial Rescue Engine
=================================

Provides trial rescue capabilities for pharmaceutical R&D.

Finds hidden value in "failed" clinical trials through:
1. Subgroup Discovery - Find hidden responder populations
2. Confounder Detection - Identify masking variables
3. Endpoint Reinterpretation - Find alternative endpoints
4. Rescue Strategy Generation - Rank by UTILITY not p-value
5. Asset Valuation - Calculate potential value with escalation

Reference: HyperCore Implementation Guide - Appendix E, Section E.7
"""

from .trial_rescue_engine import (
    TrialRescueEngine,
    TrialRescueInput,
    TrialRescueResult,
    get_trial_rescue_engine,
)

from .subgroup_discovery import (
    SubgroupDiscovery,
    ResponderSubgroup,
)

from .confounder_detector import (
    ConfounderDetector,
    Confounder,
    detect_confounders_in_trial,
)

from .endpoint_reinterpreter import (
    EndpointReinterpreter,
    AlternativeEndpoint,
)

from .rescue_report_builder import (
    RescueReportBuilder,
    RescueOpportunity,
    TrialRescueReport,
)

__all__ = [
    # Main engine
    "TrialRescueEngine",
    "TrialRescueInput",
    "TrialRescueResult",
    "get_trial_rescue_engine",
    # Subgroup discovery
    "SubgroupDiscovery",
    "ResponderSubgroup",
    # Confounder detection
    "ConfounderDetector",
    "Confounder",
    "detect_confounders_in_trial",
    # Endpoint reinterpretation
    "EndpointReinterpreter",
    "AlternativeEndpoint",
    # Report building
    "RescueReportBuilder",
    "RescueOpportunity",
    "TrialRescueReport",
]
