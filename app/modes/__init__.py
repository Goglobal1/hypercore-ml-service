"""
HyperCore Modes
===============

Deployment modes for different use cases:
- Pharma: Clinical trial rescue and drug development support
- (Future) Hospital: Clinical decision support
- (Future) Government: Public health surveillance
"""

from .pharma import (
    TrialRescueEngine,
    TrialRescueInput,
    TrialRescueResult,
    get_trial_rescue_engine,
)

__all__ = [
    "TrialRescueEngine",
    "TrialRescueInput",
    "TrialRescueResult",
    "get_trial_rescue_engine",
]
