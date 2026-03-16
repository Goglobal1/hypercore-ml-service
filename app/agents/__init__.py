"""
HyperCore Diagnostic Agents

Four specialized agents for clinical intelligence:
1. BiomarkerAgent - Multi-omic signal interpretation
2. DiagnosticAgent - Differential reasoning engine
3. TrialRescueAgent - Pharma trial intelligence
4. SurveillanceAgent - Population anomaly detection

All agents inherit from BaseAgent and can communicate via message protocol.
"""

from app.agents.base_agent import (
    BaseAgent,
    AgentMessage,
    AgentFinding,
    ConfidenceLevel,
    AgentRegistry,
)
from app.agents.biomarker_agent import BiomarkerAgent
from app.agents.diagnostic_agent import DiagnosticAgent
from app.agents.trial_rescue_agent import TrialRescueAgent
from app.agents.surveillance_agent import SurveillanceAgent

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "AgentFinding",
    "ConfidenceLevel",
    "AgentRegistry",
    "BiomarkerAgent",
    "DiagnosticAgent",
    "TrialRescueAgent",
    "SurveillanceAgent",
]
