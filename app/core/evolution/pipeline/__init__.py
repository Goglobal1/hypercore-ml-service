"""
Evolution Pipeline Module
=========================

Orchestrates the evolution loop:
1. Sample parent nodes (UCB1 + utility-weighted)
2. Generate hypothesis (Researcher agent)
3. Evaluate candidate (Evaluator agent + safety checks)
4. Analyze results (Analyzer agent)
5. Route through approval workflow
"""

from .orchestrator import (
    EvolutionOrchestrator,
    EvolutionStep,
    OrchestratorConfig,
    create_orchestrator,
)
from .agents import (
    ResearcherAgent,
    ResearcherConfig,
    EvaluatorAgent,
    EvaluatorConfig,
    AnalyzerAgent,
    AnalyzerConfig,
)

__all__ = [
    # Orchestrator
    "EvolutionOrchestrator",
    "EvolutionStep",
    "OrchestratorConfig",
    "create_orchestrator",
    # Agents
    "ResearcherAgent",
    "ResearcherConfig",
    "EvaluatorAgent",
    "EvaluatorConfig",
    "AnalyzerAgent",
    "AnalyzerConfig",
]
