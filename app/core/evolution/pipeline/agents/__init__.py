"""
Evolution Agents
================

Healthcare-adapted agents for the evolution pipeline:
- Researcher: Hypothesis generation from prior knowledge
- Evaluator: Clinical trial simulation and validation
- Analyzer: Lesson extraction and knowledge synthesis
"""

from .researcher import ResearcherAgent, ResearcherConfig, HypothesisTemplate
from .evaluator import EvaluatorAgent, EvaluatorConfig, TestCase, TestResult
from .analyzer import AnalyzerAgent, AnalyzerConfig, Lesson, AnalysisResult

__all__ = [
    # Researcher
    "ResearcherAgent",
    "ResearcherConfig",
    "HypothesisTemplate",
    # Evaluator
    "EvaluatorAgent",
    "EvaluatorConfig",
    "TestCase",
    "TestResult",
    # Analyzer
    "AnalyzerAgent",
    "AnalyzerConfig",
    "Lesson",
    "AnalysisResult",
]
