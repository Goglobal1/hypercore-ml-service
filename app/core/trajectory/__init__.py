"""
Trajectory Analysis System - Early Warning Engine

Analyzes biomarker TRAJECTORIES to detect disease onset WEEKS before threshold crossing.

Current system: Detects when procalcitonin > 0.5 (threshold) -> 3 days warning
This system: Detects when procalcitonin STARTS RISING abnormally -> 14-21 days warning
"""

from .rate_analysis import RateOfChangeAnalyzer, RateOfChangeResult
from .inflection_detection import InflectionDetector, InflectionPoint, TrajectoryPhase
from .pattern_library import PatternLibrary, PatternMatch, DiseasePattern
from .forecasting import TrajectoryForecaster, ForecastResult, EarlyWarningEngine, EarlyWarningReport

__all__ = [
    'RateOfChangeAnalyzer', 'RateOfChangeResult',
    'InflectionDetector', 'InflectionPoint', 'TrajectoryPhase',
    'PatternLibrary', 'PatternMatch', 'DiseasePattern',
    'TrajectoryForecaster', 'ForecastResult',
    'EarlyWarningEngine', 'EarlyWarningReport'
]
