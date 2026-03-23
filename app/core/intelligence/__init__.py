"""
Unified Intelligence Layer - The Brain of HyperCore

Every module contributes patterns here.
Every module queries insights from here.
Cross-domain correlations enable insights no single module could generate.
"""

from .patterns import (
    Pattern, PatternType, PatternSource,
    TrajectoryPattern, GenomicPattern, PharmaPattern,
    PathogenPattern, MultiomicPattern, AlertPattern,
    SurveillancePattern, ClinicalPattern
)
from .pattern_store import PatternStore
from .correlator import CrossDomainCorrelator, Correlation, CorrelationType
from .insights import InsightGenerator, UnifiedInsight, ViewFocus
from .unified_layer import UnifiedIntelligenceLayer

# Singleton instance - ALL modules use this same instance
_intelligence_instance = None


def get_intelligence() -> 'UnifiedIntelligenceLayer':
    """Get the singleton intelligence layer instance."""
    global _intelligence_instance
    if _intelligence_instance is None:
        _intelligence_instance = UnifiedIntelligenceLayer()
    return _intelligence_instance


def reset_intelligence():
    """Reset intelligence layer (for testing)."""
    global _intelligence_instance
    _intelligence_instance = None


__all__ = [
    'Pattern', 'PatternType', 'PatternSource',
    'TrajectoryPattern', 'GenomicPattern', 'PharmaPattern',
    'PathogenPattern', 'MultiomicPattern', 'AlertPattern',
    'SurveillancePattern', 'ClinicalPattern',
    'PatternStore',
    'CrossDomainCorrelator', 'Correlation', 'CorrelationType',
    'InsightGenerator', 'UnifiedInsight', 'ViewFocus',
    'UnifiedIntelligenceLayer',
    'get_intelligence', 'reset_intelligence'
]
