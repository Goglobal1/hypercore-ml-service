"""
Evolution Sampling Module
=========================

Sampling algorithms for parent selection in evolution.
Adapted from GAIR-NLP ASI-Evolve.

Algorithms:
- UCB1: Upper Confidence Bound for balancing explore/exploit
- Utility-weighted: Weight by utility score
"""

from .ucb1 import UCB1Sampler

__all__ = ["UCB1Sampler"]
