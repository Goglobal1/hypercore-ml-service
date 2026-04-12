"""
Evolution Cognition Module
==========================

Domain knowledge and regulatory guidance storage:
- Medical knowledge base (clinical protocols, drug interactions)
- Regulatory guidance (FDA, EMA, ICH guidelines)
- Prior experiment learnings

Components:
- CognitionStore: Main storage with embedding-based retrieval
- Loaders: Pre-built knowledge loaders (FDA, clinical, safety)
"""

from .store import (
    CognitionStore,
    CognitionConfig,
    RetrievalResult,
    get_cognition_store,
)
from .loaders import (
    BaseLoader,
    FDAGuidanceLoader,
    ClinicalProtocolLoader,
    SafetyConstraintLoader,
    EvolutionLessonLoader,
    JSONFileLoader,
    load_all_default_knowledge,
)

__all__ = [
    # Store
    "CognitionStore",
    "CognitionConfig",
    "RetrievalResult",
    "get_cognition_store",
    # Loaders
    "BaseLoader",
    "FDAGuidanceLoader",
    "ClinicalProtocolLoader",
    "SafetyConstraintLoader",
    "EvolutionLessonLoader",
    "JSONFileLoader",
    "load_all_default_knowledge",
]
