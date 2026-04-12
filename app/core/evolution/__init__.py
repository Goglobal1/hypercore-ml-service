"""
HyperCore Evolution System
==========================

CENTRAL NERVOUS SYSTEM for DiviScan.

Every agent (3 active, scaling to 96+) uses this system to:
1. EMIT evolution signals after operations
2. RECEIVE parameter updates from the Evolution Controller
3. LEARN continuously from outcomes

Architecture:
- Production Lane: Validated, human-approved capabilities (read-only for AI)
- Shadow Lane: AI-generated candidates under evaluation
- Promotion Lane: Human approval queue for shadow → production

Components:
- schemas: Core data structures (EvolutionSignal, EvolutionNode, etc.)
- emitter: EvolutionEmitter class (every agent uses this)
- controller: Central Evolution Controller
- database: Three-lane storage with sampling algorithms
- pipeline: Evolution orchestration and agents
- cognition: Domain knowledge and regulatory guidance
- audit: FDA-compliant audit trails
"""

from .schemas import (
    # Enums
    Lane,
    CapabilityTier,
    ApprovalStatus,
    EscalationLevel,
    EvolutionNodeType,
    DeploymentDomain,
    SignalType,
    ParameterType,
    CognitionItemType,
    # Signal/Update structures (Nervous System Core)
    EvolutionSignal,
    ParameterUpdate,
    AgentRegistration,
    # Core structures
    EvolutionNode,
    UtilityBreakdown,
    EvaluationResult,
    AuditEntry,
    CognitionItem,
    PromotionRequest,
    # Type aliases
    NodeId,
    ItemId,
    RequestId,
    EntryId,
)

from .emitter import EvolutionEmitter, create_emitter
from .controller import (
    EvolutionController,
    get_evolution_controller,
    start_evolution_controller,
    stop_evolution_controller,
)
from .pipeline import (
    EvolutionOrchestrator,
    EvolutionStep,
    OrchestratorConfig,
    create_orchestrator,
    ResearcherAgent,
    EvaluatorAgent,
    AnalyzerAgent,
)
from .database import (
    ProductionStore,
    ShadowStore,
    PromotionQueue,
)
from .audit import AuditTrail, get_audit_trail, audit_log
from .sampling import UCB1Sampler

__all__ = [
    # Enums
    "Lane",
    "CapabilityTier",
    "ApprovalStatus",
    "EscalationLevel",
    "EvolutionNodeType",
    "DeploymentDomain",
    "SignalType",
    "ParameterType",
    "CognitionItemType",
    # Signal/Update structures (Nervous System Core)
    "EvolutionSignal",
    "ParameterUpdate",
    "AgentRegistration",
    # Core structures
    "EvolutionNode",
    "UtilityBreakdown",
    "EvaluationResult",
    "AuditEntry",
    "CognitionItem",
    "PromotionRequest",
    # Type aliases
    "NodeId",
    "ItemId",
    "RequestId",
    "EntryId",
    # Emitter (every agent uses this)
    "EvolutionEmitter",
    "create_emitter",
    # Controller (central coordinator)
    "EvolutionController",
    "get_evolution_controller",
    "start_evolution_controller",
    "stop_evolution_controller",
    # Pipeline
    "EvolutionOrchestrator",
    "EvolutionStep",
    "OrchestratorConfig",
    "create_orchestrator",
    "ResearcherAgent",
    "EvaluatorAgent",
    "AnalyzerAgent",
    # Database (three-lane architecture)
    "ProductionStore",
    "ShadowStore",
    "PromotionQueue",
    # Audit
    "AuditTrail",
    "get_audit_trail",
    "audit_log",
    # Sampling
    "UCB1Sampler",
]

__version__ = "1.0.0"
