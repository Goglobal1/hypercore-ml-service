# Alert System - Merged Implementation
# Combines hypercore-ml-service + cse.py features
#
# Unified Clinical State Engine (CSE) with:
# - 4-state clinical model (S0-S3)
# - 15 risk domains with domain-specific thresholds
# - 4 break rules (velocity, novelty, TTH shortening, dwell)
# - Episode lifecycle management
# - Alert routing with escalation timers
# - Real-time push via WebSocket/SSE
# - Full audit logging

from .models import (
    ClinicalState,
    AlertType,
    AlertSeverity,
    ConfidenceLevel,
    SuppressionReason,
    BreakRule,
    HarmType,
    EventType,
    RiskDomain,
    AlertEvent,
    EpisodeState,
    PatientState,
    EvaluationResult,
    AcknowledgmentRecord,
    BreakRuleResult,
    generate_event_id,
    generate_episode_id,
    generate_ack_id,
)
from .config import (
    DomainConfig,
    BiomarkerThreshold,
    SiteConfig,
    get_domain_config,
    get_biomarker_thresholds,
    DOMAIN_CONFIGS,
    BIOMARKER_THRESHOLDS,
    RATIONALE_TEMPLATES,
    RECOMMENDATIONS,
)
from .engine import (
    ClinicalStateEngine,
    evaluate_patient,
    get_engine,
    set_engine,
)
from .storage import (
    StorageBackend,
    InMemoryStorage,
    PostgreSQLStorage,
    RedisCacheWrapper,
    get_storage,
    set_storage,
    init_storage,
)
from .routing import (
    AlertRouter,
    RoutingRule,
    EscalationManager,
    PendingEscalation,
    get_router,
    get_escalation_manager,
    DEFAULT_ROUTING_RULES,
)
from .pipeline import (
    AlertPipeline,
    PipelineResult,
    PipelineStepResult,
    AgentIntegration,
    TTHIntegration,
    process_patient_intake,
    get_pipeline,
    set_pipeline,
)
from .realtime import (
    RealtimeHub,
    ConnectionManager,
    RealtimeMessage,
    MessageType,
    get_hub,
    set_hub,
)
from .risk_calculator import (
    calculate_risk_score,
    quick_risk_score,
    calculate_biomarker_score,
    normalize_biomarker_name,
    get_domain_thresholds,
    BIOMARKER_ALIASES,
)
from .email_config import (
    SMTPSettings,
    get_smtp_settings,
    reset_smtp_settings,
)
from .email_notifier import (
    EmailNotifier,
    EmailResult,
    create_email_callback,
)
from .router import router as alert_router

__all__ = [
    # Models - Enums
    "ClinicalState",
    "AlertType",
    "AlertSeverity",
    "ConfidenceLevel",
    "SuppressionReason",
    "BreakRule",
    "HarmType",
    "EventType",
    "RiskDomain",
    # Models - Dataclasses
    "AlertEvent",
    "EpisodeState",
    "PatientState",
    "EvaluationResult",
    "AcknowledgmentRecord",
    "BreakRuleResult",
    # Models - Helpers
    "generate_event_id",
    "generate_episode_id",
    "generate_ack_id",
    # Config
    "DomainConfig",
    "BiomarkerThreshold",
    "SiteConfig",
    "get_domain_config",
    "get_biomarker_thresholds",
    "DOMAIN_CONFIGS",
    "BIOMARKER_THRESHOLDS",
    "RATIONALE_TEMPLATES",
    "RECOMMENDATIONS",
    # Engine
    "ClinicalStateEngine",
    "evaluate_patient",
    "get_engine",
    "set_engine",
    # Storage
    "StorageBackend",
    "InMemoryStorage",
    "PostgreSQLStorage",
    "RedisCacheWrapper",
    "get_storage",
    "set_storage",
    "init_storage",
    # Routing
    "AlertRouter",
    "RoutingRule",
    "EscalationManager",
    "PendingEscalation",
    "get_router",
    "get_escalation_manager",
    "DEFAULT_ROUTING_RULES",
    # Pipeline
    "AlertPipeline",
    "PipelineResult",
    "PipelineStepResult",
    "AgentIntegration",
    "TTHIntegration",
    "process_patient_intake",
    "get_pipeline",
    "set_pipeline",
    # Realtime
    "RealtimeHub",
    "ConnectionManager",
    "RealtimeMessage",
    "MessageType",
    "get_hub",
    "set_hub",
    # Risk Calculator
    "calculate_risk_score",
    "quick_risk_score",
    "calculate_biomarker_score",
    "normalize_biomarker_name",
    "get_domain_thresholds",
    "BIOMARKER_ALIASES",
    # Email Notifications
    "SMTPSettings",
    "get_smtp_settings",
    "reset_smtp_settings",
    "EmailNotifier",
    "EmailResult",
    "create_email_callback",
    # FastAPI Router
    "alert_router",
]
