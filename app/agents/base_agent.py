"""
Base Agent Framework for HyperCore Diagnostic Agents

Provides:
- Standardized I/O schema
- Confidence scoring system
- Inter-agent message protocol
- Agent registry for communication
- Evolution system integration (signal emission, parameter updates)
"""

import uuid
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Type, Callable
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from functools import wraps

from app.core.evolution import (
    EvolutionEmitter,
    SignalType,
    ParameterUpdate,
    DeploymentDomain,
)

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence levels for agent findings."""
    VERY_LOW = "very_low"      # <20%
    LOW = "low"                 # 20-40%
    MODERATE = "moderate"       # 40-60%
    HIGH = "high"               # 60-80%
    VERY_HIGH = "very_high"     # 80-95%
    DEFINITIVE = "definitive"   # >95%


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    FINDING = "finding"           # Share a discovery
    QUERY = "query"               # Request information
    RESPONSE = "response"         # Answer a query
    ALERT = "alert"               # Urgent notification
    CORRELATION = "correlation"   # Cross-agent correlation
    CONSENSUS = "consensus"       # Agreed conclusion


class AgentType(str, Enum):
    """Types of diagnostic agents."""
    BIOMARKER = "biomarker"
    DIAGNOSTIC = "diagnostic"
    TRIAL_RESCUE = "trial_rescue"
    SURVEILLANCE = "surveillance"


@dataclass
class AgentFinding:
    """A single finding from an agent."""
    finding_id: str
    agent_type: AgentType
    category: str
    description: str
    confidence: float
    confidence_level: ConfidenceLevel
    evidence: List[str]
    related_entities: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["agent_type"] = self.agent_type.value
        result["confidence_level"] = self.confidence_level.value
        return result

    @staticmethod
    def get_confidence_level(confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to level."""
        if confidence < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.4:
            return ConfidenceLevel.LOW
        elif confidence < 0.6:
            return ConfidenceLevel.MODERATE
        elif confidence < 0.8:
            return ConfidenceLevel.HIGH
        elif confidence < 0.95:
            return ConfidenceLevel.VERY_HIGH
        else:
            return ConfidenceLevel.DEFINITIVE


@dataclass
class AgentMessage:
    """Message for inter-agent communication."""
    message_id: str
    message_type: MessageType
    sender: AgentType
    recipient: Optional[AgentType]  # None = broadcast
    content: Dict[str, Any]
    findings: List[AgentFinding] = field(default_factory=list)
    priority: int = 1  # 1=normal, 2=elevated, 3=urgent
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None  # For tracking related messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender": self.sender.value,
            "recipient": self.recipient.value if self.recipient else None,
            "content": self.content,
            "findings": [f.to_dict() for f in self.findings],
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
        }


class AgentRegistry:
    """
    Central registry for agent communication.
    Enables agents to discover and communicate with each other.
    """
    _instance = None
    _agents: Dict[AgentType, "BaseAgent"] = {}
    _message_queue: Dict[AgentType, List[AgentMessage]] = defaultdict(list)
    _shared_findings: List[AgentFinding] = []
    _correlation_store: Dict[str, List[AgentFinding]] = defaultdict(list)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, agent: "BaseAgent"):
        """Register an agent in the registry."""
        cls._agents[agent.agent_type] = agent
        logger.info(f"Registered agent: {agent.agent_type.value}")

    @classmethod
    def get_agent(cls, agent_type: AgentType) -> Optional["BaseAgent"]:
        """Get an agent by type."""
        return cls._agents.get(agent_type)

    @classmethod
    def list_agents(cls) -> List[AgentType]:
        """List all registered agents."""
        return list(cls._agents.keys())

    @classmethod
    def send_message(cls, message: AgentMessage):
        """Send a message to an agent or broadcast."""
        if message.recipient:
            cls._message_queue[message.recipient].append(message)
        else:
            # Broadcast to all agents except sender
            for agent_type in cls._agents:
                if agent_type != message.sender:
                    cls._message_queue[agent_type].append(message)

        # Store findings for correlation
        for finding in message.findings:
            cls._shared_findings.append(finding)
            if message.correlation_id:
                cls._correlation_store[message.correlation_id].append(finding)

        logger.debug(f"Message sent: {message.message_type.value} from {message.sender.value}")

    @classmethod
    def get_messages(cls, agent_type: AgentType) -> List[AgentMessage]:
        """Get pending messages for an agent."""
        messages = cls._message_queue[agent_type]
        cls._message_queue[agent_type] = []
        return messages

    @classmethod
    def get_shared_findings(
        cls,
        agent_type: Optional[AgentType] = None,
        category: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[AgentFinding]:
        """Get shared findings with optional filters."""
        findings = cls._shared_findings

        if agent_type:
            findings = [f for f in findings if f.agent_type == agent_type]
        if category:
            findings = [f for f in findings if f.category == category]
        if min_confidence > 0:
            findings = [f for f in findings if f.confidence >= min_confidence]

        return findings

    @classmethod
    def get_correlated_findings(cls, correlation_id: str) -> List[AgentFinding]:
        """Get all findings for a correlation ID."""
        return cls._correlation_store.get(correlation_id, [])

    @classmethod
    def clear_findings(cls):
        """Clear all shared findings (for new analysis session)."""
        cls._shared_findings = []
        cls._correlation_store = defaultdict(list)


class BaseAgent(ABC):
    """
    Base class for all HyperCore diagnostic agents.

    Provides:
    - Standardized analyze() interface
    - Confidence scoring
    - Inter-agent communication
    - Finding generation
    - Evolution system integration (signal emission, parameter updates)
    """

    # Version for all agents (can be overridden)
    VERSION = "1.0.0"

    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.agent_id = f"{agent_type.value}_{uuid.uuid4().hex[:8]}"
        self._findings: List[AgentFinding] = []
        self._received_messages: List[AgentMessage] = []

        # Evolution system integration
        self._emitter = EvolutionEmitter(
            agent_id=self.agent_id,
            agent_type=agent_type.value,
            version=self.VERSION,
            domain=DeploymentDomain.CLINICAL,
            configurable_parameters=self._get_configurable_parameters(),
        )

        # Track pending outcomes for feedback loop
        self._pending_outcomes: Dict[str, Dict[str, Any]] = {}

        # Register with central registry
        AgentRegistry.register(self)

    def _get_configurable_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Override to define configurable parameters for evolution.

        Returns:
            Dictionary of parameter definitions:
            {
                "param_name": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.5
                }
            }
        """
        return {}

    @property
    def emitter(self) -> EvolutionEmitter:
        """Get the evolution emitter for this agent."""
        return self._emitter

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent description."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """List of agent capabilities."""
        pass

    @abstractmethod
    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main analysis method. Must be implemented by subclasses.

        Args:
            input_data: Agent-specific input data

        Returns:
            Analysis results with findings
        """
        pass

    def create_finding(
        self,
        category: str,
        description: str,
        confidence: float,
        evidence: List[str],
        related_entities: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentFinding:
        """Create a new finding."""
        finding = AgentFinding(
            finding_id=f"{self.agent_type.value}_{uuid.uuid4().hex[:8]}",
            agent_type=self.agent_type,
            category=category,
            description=description,
            confidence=confidence,
            confidence_level=AgentFinding.get_confidence_level(confidence),
            evidence=evidence,
            related_entities=related_entities or {},
            metadata=metadata or {}
        )
        self._findings.append(finding)
        return finding

    def send_finding(
        self,
        finding: AgentFinding,
        recipient: Optional[AgentType] = None,
        priority: int = 1,
        correlation_id: Optional[str] = None
    ):
        """Send a finding to another agent or broadcast."""
        message = AgentMessage(
            message_id=uuid.uuid4().hex,
            message_type=MessageType.FINDING,
            sender=self.agent_type,
            recipient=recipient,
            content={"finding_summary": finding.description},
            findings=[finding],
            priority=priority,
            correlation_id=correlation_id
        )
        AgentRegistry.send_message(message)

    def broadcast_findings(
        self,
        findings: List[AgentFinding],
        correlation_id: Optional[str] = None
    ):
        """Broadcast multiple findings to all agents."""
        message = AgentMessage(
            message_id=uuid.uuid4().hex,
            message_type=MessageType.FINDING,
            sender=self.agent_type,
            recipient=None,
            content={"finding_count": len(findings)},
            findings=findings,
            priority=2 if len(findings) > 3 else 1,
            correlation_id=correlation_id
        )
        AgentRegistry.send_message(message)

    def send_alert(
        self,
        alert_message: str,
        severity: str,
        related_findings: List[AgentFinding],
        recipient: Optional[AgentType] = None
    ):
        """Send an urgent alert."""
        message = AgentMessage(
            message_id=uuid.uuid4().hex,
            message_type=MessageType.ALERT,
            sender=self.agent_type,
            recipient=recipient,
            content={
                "alert": alert_message,
                "severity": severity,
            },
            findings=related_findings,
            priority=3
        )
        AgentRegistry.send_message(message)

    def query_agent(
        self,
        recipient: AgentType,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send a query to another agent."""
        query_id = uuid.uuid4().hex
        message = AgentMessage(
            message_id=query_id,
            message_type=MessageType.QUERY,
            sender=self.agent_type,
            recipient=recipient,
            content={
                "query": query,
                "context": context or {}
            },
            correlation_id=query_id
        )
        AgentRegistry.send_message(message)
        return query_id

    def receive_messages(self) -> List[AgentMessage]:
        """Receive pending messages from other agents."""
        messages = AgentRegistry.get_messages(self.agent_type)
        self._received_messages.extend(messages)
        return messages

    def get_peer_findings(
        self,
        agent_type: Optional[AgentType] = None,
        min_confidence: float = 0.5
    ) -> List[AgentFinding]:
        """Get findings from peer agents."""
        return AgentRegistry.get_shared_findings(
            agent_type=agent_type,
            min_confidence=min_confidence
        )

    def calculate_aggregate_confidence(
        self,
        confidences: List[float],
        weights: Optional[List[float]] = None
    ) -> float:
        """Calculate weighted aggregate confidence."""
        if not confidences:
            return 0.0

        if weights is None:
            weights = [1.0] * len(confidences)

        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        total_weight = sum(weights)

        return min(0.99, weighted_sum / total_weight) if total_weight > 0 else 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "findings_count": len(self._findings),
            "messages_received": len(self._received_messages),
            "status": "active",
            "evolution": {
                "signals_emitted": self._emitter.get_stats().get("signals_emitted", 0),
                "outcomes_recorded": self._emitter.get_stats().get("outcomes_recorded", 0),
                "pending_outcomes": len(self._pending_outcomes),
            },
        }

    def clear_session(self):
        """Clear findings for a new session."""
        self._findings = []
        self._received_messages = []

    # =========================================================================
    # EVOLUTION SYSTEM INTEGRATION
    # =========================================================================

    def emit_signal(
        self,
        signal_type: SignalType,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> str:
        """
        Emit an evolution signal.

        Args:
            signal_type: Type of signal (PREDICTION, DECISION, etc.)
            payload: Signal payload data
            session_id: Optional session identifier

        Returns:
            Request ID for tracking outcomes
        """
        signal = self._emitter.emit(
            signal_type=signal_type,
            payload=payload,
            session_id=session_id,
        )
        return signal.request_id

    def record_outcome(
        self,
        request_id: str,
        outcome: Dict[str, Any],
    ) -> None:
        """
        Record the outcome of a previous signal for feedback loop.

        Args:
            request_id: The request_id from emit_signal
            outcome: Outcome data (e.g., {"correct": True, "actual": "diagnosis"})
        """
        self._emitter.record_outcome(request_id, outcome)

    def apply_parameter_update(self, update: ParameterUpdate) -> bool:
        """
        Apply a parameter update from the Evolution Controller.

        Args:
            update: The parameter update to apply

        Returns:
            True if applied successfully
        """
        return self._emitter.apply_update(update)

    def get_parameter(self, name: str) -> Any:
        """Get current value of a configurable parameter."""
        return self._emitter.get_parameter(name)

    def on_parameter_change(
        self,
        parameter_name: str,
        callback: Callable[[Any, Any], None],
    ) -> None:
        """
        Register callback for parameter changes.

        Args:
            parameter_name: Name of parameter to watch
            callback: Function(old_value, new_value) called on change
        """
        self._emitter.on_parameter_change(parameter_name, callback)

    async def analyze_with_evolution(
        self,
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run analysis with automatic evolution signal emission.

        Wraps analyze() with:
        - Pre-analysis signal emission
        - Timing measurement
        - Post-analysis outcome recording

        Args:
            input_data: Agent-specific input data
            session_id: Optional session identifier

        Returns:
            Analysis results with evolution metadata
        """
        start_time = time.perf_counter()

        # Emit start signal
        request_id = self.emit_signal(
            signal_type=SignalType.PREDICTION,
            payload={
                "agent_type": self.agent_type.value,
                "input_keys": list(input_data.keys()),
            },
            session_id=session_id,
        )

        try:
            # Run actual analysis
            result = await self.analyze(input_data)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Emit latency signal
            self._emitter.emit(
                signal_type=SignalType.LATENCY,
                payload={"latency_ms": latency_ms},
                session_id=session_id,
            )

            # Add evolution metadata to result
            result["_evolution"] = {
                "request_id": request_id,
                "latency_ms": latency_ms,
                "agent_id": self.agent_id,
            }

            # Store for later outcome recording
            self._pending_outcomes[request_id] = {
                "started_at": start_time,
                "input_data": input_data,
                "result": result,
            }

            return result

        except Exception as e:
            # Emit error signal
            self._emitter.emit(
                signal_type=SignalType.ERROR,
                payload={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                session_id=session_id,
            )
            raise

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution-related statistics for this agent."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "emitter_stats": self._emitter.get_stats(),
            "pending_outcomes": len(self._pending_outcomes),
        }
