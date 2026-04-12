"""
Evolution Emitter - HyperCore
=============================

Every agent in the system uses EvolutionEmitter to:
1. Register with the Evolution Controller
2. Emit signals after operations (predictions, decisions, errors)
3. Receive and apply parameter updates

This is the primary integration point for agents into the
central nervous system.

Usage:
    from app.core.evolution import EvolutionEmitter, SignalType

    class DiagnosticEngine:
        def __init__(self):
            self.emitter = EvolutionEmitter(
                agent_id="diagnostic_engine",
                agent_type="diagnostic",
                version="1.0.0",
                configurable_parameters={
                    "confidence_threshold": {"type": "float", "min": 0.5, "max": 1.0, "default": 0.85},
                    "ensemble_weights": {"type": "dict", "default": {}},
                }
            )
            self.confidence_threshold = 0.85

        def predict(self, features):
            start_time = time.time()
            prediction = self._run_model(features)
            latency = (time.time() - start_time) * 1000

            # Emit prediction signal
            self.emitter.emit(
                signal_type=SignalType.PREDICTION,
                payload={
                    "prediction": prediction.score,
                    "confidence": prediction.confidence,
                    "features_used": list(features.keys()),
                },
                latency_ms=latency,
            )

            return prediction

        def receive_outcome(self, request_id: str, actual_outcome: dict):
            # Link outcome to original prediction
            self.emitter.record_outcome(request_id, actual_outcome)
"""

from __future__ import annotations
import asyncio
import logging
import time
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from contextlib import contextmanager

from .schemas import (
    DeploymentDomain,
    SignalType,
    ParameterType,
    EvolutionSignal,
    ParameterUpdate,
    AgentRegistration,
)

if TYPE_CHECKING:
    from .controller import EvolutionController

logger = logging.getLogger(__name__)


class EvolutionEmitter:
    """
    Signal emitter for agents in the evolution system.

    Each agent creates one EvolutionEmitter instance and uses it to:
    - Register with the Evolution Controller
    - Emit signals after operations
    - Receive and apply parameter updates
    - Track outcomes for feedback loop

    Thread-safe and supports both sync and async emission.
    """

    # Global controller reference (set by controller on startup)
    _controller: Optional["EvolutionController"] = None

    # Signal buffer for offline operation
    _offline_buffer: deque = deque(maxlen=10000)
    _offline_mode: bool = False

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        version: str = "1.0.0",
        domain: DeploymentDomain = DeploymentDomain.RESEARCH,
        description: str = "",
        emits_signals: Optional[List[SignalType]] = None,
        configurable_parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize an emitter for an agent.

        Args:
            agent_id: Unique identifier for this agent instance
            agent_type: Type of agent (e.g., "diagnostic", "utility_gate")
            version: Agent version string
            domain: Deployment domain (pharma, clinical, research, admin)
            description: Human-readable description of agent
            emits_signals: List of signal types this agent emits
            configurable_parameters: Dict of parameter specs
                Format: {"param_name": {"type": "float", "min": 0, "max": 1, "default": 0.5}}
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.version = version
        self.domain = domain
        self.description = description

        self.emits_signals = emits_signals or [
            SignalType.PREDICTION,
            SignalType.ERROR,
            SignalType.LATENCY,
        ]

        self.configurable_parameters = configurable_parameters or {}
        self.current_parameters: Dict[str, Any] = {
            name: spec.get("default")
            for name, spec in self.configurable_parameters.items()
        }

        # Parameter update callbacks
        self._parameter_callbacks: Dict[str, List[Callable[[Any, Any], None]]] = {}

        # Signal tracking for outcome linking
        self._pending_signals: Dict[str, EvolutionSignal] = {}
        self._signal_lock = threading.Lock()

        # Session tracking
        self._current_session_id: Optional[str] = None

        # Stats
        self._signals_emitted = 0
        self._outcomes_recorded = 0
        self._updates_received = 0

        # Register on creation
        self._register()

    def _register(self) -> None:
        """Register this agent with the Evolution Controller."""
        registration = AgentRegistration(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            version=self.version,
            description=self.description,
            domain=self.domain,
            emits_signals=[s.value for s in self.emits_signals],
            configurable_parameters=self.configurable_parameters,
            current_parameters=self.current_parameters,
        )

        if self._controller:
            try:
                self._controller.register_agent(registration)
                logger.info(f"Registered agent {self.agent_id} with Evolution Controller")
            except Exception as e:
                logger.warning(f"Failed to register agent {self.agent_id}: {e}")
        else:
            logger.debug(f"Evolution Controller not available, agent {self.agent_id} will register later")

    @classmethod
    def set_controller(cls, controller: "EvolutionController") -> None:
        """Set the global Evolution Controller reference."""
        cls._controller = controller
        cls._offline_mode = False

        # Flush any buffered signals
        while cls._offline_buffer:
            signal = cls._offline_buffer.popleft()
            try:
                controller.receive_signal(signal)
            except Exception as e:
                logger.error(f"Failed to flush buffered signal: {e}")

    @classmethod
    def enable_offline_mode(cls) -> None:
        """Enable offline mode - signals buffer locally."""
        cls._offline_mode = True
        logger.warning("Evolution Emitter entering offline mode - signals will be buffered")

    def emit(
        self,
        signal_type: SignalType,
        payload: Dict[str, Any],
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        resource_usage: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvolutionSignal:
        """
        Emit an evolution signal.

        Args:
            signal_type: Type of signal
            payload: Signal-specific data
            request_id: Original request ID (for linking outcomes)
            session_id: Session ID (groups related signals)
            patient_id: Anonymized patient reference
            latency_ms: Operation latency in milliseconds
            resource_usage: Resource usage metrics
            metadata: Additional metadata

        Returns:
            The emitted EvolutionSignal
        """
        signal = EvolutionSignal(
            agent_id=self.agent_id,
            agent_version=self.version,
            domain=self.domain,
            signal_type=signal_type,
            payload=payload,
            session_id=session_id or self._current_session_id,
            request_id=request_id,
            patient_id=patient_id,
            latency_ms=latency_ms,
            resource_usage=resource_usage,
            metadata=metadata or {},
        )

        # Track for outcome linking using signal_id
        # This allows outcomes to be linked even without a request_id
        with self._signal_lock:
            self._pending_signals[signal.signal_id] = signal
            # Also track by request_id if provided
            if request_id:
                self._pending_signals[request_id] = signal

        # Send to controller
        self._send_signal(signal)
        self._signals_emitted += 1

        return signal

    def _send_signal(self, signal: EvolutionSignal) -> None:
        """Send signal to controller or buffer if offline."""
        if self._offline_mode or self._controller is None:
            self._offline_buffer.append(signal)
            return

        try:
            self._controller.receive_signal(signal)
        except Exception as e:
            logger.error(f"Failed to send signal {signal.signal_id}: {e}")
            self._offline_buffer.append(signal)

    async def emit_async(
        self,
        signal_type: SignalType,
        payload: Dict[str, Any],
        **kwargs,
    ) -> EvolutionSignal:
        """Async version of emit."""
        # For now, just run sync version in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.emit(signal_type, payload, **kwargs)
        )

    def record_outcome(
        self,
        request_id: str,
        outcome: Dict[str, Any],
    ) -> Optional[EvolutionSignal]:
        """
        Record an outcome for a previous prediction.

        Links the outcome to the original signal for feedback learning.

        Args:
            request_id: Request ID of the original prediction
            outcome: The actual outcome data

        Returns:
            Updated signal with outcome attached, or None if not found
        """
        with self._signal_lock:
            original_signal = self._pending_signals.pop(request_id, None)

        if original_signal is None:
            logger.warning(f"No pending signal found for request_id: {request_id}")
            return None

        # Create signal with outcome
        updated_signal = original_signal.with_outcome(outcome)

        # Emit outcome signal
        self.emit(
            signal_type=SignalType.OUTCOME_OBSERVED,
            payload={
                "original_signal_id": original_signal.signal_id,
                "outcome": outcome,
            },
            request_id=request_id,
        )

        self._outcomes_recorded += 1
        return updated_signal

    def apply_update(self, update: ParameterUpdate) -> bool:
        """
        Apply a parameter update from the controller.

        Args:
            update: The parameter update to apply

        Returns:
            True if applied successfully, False otherwise
        """
        if update.target_agent != self.agent_id:
            return False

        param_name = update.parameter_name

        if param_name not in self.configurable_parameters:
            update.error = f"Unknown parameter: {param_name}"
            logger.warning(f"Unknown parameter {param_name} for agent {self.agent_id}")
            return False

        old_value = self.current_parameters.get(param_name)
        new_value = update.new_value

        # Validate against spec
        spec = self.configurable_parameters[param_name]
        if not self._validate_parameter(new_value, spec):
            update.error = f"Value {new_value} violates parameter spec"
            logger.warning(f"Invalid value {new_value} for parameter {param_name}")
            return False

        # Apply update
        self.current_parameters[param_name] = new_value
        update.applied = True
        update.applied_at = datetime.now(timezone.utc).isoformat()

        # Notify callbacks
        if param_name in self._parameter_callbacks:
            for callback in self._parameter_callbacks[param_name]:
                try:
                    callback(old_value, new_value)
                except Exception as e:
                    logger.error(f"Parameter callback failed for {param_name}: {e}")

        self._updates_received += 1
        logger.info(f"Applied update to {param_name}: {old_value} -> {new_value}")

        return True

    def _validate_parameter(self, value: Any, spec: Dict[str, Any]) -> bool:
        """Validate a parameter value against its spec."""
        param_type = spec.get("type", "any")

        if param_type == "float":
            if not isinstance(value, (int, float)):
                return False
            min_val = spec.get("min")
            max_val = spec.get("max")
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False

        elif param_type == "int":
            if not isinstance(value, int):
                return False
            min_val = spec.get("min")
            max_val = spec.get("max")
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False

        elif param_type == "bool":
            if not isinstance(value, bool):
                return False

        elif param_type == "str":
            if not isinstance(value, str):
                return False
            allowed = spec.get("allowed")
            if allowed and value not in allowed:
                return False

        elif param_type == "dict":
            if not isinstance(value, dict):
                return False

        elif param_type == "list":
            if not isinstance(value, list):
                return False

        return True

    def on_parameter_change(
        self,
        parameter_name: str,
        callback: Callable[[Any, Any], None],
    ) -> None:
        """
        Register a callback for when a parameter changes.

        Args:
            parameter_name: Name of the parameter to watch
            callback: Function(old_value, new_value) to call on change
        """
        if parameter_name not in self._parameter_callbacks:
            self._parameter_callbacks[parameter_name] = []
        self._parameter_callbacks[parameter_name].append(callback)

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get current value of a parameter."""
        return self.current_parameters.get(name, default)

    def set_parameter_local(self, name: str, value: Any) -> bool:
        """
        Set parameter locally (without controller involvement).

        Use this for initialization or testing. In production,
        parameters should be set by the controller.
        """
        if name not in self.configurable_parameters:
            return False

        spec = self.configurable_parameters[name]
        if not self._validate_parameter(value, spec):
            return False

        old_value = self.current_parameters.get(name)
        self.current_parameters[name] = value

        if name in self._parameter_callbacks:
            for callback in self._parameter_callbacks[name]:
                try:
                    callback(old_value, value)
                except Exception as e:
                    logger.error(f"Parameter callback failed: {e}")

        return True

    @contextmanager
    def session(self, session_id: str):
        """
        Context manager for session tracking.

        All signals emitted within the session will have the session_id attached.

        Usage:
            with emitter.session("patient-visit-123"):
                emitter.emit(...)  # automatically gets session_id
        """
        old_session = self._current_session_id
        self._current_session_id = session_id
        try:
            yield
        finally:
            self._current_session_id = old_session

    @contextmanager
    def timed_operation(
        self,
        signal_type: SignalType = SignalType.LATENCY,
        operation_name: str = "operation",
    ):
        """
        Context manager that emits latency signal after operation.

        Usage:
            with emitter.timed_operation(operation_name="predict"):
                result = model.predict(data)
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.emit(
                signal_type=signal_type,
                payload={"operation": operation_name},
                latency_ms=elapsed_ms,
            )

    def heartbeat(self) -> None:
        """Send heartbeat to controller."""
        if self._controller:
            try:
                self._controller.agent_heartbeat(self.agent_id)
            except Exception as e:
                logger.debug(f"Heartbeat failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get emitter statistics."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "version": self.version,
            "signals_emitted": self._signals_emitted,
            "outcomes_recorded": self._outcomes_recorded,
            "updates_received": self._updates_received,
            "pending_signals": len(self._pending_signals),
            "offline_buffered": len(self._offline_buffer),
            "current_parameters": self.current_parameters,
        }


# Convenience function for creating emitters
def create_emitter(
    agent_id: str,
    agent_type: str,
    **kwargs,
) -> EvolutionEmitter:
    """
    Create an EvolutionEmitter with sensible defaults.

    Args:
        agent_id: Unique agent identifier
        agent_type: Type of agent
        **kwargs: Additional arguments for EvolutionEmitter

    Returns:
        Configured EvolutionEmitter instance
    """
    return EvolutionEmitter(
        agent_id=agent_id,
        agent_type=agent_type,
        **kwargs,
    )
