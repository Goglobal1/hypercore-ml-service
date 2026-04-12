"""
Evolution Controller - HyperCore
=================================

Central coordinator for the evolution system (the "brain").

Responsibilities:
1. Receive signals from all agents (afferent pathway)
2. Aggregate and analyze signals for learning
3. Generate parameter updates (efferent pathway)
4. Manage the three-lane architecture
5. Coordinate the evolution pipeline

Integration Points:
- DiagnosticEngine → emits predictions, receives threshold updates
- UtilityGate → emits decisions, receives policy updates
- TrialRescueEngine → emits hypotheses, receives scoring weights

Usage:
    # Startup
    controller = EvolutionController()
    await controller.start()

    # Agents auto-register via EvolutionEmitter
    # Signals flow automatically

    # Shutdown
    await controller.stop()
"""

from __future__ import annotations
import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
import json

from .schemas import (
    Lane,
    DeploymentDomain,
    SignalType,
    ParameterType,
    EvolutionSignal,
    ParameterUpdate,
    AgentRegistration,
    EvolutionNode,
    AuditEntry,
)

logger = logging.getLogger(__name__)


class SignalAggregator:
    """
    Aggregates signals over time windows for analysis.

    Tracks rolling statistics for each agent and signal type.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._signals: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.Lock()

    def add(self, signal: EvolutionSignal) -> None:
        """Add a signal to the aggregator."""
        key = f"{signal.agent_id}:{signal.signal_type.value}"
        with self._lock:
            self._signals[key].append(signal)

    def get_recent(
        self,
        agent_id: str,
        signal_type: SignalType,
        limit: int = 100,
    ) -> List[EvolutionSignal]:
        """Get recent signals for an agent/type combination."""
        key = f"{agent_id}:{signal_type.value}"
        with self._lock:
            signals = list(self._signals[key])
        return signals[-limit:]

    def get_stats(self, agent_id: str, signal_type: SignalType) -> Dict[str, Any]:
        """Get aggregated statistics for signals."""
        signals = self.get_recent(agent_id, signal_type, limit=self.window_size)

        if not signals:
            return {"count": 0}

        # Calculate latency stats if available
        latencies = [s.latency_ms for s in signals if s.latency_ms is not None]

        stats = {
            "count": len(signals),
            "oldest": signals[0].timestamp if signals else None,
            "newest": signals[-1].timestamp if signals else None,
        }

        if latencies:
            latencies.sort()
            stats["latency_avg_ms"] = sum(latencies) / len(latencies)
            stats["latency_p50_ms"] = latencies[len(latencies) // 2]
            stats["latency_p95_ms"] = latencies[int(len(latencies) * 0.95)]
            stats["latency_p99_ms"] = latencies[int(len(latencies) * 0.99)]

        return stats


class CalibrationTracker:
    """
    Tracks prediction calibration for agents.

    Compares predicted probabilities to actual outcomes
    to detect calibration drift.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._predictions: Dict[str, List[tuple]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(
        self,
        agent_id: str,
        predicted_prob: float,
        actual_outcome: bool,
    ) -> None:
        """Record a prediction/outcome pair."""
        with self._lock:
            self._predictions[agent_id].append((predicted_prob, actual_outcome))

            # Keep only recent data
            if len(self._predictions[agent_id]) > 10000:
                self._predictions[agent_id] = self._predictions[agent_id][-10000:]

    def get_calibration_error(self, agent_id: str) -> Optional[float]:
        """
        Calculate Expected Calibration Error (ECE).

        Returns None if not enough data.
        """
        with self._lock:
            data = self._predictions.get(agent_id, [])

        if len(data) < 100:
            return None

        # Bin predictions
        bins = [[] for _ in range(self.n_bins)]
        for pred, actual in data:
            bin_idx = min(int(pred * self.n_bins), self.n_bins - 1)
            bins[bin_idx].append((pred, actual))

        # Calculate ECE
        ece = 0.0
        total = len(data)

        for bin_data in bins:
            if not bin_data:
                continue

            n = len(bin_data)
            avg_pred = sum(p for p, _ in bin_data) / n
            avg_actual = sum(a for _, a in bin_data) / n
            ece += (n / total) * abs(avg_pred - avg_actual)

        return ece


class EvolutionController:
    """
    Central coordinator for the evolution system.

    This is the "brain" that:
    - Receives signals from all agents
    - Tracks performance and calibration
    - Generates parameter updates
    - Manages the three-lane architecture
    """

    def __init__(
        self,
        signal_buffer_size: int = 10000,
        update_interval_seconds: float = 60.0,
        calibration_threshold: float = 0.05,
    ):
        """
        Initialize the Evolution Controller.

        Args:
            signal_buffer_size: Size of signal buffer per agent/type
            update_interval_seconds: How often to check for updates
            calibration_threshold: ECE threshold for triggering recalibration
        """
        # Configuration
        self.signal_buffer_size = signal_buffer_size
        self.update_interval = update_interval_seconds
        self.calibration_threshold = calibration_threshold

        # Agent registry
        self._agents: Dict[str, AgentRegistration] = {}
        self._agent_lock = threading.Lock()

        # Signal processing
        self._signal_queue: asyncio.Queue = None  # Set in start()
        self._aggregator = SignalAggregator(window_size=signal_buffer_size)
        self._calibration = CalibrationTracker()

        # Update tracking
        self._pending_updates: Dict[str, List[ParameterUpdate]] = defaultdict(list)
        self._applied_updates: List[ParameterUpdate] = []

        # Audit log
        self._audit_entries: List[AuditEntry] = []
        self._audit_lock = threading.Lock()

        # Lifecycle
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._tasks: List[asyncio.Task] = []

        # Signal handlers
        self._signal_handlers: Dict[SignalType, List[Callable]] = defaultdict(list)

        # Stats
        self._signals_received = 0
        self._updates_generated = 0

        logger.info("Evolution Controller initialized")

    async def start(self) -> None:
        """Start the Evolution Controller."""
        if self._running:
            return

        self._running = True
        self._signal_queue = asyncio.Queue(maxsize=10000)

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._signal_processor()),
            asyncio.create_task(self._update_generator()),
            asyncio.create_task(self._health_monitor()),
        ]

        # Register controller with emitters
        from .emitter import EvolutionEmitter
        EvolutionEmitter.set_controller(self)

        logger.info("Evolution Controller started")

    async def stop(self) -> None:
        """Stop the Evolution Controller."""
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._executor.shutdown(wait=False)
        logger.info("Evolution Controller stopped")

    def register_agent(self, registration: AgentRegistration) -> None:
        """Register an agent with the controller."""
        with self._agent_lock:
            self._agents[registration.agent_id] = registration

        self._audit(
            action="agent_registered",
            actor=f"agent:{registration.agent_id}",
            description=f"Agent {registration.agent_id} registered",
            evidence=registration.to_dict(),
        )

        logger.info(f"Registered agent: {registration.agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        with self._agent_lock:
            if agent_id in self._agents:
                del self._agents[agent_id]

        self._audit(
            action="agent_unregistered",
            actor=f"system",
            description=f"Agent {agent_id} unregistered",
        )

    def agent_heartbeat(self, agent_id: str) -> None:
        """Update agent heartbeat timestamp."""
        with self._agent_lock:
            if agent_id in self._agents:
                self._agents[agent_id].last_heartbeat = datetime.now(timezone.utc).isoformat()

    def receive_signal(self, signal: EvolutionSignal) -> None:
        """
        Receive a signal from an agent.

        This is called by EvolutionEmitter when agents emit signals.
        """
        self._signals_received += 1

        # Add to aggregator
        self._aggregator.add(signal)

        # Queue for async processing
        if self._signal_queue and not self._signal_queue.full():
            try:
                self._signal_queue.put_nowait(signal)
            except asyncio.QueueFull:
                logger.warning("Signal queue full, dropping signal")

        # Track calibration for prediction signals with outcomes
        if signal.signal_type == SignalType.OUTCOME_OBSERVED:
            self._process_outcome(signal)

    def _process_outcome(self, signal: EvolutionSignal) -> None:
        """Process an outcome signal for calibration tracking."""
        payload = signal.payload
        outcome = payload.get("outcome", {})

        if "predicted_prob" in payload and "actual" in outcome:
            self._calibration.record(
                agent_id=signal.agent_id,
                predicted_prob=payload["predicted_prob"],
                actual_outcome=bool(outcome["actual"]),
            )

    async def _signal_processor(self) -> None:
        """Background task to process signals."""
        while self._running:
            try:
                signal = await asyncio.wait_for(
                    self._signal_queue.get(),
                    timeout=1.0,
                )

                # Call registered handlers
                handlers = self._signal_handlers.get(signal.signal_type, [])
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(signal)
                        else:
                            await asyncio.get_event_loop().run_in_executor(
                                self._executor,
                                handler,
                                signal,
                            )
                    except Exception as e:
                        logger.error(f"Signal handler error: {e}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Signal processor error: {e}")

    async def _update_generator(self) -> None:
        """
        Background task to generate parameter updates.

        Periodically checks calibration and performance,
        generating updates when needed.
        """
        while self._running:
            try:
                await asyncio.sleep(self.update_interval)

                # Check each registered agent
                with self._agent_lock:
                    agents = list(self._agents.values())

                for agent in agents:
                    await self._check_agent_calibration(agent)
                    await self._check_agent_performance(agent)

            except Exception as e:
                logger.error(f"Update generator error: {e}")

    async def _check_agent_calibration(self, agent: AgentRegistration) -> None:
        """Check agent calibration and generate update if needed."""
        ece = self._calibration.get_calibration_error(agent.agent_id)

        if ece is None:
            return  # Not enough data

        if ece > self.calibration_threshold:
            # Agent is miscalibrated - suggest threshold adjustment
            current_threshold = agent.current_parameters.get("confidence_threshold")

            if current_threshold is not None:
                # Raise threshold if overconfident
                new_threshold = min(current_threshold + 0.02, 0.99)

                update = ParameterUpdate(
                    target_agent=agent.agent_id,
                    parameter_type=ParameterType.THRESHOLD,
                    parameter_name="confidence_threshold",
                    old_value=current_threshold,
                    new_value=new_threshold,
                    rationale=f"Calibration drift detected (ECE={ece:.4f}), raising threshold",
                    lane=Lane.SHADOW,  # Always shadow first
                )

                self._queue_update(update)

    async def _check_agent_performance(self, agent: AgentRegistration) -> None:
        """Check agent performance metrics."""
        stats = self._aggregator.get_stats(
            agent.agent_id,
            SignalType.PREDICTION,
        )

        # Check for latency issues
        if stats.get("latency_p95_ms", 0) > 1000:  # >1s p95
            logger.warning(
                f"Agent {agent.agent_id} has high latency: "
                f"p95={stats['latency_p95_ms']:.0f}ms"
            )

    def _queue_update(self, update: ParameterUpdate) -> None:
        """Queue an update for an agent."""
        self._pending_updates[update.target_agent].append(update)
        self._updates_generated += 1

        self._audit(
            action="update_generated",
            actor="system:controller",
            description=f"Generated update for {update.target_agent}",
            evidence=update.to_dict(),
        )

        logger.info(
            f"Generated update for {update.target_agent}: "
            f"{update.parameter_name} = {update.new_value}"
        )

    def get_pending_updates(self, agent_id: str) -> List[ParameterUpdate]:
        """Get pending updates for an agent."""
        updates = self._pending_updates.pop(agent_id, [])
        return updates

    async def _health_monitor(self) -> None:
        """Monitor agent health and log statistics."""
        while self._running:
            try:
                await asyncio.sleep(60)

                # Check for inactive agents
                now = datetime.now(timezone.utc)
                with self._agent_lock:
                    for agent_id, agent in self._agents.items():
                        if agent.last_heartbeat:
                            last = datetime.fromisoformat(agent.last_heartbeat.replace('Z', '+00:00'))
                            if (now - last).total_seconds() > 300:  # 5 min
                                if agent.status != "inactive":
                                    agent.status = "inactive"
                                    logger.warning(f"Agent {agent_id} is inactive")

                # Log stats
                logger.info(
                    f"Controller stats: signals={self._signals_received}, "
                    f"updates={self._updates_generated}, "
                    f"agents={len(self._agents)}"
                )

            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    def on_signal(
        self,
        signal_type: SignalType,
        handler: Callable[[EvolutionSignal], None],
    ) -> None:
        """
        Register a handler for a specific signal type.

        Args:
            signal_type: Type of signal to handle
            handler: Function to call when signal received
        """
        self._signal_handlers[signal_type].append(handler)

    def _audit(
        self,
        action: str,
        actor: str,
        description: str = "",
        evidence: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> AuditEntry:
        """Create an audit entry."""
        entry = AuditEntry(
            action=action,
            actor=actor,
            description=description,
            evidence=evidence or {},
            node_id=node_id,
        )

        with self._audit_lock:
            self._audit_entries.append(entry)

            # Keep only recent entries in memory
            if len(self._audit_entries) > 10000:
                self._audit_entries = self._audit_entries[-10000:]

        return entry

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for an agent."""
        with self._agent_lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return {"error": "Agent not found"}

        stats = {
            "agent_id": agent_id,
            "agent_type": agent.agent_type,
            "version": agent.version,
            "status": agent.status,
            "current_parameters": agent.current_parameters,
        }

        # Add signal stats
        for signal_type in SignalType:
            signal_stats = self._aggregator.get_stats(agent_id, signal_type)
            if signal_stats.get("count", 0) > 0:
                stats[f"signals_{signal_type.value}"] = signal_stats

        # Add calibration
        ece = self._calibration.get_calibration_error(agent_id)
        if ece is not None:
            stats["calibration_error"] = ece

        return stats

    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get summary of all registered agents."""
        with self._agent_lock:
            return [
                {
                    "agent_id": a.agent_id,
                    "agent_type": a.agent_type,
                    "version": a.version,
                    "status": a.status,
                    "last_heartbeat": a.last_heartbeat,
                }
                for a in self._agents.values()
            ]

    def get_controller_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "running": self._running,
            "agents_registered": len(self._agents),
            "signals_received": self._signals_received,
            "updates_generated": self._updates_generated,
            "audit_entries": len(self._audit_entries),
        }


# Global controller instance
_controller_instance: Optional[EvolutionController] = None


def get_evolution_controller() -> EvolutionController:
    """Get or create the global Evolution Controller instance."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = EvolutionController()
    return _controller_instance


async def start_evolution_controller() -> EvolutionController:
    """Start the global Evolution Controller."""
    controller = get_evolution_controller()
    await controller.start()
    return controller


async def stop_evolution_controller() -> None:
    """Stop the global Evolution Controller."""
    global _controller_instance
    if _controller_instance:
        await _controller_instance.stop()
        _controller_instance = None
