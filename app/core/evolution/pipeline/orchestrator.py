"""
Evolution Pipeline Orchestrator - HyperCore
============================================

Central orchestrator for the evolution loop.
Adapted from GAIR-NLP ASI-Evolve with healthcare additions.

The Loop:
1. LEARN - Retrieve prior knowledge and sample parent nodes
2. DESIGN - Generate hypothesis (via Researcher agent)
3. EXPERIMENT - Evaluate in shadow lane (via Evaluator agent)
4. ANALYZE - Extract lessons and update knowledge (via Analyzer agent)

Integration with Three-Lane Architecture:
- All experiments run in Shadow lane
- Successful experiments queue for Promotion
- Human approval required for Production
"""

from __future__ import annotations
import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ..schemas import (
    Lane,
    EvolutionNode,
    EvolutionNodeType,
    ApprovalStatus,
    EscalationLevel,
    DeploymentDomain,
    SignalType,
    CognitionItem,
)
from ..database import ShadowStore, ProductionStore, PromotionQueue
from ..sampling import UCB1Sampler
from ..audit import AuditTrail, audit_log
from ..emitter import EvolutionEmitter

if TYPE_CHECKING:
    from .agents.researcher import ResearcherAgent
    from .agents.evaluator import EvaluatorAgent
    from .agents.analyzer import AnalyzerAgent

logger = logging.getLogger(__name__)


@dataclass
class EvolutionStep:
    """Record of a single evolution step."""
    step_id: int
    started_at: str
    completed_at: Optional[str] = None

    # Results
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    score: float = 0.0
    success: bool = False

    # Timing
    learn_time_ms: float = 0.0
    design_time_ms: float = 0.0
    experiment_time_ms: float = 0.0
    analyze_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Details
    parents_sampled: int = 0
    cognition_retrieved: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "score": self.score,
            "success": self.success,
            "learn_time_ms": self.learn_time_ms,
            "design_time_ms": self.design_time_ms,
            "experiment_time_ms": self.experiment_time_ms,
            "analyze_time_ms": self.analyze_time_ms,
            "total_time_ms": self.total_time_ms,
            "parents_sampled": self.parents_sampled,
            "cognition_retrieved": self.cognition_retrieved,
            "error": self.error,
        }


@dataclass
class OrchestratorConfig:
    """Configuration for the Evolution Orchestrator."""
    # Storage
    storage_dir: Path = Path("data/evolution")

    # Sampling
    sample_n: int = 3  # Parents to sample per step
    ucb1_c: float = 1.414  # Exploration coefficient

    # Limits
    max_shadow_size: int = 10000
    max_steps_per_run: int = 100
    step_timeout_seconds: float = 300.0

    # Auto-promotion
    auto_promote_threshold: Optional[float] = 0.9  # Score for auto-queue

    # Parallelism
    num_workers: int = 1  # 1 = sequential

    # Domain
    default_domain: DeploymentDomain = DeploymentDomain.RESEARCH

    def __post_init__(self):
        self.storage_dir = Path(self.storage_dir)


class EvolutionOrchestrator:
    """
    Central orchestrator for the evolution pipeline.

    Coordinates:
    - Three-lane database (shadow, production, promotion)
    - Agent pipeline (researcher, evaluator, analyzer)
    - UCB1 sampling for parent selection
    - Audit logging for FDA compliance

    Usage:
        orchestrator = EvolutionOrchestrator(config)
        await orchestrator.start()

        # Run evolution steps
        result = await orchestrator.run_step(task_description="...")

        # Or run multiple steps
        await orchestrator.run(max_steps=10, task_description="...")

        await orchestrator.stop()
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the orchestrator."""
        self.config = config or OrchestratorConfig()
        self.config.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize stores
        self.shadow = ShadowStore(
            storage_dir=self.config.storage_dir / "shadow",
            max_size=self.config.max_shadow_size,
            auto_promote_threshold=self.config.auto_promote_threshold,
        )
        self.production = ProductionStore(
            storage_dir=self.config.storage_dir / "production",
        )
        self.promotion = PromotionQueue(
            storage_dir=self.config.storage_dir / "promotion",
        )

        # Initialize sampler
        self.sampler = UCB1Sampler(c=self.config.ucb1_c)
        self.shadow.set_sampler(self.sampler)

        # Connect promotion callback
        self.shadow.set_promotion_callback(self._on_promotion_queued)

        # Initialize audit
        self.audit = AuditTrail(
            storage_dir=self.config.storage_dir / "audit",
        )

        # Agents (set via set_agents or created later)
        self.researcher: Optional["ResearcherAgent"] = None
        self.evaluator: Optional["EvaluatorAgent"] = None
        self.analyzer: Optional["AnalyzerAgent"] = None

        # Cognition store (optional)
        self._cognition_items: List[CognitionItem] = []

        # Evolution emitter for orchestrator signals
        self.emitter = EvolutionEmitter(
            agent_id="evolution_orchestrator",
            agent_type="orchestrator",
            version="1.0.0",
            domain=self.config.default_domain,
        )

        # State
        self._running = False
        self._step_count = 0
        self._step_lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)

        # History
        self._step_history: List[EvolutionStep] = []

        # Callbacks
        self._on_step_complete: List[Callable[[EvolutionStep], None]] = []

        logger.info(f"EvolutionOrchestrator initialized at {self.config.storage_dir}")

    def set_agents(
        self,
        researcher: "ResearcherAgent",
        evaluator: "EvaluatorAgent",
        analyzer: "AnalyzerAgent",
    ) -> None:
        """Set the pipeline agents."""
        self.researcher = researcher
        self.evaluator = evaluator
        self.analyzer = analyzer
        logger.info("Pipeline agents configured")

    def add_cognition(self, items: List[CognitionItem]) -> None:
        """Add items to the cognition store."""
        self._cognition_items.extend(items)
        logger.info(f"Added {len(items)} cognition items (total: {len(self._cognition_items)})")

    async def start(self) -> None:
        """Start the orchestrator."""
        if self._running:
            return

        self._running = True

        audit_log(
            action="orchestrator_started",
            actor="system:orchestrator",
            description="Evolution orchestrator started",
        )

        logger.info("Evolution orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        self._executor.shutdown(wait=False)

        audit_log(
            action="orchestrator_stopped",
            actor="system:orchestrator",
            description="Evolution orchestrator stopped",
        )

        logger.info("Evolution orchestrator stopped")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def run(
        self,
        max_steps: int = 10,
        task_description: str = "",
        eval_config: Optional[Dict[str, Any]] = None,
    ) -> List[EvolutionStep]:
        """
        Run multiple evolution steps.

        Args:
            max_steps: Maximum steps to run
            task_description: Description of the optimization task
            eval_config: Configuration for evaluation

        Returns:
            List of step results
        """
        max_steps = min(max_steps, self.config.max_steps_per_run)
        results = []

        logger.info(f"Starting evolution run: {max_steps} steps")

        if self.config.num_workers == 1:
            # Sequential execution
            for _ in range(max_steps):
                if not self._running:
                    break

                result = await self.run_step(
                    task_description=task_description,
                    eval_config=eval_config,
                )
                results.append(result)

        else:
            # Parallel execution
            tasks = [
                self.run_step(task_description, eval_config)
                for _ in range(max_steps)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to failed steps
            processed = []
            for r in results:
                if isinstance(r, Exception):
                    step = EvolutionStep(
                        step_id=-1,
                        started_at=datetime.now(timezone.utc).isoformat(),
                        success=False,
                        error=str(r),
                    )
                    processed.append(step)
                else:
                    processed.append(r)
            results = processed

        # Summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Evolution run complete: {successful}/{len(results)} successful")

        return results

    async def run_step(
        self,
        task_description: str = "",
        eval_config: Optional[Dict[str, Any]] = None,
    ) -> EvolutionStep:
        """
        Run a single evolution step.

        The step follows: LEARN → DESIGN → EXPERIMENT → ANALYZE

        Args:
            task_description: What we're trying to optimize
            eval_config: Configuration for evaluation

        Returns:
            EvolutionStep with results
        """
        # Increment step counter
        with self._step_lock:
            self._step_count += 1
            step_id = self._step_count

        step = EvolutionStep(
            step_id=step_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        start_time = time.perf_counter()

        self.emitter.emit(
            signal_type=SignalType.EXPERIMENT_START,
            payload={"step_id": step_id},
        )

        try:
            # ===== STEP 1: LEARN =====
            learn_start = time.perf_counter()

            # Sample parent nodes
            parent_nodes = self.shadow.sample_parents(n=self.config.sample_n)
            step.parents_sampled = len(parent_nodes)

            # Retrieve cognition items
            cognition_items = self._retrieve_cognition(parent_nodes)
            step.cognition_retrieved = len(cognition_items)

            step.learn_time_ms = (time.perf_counter() - learn_start) * 1000
            logger.debug(f"Step {step_id} LEARN: {len(parent_nodes)} parents, {len(cognition_items)} cognition items")

            # ===== STEP 2: DESIGN =====
            design_start = time.perf_counter()

            if self.researcher is None:
                raise RuntimeError("Researcher agent not configured")

            hypothesis = await self._run_in_executor(
                self.researcher.generate_hypothesis,
                task_description=task_description,
                parent_nodes=parent_nodes,
                cognition_items=cognition_items,
            )

            # Create node in shadow
            node = self.shadow.create_hypothesis(
                name=hypothesis.get("name", f"hypothesis_{step_id}"),
                motivation=hypothesis.get("motivation", ""),
                code=hypothesis.get("code", ""),
                parent_ids=[n.node_id for n in parent_nodes],
                domain=self.config.default_domain.value,
            )

            step.node_id = node.node_id
            step.node_name = node.name
            step.design_time_ms = (time.perf_counter() - design_start) * 1000

            self.emitter.emit(
                signal_type=SignalType.HYPOTHESIS_GENERATED,
                payload={
                    "step_id": step_id,
                    "node_id": node.node_id,
                    "name": node.name,
                },
            )

            logger.debug(f"Step {step_id} DESIGN: created {node.name}")

            # ===== STEP 3: EXPERIMENT =====
            experiment_start = time.perf_counter()

            if self.evaluator is None:
                raise RuntimeError("Evaluator agent not configured")

            eval_result = await self._run_in_executor(
                self.evaluator.evaluate,
                node=node,
                config=eval_config or {},
            )

            # Update node with evaluation result
            self.shadow.update_score(
                node_id=node.node_id,
                new_score=eval_result.get("score", 0.0),
                evaluation_result=eval_result,
            )

            step.score = eval_result.get("score", 0.0)
            step.experiment_time_ms = (time.perf_counter() - experiment_start) * 1000

            logger.debug(f"Step {step_id} EXPERIMENT: score={step.score:.4f}")

            # ===== STEP 4: ANALYZE =====
            analyze_start = time.perf_counter()

            if self.analyzer is None:
                raise RuntimeError("Analyzer agent not configured")

            # Find best parent for comparison
            best_parent = None
            if parent_nodes:
                best_parent = max(parent_nodes, key=lambda n: n.score)

            analysis = await self._run_in_executor(
                self.analyzer.analyze,
                node=node,
                eval_result=eval_result,
                best_parent=best_parent,
                task_description=task_description,
            )

            # Update node with analysis
            node.analysis = analysis.get("analysis", "")
            self.shadow.update(node)

            step.analyze_time_ms = (time.perf_counter() - analyze_start) * 1000

            logger.debug(f"Step {step_id} ANALYZE: complete")

            # Step successful
            step.success = True
            step.completed_at = datetime.now(timezone.utc).isoformat()
            step.total_time_ms = (time.perf_counter() - start_time) * 1000

            # Audit
            audit_log(
                action="evolution_step_complete",
                actor="system:orchestrator",
                node_id=node.node_id,
                description=f"Step {step_id} completed: {node.name} (score={step.score:.4f})",
                evidence=step.to_dict(),
            )

        except Exception as e:
            step.success = False
            step.error = str(e)
            step.completed_at = datetime.now(timezone.utc).isoformat()
            step.total_time_ms = (time.perf_counter() - start_time) * 1000

            logger.error(f"Step {step_id} failed: {e}")
            logger.debug(traceback.format_exc())

            audit_log(
                action="evolution_step_failed",
                actor="system:orchestrator",
                description=f"Step {step_id} failed: {e}",
                evidence={"error": str(e), "traceback": traceback.format_exc()},
            )

        # Emit completion signal
        self.emitter.emit(
            signal_type=SignalType.EXPERIMENT_END,
            payload={
                "step_id": step_id,
                "success": step.success,
                "score": step.score,
            },
        )

        # Store history
        self._step_history.append(step)
        if len(self._step_history) > 1000:
            self._step_history = self._step_history[-1000:]

        # Callbacks
        for callback in self._on_step_complete:
            try:
                callback(step)
            except Exception as e:
                logger.error(f"Step callback failed: {e}")

        return step

    async def _run_in_executor(self, func: Callable, **kwargs) -> Any:
        """Run a function in the thread pool executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(**kwargs)
        )

    def _retrieve_cognition(
        self,
        context_nodes: List[EvolutionNode],
    ) -> List[CognitionItem]:
        """
        Retrieve relevant cognition items.

        Uses node motivation/analysis to find relevant prior knowledge.
        """
        if not self._cognition_items:
            return []

        # Simple retrieval: return top items
        # In production, would use embedding similarity
        return self._cognition_items[:5]

    def _on_promotion_queued(self, node: EvolutionNode) -> None:
        """Handle node queued for promotion."""
        logger.info(f"Node queued for promotion: {node.node_id}")

        # Add to promotion queue
        self.promotion.queue_for_review(
            node=node,
            requested_by="ai:orchestrator",
            rationale=f"High score ({node.score:.4f}) and passed safety checks",
        )

        audit_log(
            action="promotion_queued",
            actor="ai:orchestrator",
            node_id=node.node_id,
            lane=Lane.PROMOTION,
            description=f"Queued {node.name} for human review",
        )

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_step_complete(self, callback: Callable[[EvolutionStep], None]) -> None:
        """Register callback for step completion."""
        self._on_step_complete.append(callback)

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_best_nodes(self, n: int = 10) -> List[EvolutionNode]:
        """Get best performing nodes from shadow."""
        return self.shadow.get_top_by_score(n)

    def get_pending_promotion(self) -> List[EvolutionNode]:
        """Get nodes awaiting human approval."""
        return self.promotion.get_pending_reviews()

    def get_production_nodes(self) -> List[EvolutionNode]:
        """Get all production nodes."""
        return self.production.get_active()

    def get_step_history(self, limit: int = 100) -> List[EvolutionStep]:
        """Get recent step history."""
        return self._step_history[-limit:]

    # =========================================================================
    # STATS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        steps = self._step_history

        successful = [s for s in steps if s.success]
        scores = [s.score for s in successful]

        return {
            "running": self._running,
            "total_steps": self._step_count,
            "successful_steps": len(successful),
            "failed_steps": len(steps) - len(successful),
            "success_rate": len(successful) / len(steps) if steps else 0,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "best_score": max(scores) if scores else 0,
            "shadow_size": len(self.shadow),
            "production_size": len(self.production),
            "promotion_queue_size": len(self.promotion),
            "sampler_stats": self.sampler.get_stats(),
        }


# Convenience function
def create_orchestrator(
    storage_dir: str = "data/evolution",
    **config_kwargs,
) -> EvolutionOrchestrator:
    """Create an orchestrator with the given configuration."""
    config = OrchestratorConfig(
        storage_dir=Path(storage_dir),
        **config_kwargs,
    )
    return EvolutionOrchestrator(config)
