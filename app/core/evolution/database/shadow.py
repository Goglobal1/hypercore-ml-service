"""
Shadow Store - HyperCore
========================

Shadow lane storage for AI-generated experimental nodes.

Key Characteristics:
- AI can freely add/update nodes here
- Experiments run in shadow before promotion
- No direct impact on production
- Nodes compete based on utility scoring

The shadow lane is where the evolution happens. AI generates
hypotheses, tests them, and the best performers get queued
for promotion.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Callable, TYPE_CHECKING

from .base import BaseEvolutionStore
from ..schemas import (
    Lane,
    EvolutionNode,
    ApprovalStatus,
    EscalationLevel,
    CapabilityTier,
    NodeId,
)

if TYPE_CHECKING:
    from ..sampling.ucb1 import UCB1Sampler

logger = logging.getLogger(__name__)


class ShadowStore(BaseEvolutionStore):
    """
    Shadow lane store - AI experimental playground.

    AI agents can freely create and modify nodes here.
    Nodes are evaluated, scored, and the best ones get
    promoted to production via the promotion queue.

    Features:
    - UCB1 sampling for parent selection (explore vs exploit)
    - Automatic scoring and ranking
    - Promotion queue integration
    - Experiment tracking
    """

    def __init__(
        self,
        storage_dir: Path,
        max_size: int = 10000,
        auto_promote_threshold: Optional[float] = None,
    ):
        """
        Initialize shadow store.

        Args:
            storage_dir: Directory for persistence
            max_size: Maximum nodes to keep (oldest/lowest evicted)
            auto_promote_threshold: Score threshold for auto-promotion queue
        """
        super().__init__(
            storage_dir=storage_dir,
            lane=Lane.SHADOW,
            max_size=max_size,
        )

        self.auto_promote_threshold = auto_promote_threshold

        # Sampler for parent selection (set by orchestrator)
        self._sampler: Optional["UCB1Sampler"] = None

        # Promotion queue callback
        self._promotion_callback: Optional[Callable[[EvolutionNode], None]] = None

    def can_add(self, node: EvolutionNode) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be added.

        Shadow is permissive - most nodes can be added.
        """
        # Cannot add already-promoted nodes
        if node.approval_status == ApprovalStatus.PROMOTED:
            return False, "Promoted nodes belong in production"

        # Cannot add rolled-back nodes
        if node.approval_status == ApprovalStatus.ROLLED_BACK:
            return False, "Rolled back nodes cannot be re-added"

        return True, None

    def can_update(self, node: EvolutionNode) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be updated.

        Shadow allows updates unless node is already promoted.
        """
        existing = self.get(node.node_id)
        if not existing:
            return False, "Node not found"

        if existing.approval_status == ApprovalStatus.PROMOTED:
            return False, "Cannot update promoted node in shadow"

        return True, None

    def can_remove(self, node_id: NodeId) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be removed.

        Shadow allows removal of any node.
        """
        return True, None

    # =========================================================================
    # SAMPLING (for parent selection in evolution)
    # =========================================================================

    def set_sampler(self, sampler: "UCB1Sampler") -> None:
        """Set the sampler for parent selection."""
        self._sampler = sampler

    def sample_parents(self, n: int = 3) -> List[EvolutionNode]:
        """
        Sample parent nodes for evolution.

        Uses UCB1 to balance exploration (try new parents)
        with exploitation (use proven parents).

        Args:
            n: Number of parents to sample

        Returns:
            List of sampled parent nodes
        """
        nodes = self.get_all()

        if not nodes:
            return []

        if self._sampler:
            selected = self._sampler.sample(nodes, n)
            # Update visit counts
            for node in selected:
                node.visit_count += 1
                self.update(node)
            return selected
        else:
            # Fallback: top by score
            nodes.sort(key=lambda x: x.score, reverse=True)
            return nodes[:n]

    # =========================================================================
    # EVALUATION & SCORING
    # =========================================================================

    def update_score(
        self,
        node_id: NodeId,
        new_score: float,
        evaluation_result: Optional[dict] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Update a node's score after evaluation.

        Args:
            node_id: Node to update
            new_score: New score from evaluation
            evaluation_result: Full evaluation result dict

        Returns:
            Tuple of (success, error_message)
        """
        node = self.get(node_id)
        if not node:
            return False, "Node not found"

        old_score = node.score
        node.score = new_score

        if evaluation_result:
            from ..schemas import EvaluationResult
            node.evaluation_result = EvaluationResult.from_dict(evaluation_result)

        # Update status
        node.approval_status = ApprovalStatus.EVALUATING

        success, error = self.update(node)

        if success:
            logger.info(f"Updated score for {node_id}: {old_score:.4f} -> {new_score:.4f}")

            # Check for auto-promotion
            self._check_auto_promote(node)

        return success, error

    def _check_auto_promote(self, node: EvolutionNode) -> None:
        """Check if node should be queued for promotion."""
        if self.auto_promote_threshold is None:
            return

        if node.score >= self.auto_promote_threshold:
            # Check safety
            if node.evaluation_result and node.evaluation_result.safety_passed:
                self._queue_for_promotion(node)

    def _queue_for_promotion(self, node: EvolutionNode) -> None:
        """Queue a node for promotion review."""
        node.approval_status = ApprovalStatus.PENDING_REVIEW

        # Set escalation level based on tier
        if CapabilityTier.requires_escalation(node.capability_tier):
            node.escalation_level = EscalationLevel.HARD
            node.escalation_reason = f"Tier {node.capability_tier} requires human approval"

        self.update(node)

        # Notify promotion queue
        if self._promotion_callback:
            try:
                self._promotion_callback(node)
            except Exception as e:
                logger.error(f"Promotion callback failed: {e}")

        logger.info(f"Queued {node.node_id} for promotion (score={node.score:.4f})")

    def set_promotion_callback(
        self,
        callback: Callable[[EvolutionNode], None],
    ) -> None:
        """Set callback for when nodes are queued for promotion."""
        self._promotion_callback = callback

    # =========================================================================
    # EXPERIMENT MANAGEMENT
    # =========================================================================

    def create_hypothesis(
        self,
        name: str,
        motivation: str,
        code: str,
        parent_ids: Optional[List[str]] = None,
        **metadata,
    ) -> EvolutionNode:
        """
        Create a new hypothesis node.

        Args:
            name: Name for the hypothesis
            motivation: Why this approach is being tried
            code: The code/algorithm
            parent_ids: Parent node IDs (for lineage)
            **metadata: Additional metadata

        Returns:
            The created node
        """
        from ..schemas import EvolutionNodeType, DeploymentDomain

        node = EvolutionNode(
            name=name,
            motivation=motivation,
            code=code,
            parent_ids=parent_ids or [],
            node_type=EvolutionNodeType.HYPOTHESIS,
            domain=DeploymentDomain(metadata.get("domain", "research")),
            approval_status=ApprovalStatus.DRAFT,
            metadata=metadata,
        )

        success, error = self.add(node)
        if not success:
            raise ValueError(f"Failed to create hypothesis: {error}")

        return node

    def get_pending_evaluation(self, limit: int = 10) -> List[EvolutionNode]:
        """Get nodes awaiting evaluation."""
        return [
            node for node in self.get_all()
            if node.approval_status == ApprovalStatus.DRAFT
        ][:limit]

    def get_pending_promotion(self) -> List[EvolutionNode]:
        """Get nodes queued for promotion."""
        return [
            node for node in self.get_all()
            if node.approval_status == ApprovalStatus.PENDING_REVIEW
        ]

    def get_top_candidates(self, n: int = 10) -> List[EvolutionNode]:
        """Get top scoring nodes that could be promoted."""
        evaluated = [
            node for node in self.get_all()
            if node.evaluation_result is not None
            and node.evaluation_result.safety_passed
        ]
        evaluated.sort(key=lambda x: x.score, reverse=True)
        return evaluated[:n]

    # =========================================================================
    # LINEAGE
    # =========================================================================

    def get_children(self, node_id: NodeId) -> List[EvolutionNode]:
        """Get all nodes that have this node as a parent."""
        return [
            node for node in self.get_all()
            if node_id in node.parent_ids
        ]

    def get_lineage(self, node_id: NodeId, depth: int = 5) -> List[EvolutionNode]:
        """
        Get the lineage (ancestors) of a node.

        Args:
            node_id: Node to trace back from
            depth: Maximum depth to trace

        Returns:
            List of ancestor nodes (closest first)
        """
        lineage = []
        current = self.get(node_id)

        for _ in range(depth):
            if not current or not current.parent_ids:
                break

            # Get first parent (primary lineage)
            parent_id = current.parent_ids[0]
            parent = self.get(parent_id)

            if parent:
                lineage.append(parent)
                current = parent
            else:
                break

        return lineage
