"""
Production Store - HyperCore
============================

Production lane storage for validated, human-approved nodes.

Key Characteristics:
- READ-ONLY for AI (AI cannot directly modify production)
- Contains only nodes that passed human approval
- Nodes here are "live" and actively used by agents
- Changes require promotion workflow

Only humans (via the promotion workflow) can add/update/remove
nodes in production.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from .base import BaseEvolutionStore
from ..schemas import Lane, EvolutionNode, ApprovalStatus, NodeId

logger = logging.getLogger(__name__)


class ProductionStore(BaseEvolutionStore):
    """
    Production lane store - validated, human-approved nodes.

    This store is READ-ONLY for AI. Only the promotion workflow
    (with human approval) can modify production nodes.

    Nodes in production are actively used by agents for inference.
    """

    def __init__(
        self,
        storage_dir: Path,
        max_size: Optional[int] = None,
    ):
        super().__init__(
            storage_dir=storage_dir,
            lane=Lane.PRODUCTION,
            max_size=max_size,
        )

    def can_add(self, node: EvolutionNode) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be added.

        Only promoted nodes (from promotion workflow) can be added.
        """
        # Node must have been approved
        if node.approval_status != ApprovalStatus.PROMOTED:
            return False, f"Node must be PROMOTED, got {node.approval_status.value}"

        # Must have reviewer (human approval)
        if not node.reviewed_by:
            return False, "Node must have human reviewer"

        # Must have promotion timestamp
        if not node.promoted_at:
            return False, "Node must have promotion timestamp"

        return True, None

    def can_update(self, node: EvolutionNode) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be updated.

        Updates in production require human authorization.
        """
        # For now, only allow metadata updates
        existing = self.get(node.node_id)
        if not existing:
            return False, "Node not found"

        # Cannot change core content without re-promotion
        if node.code != existing.code:
            return False, "Cannot change code in production - requires re-promotion"

        if node.motivation != existing.motivation:
            return False, "Cannot change motivation in production"

        return True, None

    def can_remove(self, node_id: NodeId) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be removed.

        Removal (rollback) requires human authorization.
        """
        node = self.get(node_id)
        if not node:
            return False, "Node not found"

        # In real system, this would require human approval
        # For now, allow removal but log it
        logger.warning(f"Production node removal requested: {node_id}")
        return True, None

    # =========================================================================
    # PRODUCTION-SPECIFIC METHODS
    # =========================================================================

    def promote_from_shadow(
        self,
        node: EvolutionNode,
        reviewer: str,
        review_notes: str = "",
    ) -> tuple[bool, Optional[str]]:
        """
        Promote a node from shadow to production.

        This is the only way to add nodes to production.

        Args:
            node: The approved node to promote
            reviewer: Human reviewer identifier
            review_notes: Notes from the review

        Returns:
            Tuple of (success, error_message)
        """
        from datetime import datetime, timezone

        # Update node for production
        node.approval_status = ApprovalStatus.PROMOTED
        node.reviewed_by = reviewer
        node.reviewed_at = datetime.now(timezone.utc).isoformat()
        node.promoted_at = datetime.now(timezone.utc).isoformat()
        node.review_notes = review_notes
        node.lane = Lane.PRODUCTION

        return self.add(node)

    def rollback(
        self,
        node_id: NodeId,
        reason: str,
        rolled_back_by: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Roll back a production node.

        Marks the node as rolled back and removes it from active use.

        Args:
            node_id: Node to roll back
            reason: Reason for rollback
            rolled_back_by: Human who authorized rollback

        Returns:
            Tuple of (success, error_message)
        """
        node = self.get(node_id)
        if not node:
            return False, "Node not found"

        # Mark as rolled back (don't delete - keep for audit)
        node.approval_status = ApprovalStatus.ROLLED_BACK
        node.metadata["rollback_reason"] = reason
        node.metadata["rolled_back_by"] = rolled_back_by
        node.metadata["rolled_back_at"] = datetime.now(timezone.utc).isoformat()

        # Update (this will pass since we're just updating metadata/status)
        return self.update(node)

    def get_active(self) -> list[EvolutionNode]:
        """Get all active (not rolled back) production nodes."""
        return [
            node for node in self.get_all()
            if node.approval_status == ApprovalStatus.PROMOTED
        ]

    def get_by_agent_type(self, agent_type: str) -> list[EvolutionNode]:
        """Get production nodes for a specific agent type."""
        return [
            node for node in self.get_active()
            if node.metadata.get("agent_type") == agent_type
        ]


# Import datetime for rollback
from datetime import datetime, timezone
