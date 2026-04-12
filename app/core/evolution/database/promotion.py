"""
Promotion Queue - HyperCore
===========================

Promotion lane for human approval workflow.

Key Characteristics:
- Nodes queued here await human review
- Humans approve or reject for production
- Tracks review status and escalations
- Provides approval workflow management

This is the bridge between shadow experiments and production.
Human oversight ensures safety and quality.
"""

from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from .base import BaseEvolutionStore
from ..schemas import (
    Lane,
    EvolutionNode,
    ApprovalStatus,
    EscalationLevel,
    PromotionRequest,
    NodeId,
)

logger = logging.getLogger(__name__)


class PromotionQueue(BaseEvolutionStore):
    """
    Promotion queue - human approval workflow.

    Nodes from shadow are queued here for human review.
    Humans can approve (promote to production) or reject.

    Features:
    - Priority queue based on score and escalation
    - Escalation tracking for high-tier nodes
    - Review workflow management
    - Audit trail for all decisions
    """

    def __init__(
        self,
        storage_dir: Path,
        max_queue_size: int = 1000,
    ):
        """
        Initialize promotion queue.

        Args:
            storage_dir: Directory for persistence
            max_queue_size: Maximum items in queue
        """
        super().__init__(
            storage_dir=storage_dir,
            lane=Lane.PROMOTION,
            max_size=max_queue_size,
        )

        # Promotion requests (separate from nodes)
        self._requests: Dict[str, PromotionRequest] = {}
        self._load_requests()

    def can_add(self, node: EvolutionNode) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be added to the queue.

        Only nodes pending review can be added.
        """
        if node.approval_status not in (
            ApprovalStatus.PENDING_REVIEW,
            ApprovalStatus.UNDER_REVIEW,
        ):
            return False, f"Node must be pending review, got {node.approval_status.value}"

        # Must have evaluation
        if node.evaluation_result is None:
            return False, "Node must be evaluated before promotion"

        # Must pass safety
        if not node.evaluation_result.safety_passed:
            return False, "Node failed safety evaluation"

        return True, None

    def can_update(self, node: EvolutionNode) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be updated.

        Allow updates for review workflow.
        """
        return True, None

    def can_remove(self, node_id: NodeId) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be removed.

        Allow removal after approval/rejection.
        """
        return True, None

    # =========================================================================
    # QUEUE MANAGEMENT
    # =========================================================================

    def queue_for_review(
        self,
        node: EvolutionNode,
        requested_by: str = "ai:evolution",
        rationale: str = "",
    ) -> tuple[bool, Optional[str]]:
        """
        Queue a node for human review.

        Args:
            node: Node to queue
            requested_by: Who requested promotion
            rationale: Why this should be promoted

        Returns:
            Tuple of (success, error_message)
        """
        # Update node status
        node.approval_status = ApprovalStatus.PENDING_REVIEW
        node.lane = Lane.PROMOTION

        # Create promotion request
        request = PromotionRequest(
            node_id=node.node_id,
            requested_by=requested_by,
            rationale=rationale,
            evidence_summary=self._generate_evidence_summary(node),
            risk_assessment=self._generate_risk_assessment(node),
            escalation_level=node.escalation_level,
        )

        # Add to queue
        success, error = self.add(node)
        if success:
            self._requests[request.request_id] = request
            self._save_requests()
            logger.info(f"Queued {node.node_id} for review (request: {request.request_id})")

        return success, error

    def _generate_evidence_summary(self, node: EvolutionNode) -> str:
        """Generate evidence summary for review."""
        parts = []

        if node.evaluation_result:
            parts.append(f"Score: {node.score:.4f}")
            parts.append(f"Safety: {'PASSED' if node.evaluation_result.safety_passed else 'FAILED'}")

            if node.evaluation_result.metrics:
                metrics = ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in list(node.evaluation_result.metrics.items())[:5]
                )
                parts.append(f"Metrics: {metrics}")

        if node.utility_breakdown:
            parts.append(f"Utility: {node.utility_breakdown.combined_score():.4f}")

        parts.append(f"Tier: {node.capability_tier}")
        parts.append(f"Parents: {len(node.parent_ids)}")

        return " | ".join(parts)

    def _generate_risk_assessment(self, node: EvolutionNode) -> str:
        """Generate risk assessment for review."""
        risks = []

        # High tier risk
        if node.capability_tier >= 7:
            risks.append(f"HIGH AUTONOMY: Tier {node.capability_tier}")

        # Escalation
        if node.escalation_level in (EscalationLevel.HARD, EscalationLevel.CRITICAL):
            risks.append(f"ESCALATION: {node.escalation_level.value}")

        # Safety warnings
        if node.evaluation_result and node.evaluation_result.safety_warnings:
            warnings = ", ".join(node.evaluation_result.safety_warnings[:3])
            risks.append(f"WARNINGS: {warnings}")

        # No parents (novel approach)
        if not node.parent_ids:
            risks.append("NOVEL: No parent lineage")

        return " | ".join(risks) if risks else "Low risk"

    # =========================================================================
    # REVIEW WORKFLOW
    # =========================================================================

    def get_pending_reviews(
        self,
        limit: int = 20,
        include_escalated: bool = True,
    ) -> List[EvolutionNode]:
        """
        Get nodes pending review.

        Returns prioritized list with escalated items first.
        """
        pending = [
            node for node in self.get_all()
            if node.approval_status == ApprovalStatus.PENDING_REVIEW
        ]

        # Sort by escalation level (critical first), then score
        def priority(node):
            escalation_order = {
                EscalationLevel.CRITICAL: 0,
                EscalationLevel.HARD: 1,
                EscalationLevel.SOFT: 2,
                EscalationLevel.NONE: 3,
            }
            return (escalation_order.get(node.escalation_level, 3), -node.score)

        pending.sort(key=priority)

        if not include_escalated:
            pending = [n for n in pending if n.escalation_level == EscalationLevel.NONE]

        return pending[:limit]

    def start_review(
        self,
        node_id: NodeId,
        reviewer: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Mark a node as under review.

        Args:
            node_id: Node to review
            reviewer: Who is reviewing

        Returns:
            Tuple of (success, error_message)
        """
        node = self.get(node_id)
        if not node:
            return False, "Node not found"

        node.approval_status = ApprovalStatus.UNDER_REVIEW
        node.reviewed_by = reviewer
        node.metadata["review_started_at"] = datetime.now(timezone.utc).isoformat()

        return self.update(node)

    def approve(
        self,
        node_id: NodeId,
        reviewer: str,
        notes: str = "",
    ) -> tuple[EvolutionNode, Optional[str]]:
        """
        Approve a node for production.

        Args:
            node_id: Node to approve
            reviewer: Who is approving
            notes: Review notes

        Returns:
            Tuple of (approved_node, error_message)
        """
        node = self.get(node_id)
        if not node:
            return None, "Node not found"

        # Check escalation requirements
        if node.escalation_level == EscalationLevel.CRITICAL:
            # Would need multiple approvers in production
            logger.warning(f"Critical escalation approved by single reviewer: {reviewer}")

        # Update node
        node.approval_status = ApprovalStatus.APPROVED
        node.reviewed_by = reviewer
        node.reviewed_at = datetime.now(timezone.utc).isoformat()
        node.review_notes = notes

        success, error = self.update(node)
        if not success:
            return None, error

        # Update request
        for request in self._requests.values():
            if request.node_id == node_id:
                request.status = ApprovalStatus.APPROVED
                request.reviewer = reviewer
                request.reviewed_at = datetime.now(timezone.utc).isoformat()
                request.review_decision = "approved"
                request.review_notes = notes
                break

        self._save_requests()

        logger.info(f"Approved {node_id} by {reviewer}")
        return node, None

    def reject(
        self,
        node_id: NodeId,
        reviewer: str,
        reason: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Reject a node.

        Args:
            node_id: Node to reject
            reviewer: Who is rejecting
            reason: Reason for rejection

        Returns:
            Tuple of (success, error_message)
        """
        node = self.get(node_id)
        if not node:
            return False, "Node not found"

        # Update node
        node.approval_status = ApprovalStatus.REJECTED
        node.reviewed_by = reviewer
        node.reviewed_at = datetime.now(timezone.utc).isoformat()
        node.review_notes = reason

        success, error = self.update(node)

        # Update request
        for request in self._requests.values():
            if request.node_id == node_id:
                request.status = ApprovalStatus.REJECTED
                request.reviewer = reviewer
                request.reviewed_at = datetime.now(timezone.utc).isoformat()
                request.review_decision = "rejected"
                request.review_notes = reason
                break

        self._save_requests()

        logger.info(f"Rejected {node_id} by {reviewer}: {reason}")
        return success, error

    def request_changes(
        self,
        node_id: NodeId,
        reviewer: str,
        requested_changes: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Request changes before approval.

        Node goes back to shadow for iteration.

        Args:
            node_id: Node needing changes
            reviewer: Who is requesting
            requested_changes: What needs to change

        Returns:
            Tuple of (success, error_message)
        """
        node = self.get(node_id)
        if not node:
            return False, "Node not found"

        # Update status
        node.approval_status = ApprovalStatus.DRAFT  # Back to draft
        node.review_notes = f"Changes requested: {requested_changes}"
        node.metadata["changes_requested_by"] = reviewer
        node.metadata["changes_requested_at"] = datetime.now(timezone.utc).isoformat()

        # Remove from promotion queue
        self.remove(node_id)

        logger.info(f"Changes requested for {node_id}: {requested_changes}")
        return True, None

    # =========================================================================
    # ESCALATION
    # =========================================================================

    def get_escalated(self) -> List[EvolutionNode]:
        """Get all escalated items requiring senior review."""
        return [
            node for node in self.get_all()
            if node.escalation_level in (EscalationLevel.HARD, EscalationLevel.CRITICAL)
            and node.approval_status in (ApprovalStatus.PENDING_REVIEW, ApprovalStatus.UNDER_REVIEW)
        ]

    def escalate(
        self,
        node_id: NodeId,
        level: EscalationLevel,
        reason: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Escalate a node to higher review level.

        Args:
            node_id: Node to escalate
            level: New escalation level
            reason: Reason for escalation

        Returns:
            Tuple of (success, error_message)
        """
        node = self.get(node_id)
        if not node:
            return False, "Node not found"

        node.escalation_level = level
        node.escalation_reason = reason

        return self.update(node)

    # =========================================================================
    # REQUEST MANAGEMENT
    # =========================================================================

    def get_request(self, request_id: str) -> Optional[PromotionRequest]:
        """Get a promotion request by ID."""
        return self._requests.get(request_id)

    def get_request_for_node(self, node_id: NodeId) -> Optional[PromotionRequest]:
        """Get the promotion request for a node."""
        for request in self._requests.values():
            if request.node_id == node_id:
                return request
        return None

    def get_all_requests(self) -> List[PromotionRequest]:
        """Get all promotion requests."""
        return list(self._requests.values())

    def _save_requests(self) -> None:
        """Save requests to disk."""
        import json
        requests_file = self.storage_dir / "promotion_requests.json"

        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "requests": {rid: r.to_dict() for rid, r in self._requests.items()},
        }

        with open(requests_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_requests(self) -> None:
        """Load requests from disk."""
        import json
        requests_file = self.storage_dir / "promotion_requests.json"

        if not requests_file.exists():
            return

        try:
            with open(requests_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for request_id, request_data in data.get("requests", {}).items():
                self._requests[request_id] = PromotionRequest.from_dict(request_data)

        except Exception as e:
            logger.error(f"Failed to load promotion requests: {e}")

    # =========================================================================
    # STATS
    # =========================================================================

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        nodes = self.get_all()

        by_status = {}
        for node in nodes:
            status = node.approval_status.value
            by_status[status] = by_status.get(status, 0) + 1

        by_escalation = {}
        for node in nodes:
            level = node.escalation_level.value
            by_escalation[level] = by_escalation.get(level, 0) + 1

        return {
            **self.get_stats(),
            "by_status": by_status,
            "by_escalation": by_escalation,
            "pending_count": by_status.get("pending_review", 0),
            "under_review_count": by_status.get("under_review", 0),
            "total_requests": len(self._requests),
        }
