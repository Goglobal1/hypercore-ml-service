"""
Base Evolution Store - HyperCore
================================

Abstract base class for three-lane evolution stores.
Provides common functionality for node storage, retrieval,
and persistence.

Adapted from GAIR-NLP ASI-Evolve Database pattern.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timezone

from ..schemas import (
    Lane,
    EvolutionNode,
    AuditEntry,
    NodeId,
)

logger = logging.getLogger(__name__)


class BaseEvolutionStore(ABC):
    """
    Abstract base class for evolution node stores.

    Each lane (Production, Shadow, Promotion) extends this class
    with lane-specific behavior and constraints.

    Features:
    - Thread-safe CRUD operations
    - JSON persistence
    - Node indexing by various attributes
    - Event callbacks for node changes
    """

    def __init__(
        self,
        storage_dir: Path,
        lane: Lane,
        max_size: Optional[int] = None,
    ):
        """
        Initialize the store.

        Args:
            storage_dir: Directory for persistent storage
            lane: Which lane this store serves
            max_size: Maximum number of nodes (None = unlimited)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.lane = lane
        self.max_size = max_size

        # Node storage
        self._nodes: Dict[NodeId, EvolutionNode] = {}
        self._lock = threading.RLock()

        # Indexes
        self._by_name: Dict[str, List[NodeId]] = {}
        self._by_domain: Dict[str, List[NodeId]] = {}
        self._by_tier: Dict[int, List[NodeId]] = {}

        # Event callbacks
        self._on_add: List[Callable[[EvolutionNode], None]] = []
        self._on_update: List[Callable[[EvolutionNode, EvolutionNode], None]] = []
        self._on_remove: List[Callable[[EvolutionNode], None]] = []

        # Audit entries for this store
        self._audit_entries: List[AuditEntry] = []

        # Load persisted data
        self._load()

    # =========================================================================
    # ABSTRACT METHODS (lane-specific behavior)
    # =========================================================================

    @abstractmethod
    def can_add(self, node: EvolutionNode) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be added to this store.

        Returns:
            Tuple of (can_add, reason_if_not)
        """
        pass

    @abstractmethod
    def can_update(self, node: EvolutionNode) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be updated in this store.

        Returns:
            Tuple of (can_update, reason_if_not)
        """
        pass

    @abstractmethod
    def can_remove(self, node_id: NodeId) -> tuple[bool, Optional[str]]:
        """
        Check if a node can be removed from this store.

        Returns:
            Tuple of (can_remove, reason_if_not)
        """
        pass

    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================

    def add(self, node: EvolutionNode) -> tuple[bool, Optional[str]]:
        """
        Add a node to the store.

        Args:
            node: Node to add

        Returns:
            Tuple of (success, error_message)
        """
        can_add, reason = self.can_add(node)
        if not can_add:
            return False, reason

        with self._lock:
            # Check max size
            if self.max_size and len(self._nodes) >= self.max_size:
                self._evict_lowest_score()

            # Set lane
            node.lane = self.lane
            node.updated_at = datetime.now(timezone.utc).isoformat()

            # Store
            self._nodes[node.node_id] = node
            self._index_node(node)

            # Persist
            self._save()

        # Callbacks
        for callback in self._on_add:
            try:
                callback(node)
            except Exception as e:
                logger.error(f"Add callback failed: {e}")

        # Audit
        self._audit(
            action="node_added",
            node_id=node.node_id,
            description=f"Added node {node.name} to {self.lane.value}",
        )

        logger.debug(f"Added node {node.node_id} to {self.lane.value}")
        return True, None

    def get(self, node_id: NodeId) -> Optional[EvolutionNode]:
        """Get a node by ID."""
        with self._lock:
            return self._nodes.get(node_id)

    def get_all(self) -> List[EvolutionNode]:
        """Get all nodes in the store."""
        with self._lock:
            return list(self._nodes.values())

    def update(self, node: EvolutionNode) -> tuple[bool, Optional[str]]:
        """
        Update an existing node.

        Args:
            node: Node with updated values

        Returns:
            Tuple of (success, error_message)
        """
        can_update, reason = self.can_update(node)
        if not can_update:
            return False, reason

        with self._lock:
            if node.node_id not in self._nodes:
                return False, "Node not found"

            old_node = self._nodes[node.node_id]

            # Update timestamps
            node.updated_at = datetime.now(timezone.utc).isoformat()
            node.version = old_node.version + 1

            # Re-index
            self._unindex_node(old_node)
            self._nodes[node.node_id] = node
            self._index_node(node)

            # Persist
            self._save()

        # Callbacks
        for callback in self._on_update:
            try:
                callback(old_node, node)
            except Exception as e:
                logger.error(f"Update callback failed: {e}")

        # Audit
        self._audit(
            action="node_updated",
            node_id=node.node_id,
            description=f"Updated node {node.name}",
            previous_state=old_node.to_dict(),
            new_state=node.to_dict(),
        )

        return True, None

    def remove(self, node_id: NodeId) -> tuple[bool, Optional[str]]:
        """
        Remove a node from the store.

        Args:
            node_id: ID of node to remove

        Returns:
            Tuple of (success, error_message)
        """
        can_remove, reason = self.can_remove(node_id)
        if not can_remove:
            return False, reason

        with self._lock:
            if node_id not in self._nodes:
                return False, "Node not found"

            node = self._nodes.pop(node_id)
            self._unindex_node(node)
            self._save()

        # Callbacks
        for callback in self._on_remove:
            try:
                callback(node)
            except Exception as e:
                logger.error(f"Remove callback failed: {e}")

        # Audit
        self._audit(
            action="node_removed",
            node_id=node_id,
            description=f"Removed node {node.name} from {self.lane.value}",
        )

        return True, None

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_by_name(self, name: str) -> List[EvolutionNode]:
        """Get all nodes with a given name."""
        with self._lock:
            node_ids = self._by_name.get(name, [])
            return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_by_domain(self, domain: str) -> List[EvolutionNode]:
        """Get all nodes in a given domain."""
        with self._lock:
            node_ids = self._by_domain.get(domain, [])
            return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_by_tier(self, tier: int) -> List[EvolutionNode]:
        """Get all nodes at a given capability tier."""
        with self._lock:
            node_ids = self._by_tier.get(tier, [])
            return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_top_by_score(self, n: int = 10) -> List[EvolutionNode]:
        """Get the top N nodes by score."""
        with self._lock:
            nodes = list(self._nodes.values())
            nodes.sort(key=lambda x: x.score, reverse=True)
            return nodes[:n]

    def get_recent(self, n: int = 10) -> List[EvolutionNode]:
        """Get the N most recently updated nodes."""
        with self._lock:
            nodes = list(self._nodes.values())
            nodes.sort(key=lambda x: x.updated_at, reverse=True)
            return nodes[:n]

    def search(
        self,
        name_contains: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        domain: Optional[str] = None,
        min_tier: Optional[int] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[EvolutionNode]:
        """
        Search nodes with multiple criteria.

        Args:
            name_contains: Substring to match in name
            min_score: Minimum score
            max_score: Maximum score
            domain: Domain filter
            min_tier: Minimum capability tier
            tags: Must have all these tags
            limit: Maximum results

        Returns:
            List of matching nodes
        """
        with self._lock:
            results = []

            for node in self._nodes.values():
                # Apply filters
                if name_contains and name_contains.lower() not in node.name.lower():
                    continue
                if min_score is not None and node.score < min_score:
                    continue
                if max_score is not None and node.score > max_score:
                    continue
                if domain and node.domain.value != domain:
                    continue
                if min_tier is not None and node.capability_tier < min_tier:
                    continue
                if tags:
                    if not all(tag in node.tags for tag in tags):
                        continue

                results.append(node)

                if len(results) >= limit:
                    break

            return results

    # =========================================================================
    # INDEXING
    # =========================================================================

    def _index_node(self, node: EvolutionNode) -> None:
        """Add node to all indexes."""
        # By name
        if node.name not in self._by_name:
            self._by_name[node.name] = []
        self._by_name[node.name].append(node.node_id)

        # By domain
        domain = node.domain.value
        if domain not in self._by_domain:
            self._by_domain[domain] = []
        self._by_domain[domain].append(node.node_id)

        # By tier
        tier = int(node.capability_tier)
        if tier not in self._by_tier:
            self._by_tier[tier] = []
        self._by_tier[tier].append(node.node_id)

    def _unindex_node(self, node: EvolutionNode) -> None:
        """Remove node from all indexes."""
        # By name
        if node.name in self._by_name:
            self._by_name[node.name] = [
                nid for nid in self._by_name[node.name]
                if nid != node.node_id
            ]

        # By domain
        domain = node.domain.value
        if domain in self._by_domain:
            self._by_domain[domain] = [
                nid for nid in self._by_domain[domain]
                if nid != node.node_id
            ]

        # By tier
        tier = int(node.capability_tier)
        if tier in self._by_tier:
            self._by_tier[tier] = [
                nid for nid in self._by_tier[tier]
                if nid != node.node_id
            ]

    # =========================================================================
    # EVICTION
    # =========================================================================

    def _evict_lowest_score(self) -> None:
        """Remove the lowest-scoring node to make room."""
        if not self._nodes:
            return

        # Find lowest score
        lowest_id = min(self._nodes.keys(), key=lambda nid: self._nodes[nid].score)
        lowest_node = self._nodes.pop(lowest_id)
        self._unindex_node(lowest_node)

        logger.info(f"Evicted node {lowest_id} (score={lowest_node.score})")

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save(self) -> None:
        """Save store to disk."""
        data_file = self.storage_dir / f"{self.lane.value}_nodes.json"

        data = {
            "lane": self.lane.value,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "node_count": len(self._nodes),
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
        }

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        """Load store from disk."""
        data_file = self.storage_dir / f"{self.lane.value}_nodes.json"

        if not data_file.exists():
            return

        try:
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for node_id, node_data in data.get("nodes", {}).items():
                node = EvolutionNode.from_dict(node_data)
                self._nodes[node_id] = node
                self._index_node(node)

            logger.info(f"Loaded {len(self._nodes)} nodes from {self.lane.value}")

        except Exception as e:
            logger.error(f"Failed to load {self.lane.value} store: {e}")

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_add(self, callback: Callable[[EvolutionNode], None]) -> None:
        """Register callback for node additions."""
        self._on_add.append(callback)

    def on_update(self, callback: Callable[[EvolutionNode, EvolutionNode], None]) -> None:
        """Register callback for node updates."""
        self._on_update.append(callback)

    def on_remove(self, callback: Callable[[EvolutionNode], None]) -> None:
        """Register callback for node removals."""
        self._on_remove.append(callback)

    # =========================================================================
    # AUDIT
    # =========================================================================

    def _audit(
        self,
        action: str,
        node_id: Optional[str] = None,
        description: str = "",
        previous_state: Optional[Dict] = None,
        new_state: Optional[Dict] = None,
    ) -> None:
        """Create an audit entry."""
        entry = AuditEntry(
            action=action,
            actor=f"store:{self.lane.value}",
            node_id=node_id,
            lane=self.lane.value,
            description=description,
            previous_state=previous_state,
            new_state=new_state,
        )
        self._audit_entries.append(entry)

        # Keep only recent
        if len(self._audit_entries) > 1000:
            self._audit_entries = self._audit_entries[-1000:]

    def get_audit_entries(self, limit: int = 100) -> List[AuditEntry]:
        """Get recent audit entries."""
        return self._audit_entries[-limit:]

    # =========================================================================
    # STATS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            scores = [n.score for n in self._nodes.values()]

            return {
                "lane": self.lane.value,
                "node_count": len(self._nodes),
                "max_size": self.max_size,
                "domains": {d: len(ids) for d, ids in self._by_domain.items()},
                "tiers": {t: len(ids) for t, ids in self._by_tier.items()},
                "score_min": min(scores) if scores else None,
                "score_max": max(scores) if scores else None,
                "score_avg": sum(scores) / len(scores) if scores else None,
            }

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: NodeId) -> bool:
        return node_id in self._nodes
