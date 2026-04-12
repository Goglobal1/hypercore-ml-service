"""
Audit Trail - HyperCore
=======================

FDA-compliant immutable audit logging for the evolution system.

Features:
- Immutable append-only log
- Cryptographic integrity verification (SHA-256 chain)
- Export for regulatory submission
- Query and filtering capabilities

Every significant action in the evolution system is logged here,
creating a complete audit trail for regulatory compliance.
"""

from __future__ import annotations
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import gzip

from ..schemas import AuditEntry, Lane, NodeId

logger = logging.getLogger(__name__)


class AuditTrail:
    """
    FDA-compliant immutable audit trail.

    Features:
    - Append-only log with cryptographic chaining
    - Automatic persistence to disk
    - Integrity verification
    - Export capabilities

    Each entry includes a checksum that chains to the previous entry,
    making tampering detectable.
    """

    def __init__(
        self,
        storage_dir: Path,
        max_memory_entries: int = 10000,
        auto_archive_threshold: int = 100000,
    ):
        """
        Initialize audit trail.

        Args:
            storage_dir: Directory for persistence
            max_memory_entries: Max entries to keep in memory
            auto_archive_threshold: Entries before auto-archiving
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_entries = max_memory_entries
        self.auto_archive_threshold = auto_archive_threshold

        self._entries: List[AuditEntry] = []
        self._lock = threading.RLock()
        self._last_checksum: Optional[str] = None
        self._entry_count = 0

        # Event callbacks
        self._on_entry: List[Callable[[AuditEntry], None]] = []

        # Load existing entries
        self._load()

    def log(
        self,
        action: str,
        actor: str,
        description: str = "",
        node_id: Optional[str] = None,
        lane: Optional[Lane] = None,
        rationale: str = "",
        evidence: Optional[Dict[str, Any]] = None,
        previous_state: Optional[Dict[str, Any]] = None,
        new_state: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log an audit entry.

        Args:
            action: Action type (e.g., "node_created", "approval_granted")
            actor: Who performed the action (e.g., "ai:researcher", "human:user@email.com")
            description: Human-readable description
            node_id: Related node ID if applicable
            lane: Which lane the action occurred in
            rationale: Why this action was taken
            evidence: Supporting evidence/data
            previous_state: State before the action
            new_state: State after the action

        Returns:
            The created AuditEntry
        """
        entry = AuditEntry(
            action=action,
            actor=actor,
            description=description,
            node_id=node_id,
            lane=lane.value if lane else None,
            rationale=rationale,
            evidence=evidence or {},
            previous_state=previous_state,
            new_state=new_state,
        )

        # Calculate checksum (chains to previous)
        entry.checksum = self._calculate_checksum(entry)

        with self._lock:
            self._entries.append(entry)
            self._last_checksum = entry.checksum
            self._entry_count += 1

            # Trim memory if needed
            if len(self._entries) > self.max_memory_entries:
                self._entries = self._entries[-self.max_memory_entries:]

            # Auto-archive if threshold reached
            if self._entry_count >= self.auto_archive_threshold:
                self._archive()

            # Persist
            self._save_recent()

        # Callbacks
        for callback in self._on_entry:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"Audit callback failed: {e}")

        return entry

    def _calculate_checksum(self, entry: AuditEntry) -> str:
        """Calculate SHA-256 checksum chaining to previous entry."""
        # Include previous checksum in chain
        chain_data = f"{self._last_checksum or 'GENESIS'}"

        # Include entry data
        entry_data = json.dumps({
            "entry_id": entry.entry_id,
            "timestamp": entry.timestamp,
            "action": entry.action,
            "actor": entry.actor,
            "node_id": entry.node_id,
            "description": entry.description,
        }, sort_keys=True)

        combined = f"{chain_data}:{entry_data}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def verify_integrity(self) -> tuple[bool, Optional[str]]:
        """
        Verify the integrity of the audit trail.

        Returns:
            Tuple of (is_valid, error_message)
        """
        with self._lock:
            entries = self._load_all_entries()

        if not entries:
            return True, None

        prev_checksum = None

        for i, entry in enumerate(entries):
            # Recalculate expected checksum
            chain_data = f"{prev_checksum or 'GENESIS'}"
            entry_data = json.dumps({
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp,
                "action": entry.action,
                "actor": entry.actor,
                "node_id": entry.node_id,
                "description": entry.description,
            }, sort_keys=True)

            combined = f"{chain_data}:{entry_data}"
            expected = hashlib.sha256(combined.encode()).hexdigest()

            if entry.checksum != expected:
                return False, f"Integrity violation at entry {i}: {entry.entry_id}"

            prev_checksum = entry.checksum

        return True, None

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_recent(self, limit: int = 100) -> List[AuditEntry]:
        """Get most recent entries."""
        with self._lock:
            return self._entries[-limit:]

    def get_by_node(self, node_id: NodeId) -> List[AuditEntry]:
        """Get all entries for a specific node."""
        with self._lock:
            return [e for e in self._entries if e.node_id == node_id]

    def get_by_actor(self, actor: str) -> List[AuditEntry]:
        """Get all entries by a specific actor."""
        with self._lock:
            return [e for e in self._entries if e.actor == actor]

    def get_by_action(self, action: str) -> List[AuditEntry]:
        """Get all entries of a specific action type."""
        with self._lock:
            return [e for e in self._entries if e.action == action]

    def get_by_lane(self, lane: Lane) -> List[AuditEntry]:
        """Get all entries in a specific lane."""
        with self._lock:
            return [e for e in self._entries if e.lane == lane.value]

    def get_in_timerange(
        self,
        start: datetime,
        end: datetime,
    ) -> List[AuditEntry]:
        """Get entries within a time range."""
        with self._lock:
            results = []
            for entry in self._entries:
                try:
                    ts = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
                    if start <= ts <= end:
                        results.append(entry)
                except:
                    pass
            return results

    def search(
        self,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        node_id: Optional[str] = None,
        lane: Optional[str] = None,
        description_contains: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Search entries with multiple criteria.

        Args:
            action: Filter by action type
            actor: Filter by actor (supports prefix match with *)
            node_id: Filter by node ID
            lane: Filter by lane
            description_contains: Substring match in description
            limit: Maximum results

        Returns:
            List of matching entries
        """
        with self._lock:
            results = []

            for entry in reversed(self._entries):  # Newest first
                if action and entry.action != action:
                    continue

                if actor:
                    if actor.endswith("*"):
                        if not entry.actor.startswith(actor[:-1]):
                            continue
                    elif entry.actor != actor:
                        continue

                if node_id and entry.node_id != node_id:
                    continue

                if lane and entry.lane != lane:
                    continue

                if description_contains:
                    if description_contains.lower() not in entry.description.lower():
                        continue

                results.append(entry)

                if len(results) >= limit:
                    break

            return results

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save_recent(self) -> None:
        """Save recent entries to disk."""
        recent_file = self.storage_dir / "recent.json"

        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": self._entry_count,
            "last_checksum": self._last_checksum,
            "entries": [e.to_dict() for e in self._entries[-1000:]],
        }

        with open(recent_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        """Load entries from disk."""
        recent_file = self.storage_dir / "recent.json"

        if not recent_file.exists():
            return

        try:
            with open(recent_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._entry_count = data.get("entry_count", 0)
            self._last_checksum = data.get("last_checksum")

            for entry_data in data.get("entries", []):
                entry = AuditEntry.from_dict(entry_data)
                self._entries.append(entry)

            logger.info(f"Loaded {len(self._entries)} audit entries")

        except Exception as e:
            logger.error(f"Failed to load audit trail: {e}")

    def _load_all_entries(self) -> List[AuditEntry]:
        """Load all entries including archives."""
        entries = []

        # Load archives
        for archive_file in sorted(self.storage_dir.glob("archive_*.json.gz")):
            try:
                with gzip.open(archive_file, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                for entry_data in data.get("entries", []):
                    entries.append(AuditEntry.from_dict(entry_data))
            except Exception as e:
                logger.error(f"Failed to load archive {archive_file}: {e}")

        # Add recent
        entries.extend(self._entries)

        return entries

    def _archive(self) -> None:
        """Archive old entries to compressed file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_file = self.storage_dir / f"archive_{timestamp}.json.gz"

        # Archive all but recent entries
        to_archive = self._entries[:-1000]
        self._entries = self._entries[-1000:]

        if not to_archive:
            return

        data = {
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(to_archive),
            "entries": [e.to_dict() for e in to_archive],
        }

        with gzip.open(archive_file, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        logger.info(f"Archived {len(to_archive)} entries to {archive_file}")

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_for_regulatory(
        self,
        output_path: Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_evidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Export audit trail for regulatory submission.

        Creates a comprehensive export with integrity verification.

        Args:
            output_path: Where to write the export
            start_date: Start of date range (None = all)
            end_date: End of date range (None = all)
            include_evidence: Include full evidence in export

        Returns:
            Export metadata
        """
        # Load all entries
        all_entries = self._load_all_entries()

        # Filter by date if specified
        if start_date or end_date:
            filtered = []
            for entry in all_entries:
                try:
                    ts = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
                    if start_date and ts < start_date:
                        continue
                    if end_date and ts > end_date:
                        continue
                    filtered.append(entry)
                except:
                    filtered.append(entry)  # Include if can't parse
            all_entries = filtered

        # Verify integrity
        is_valid, error = self.verify_integrity()

        # Prepare export
        export_data = {
            "export_metadata": {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "exported_by": "hypercore_evolution_system",
                "version": "1.0.0",
                "entry_count": len(all_entries),
                "date_range": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None,
                },
                "integrity_verified": is_valid,
                "integrity_error": error,
            },
            "entries": [],
        }

        for entry in all_entries:
            entry_dict = entry.to_dict()
            if not include_evidence:
                entry_dict.pop("evidence", None)
                entry_dict.pop("previous_state", None)
                entry_dict.pop("new_state", None)
            export_data["entries"].append(entry_dict)

        # Calculate export checksum
        export_json = json.dumps(export_data, sort_keys=True)
        export_checksum = hashlib.sha256(export_json.encode()).hexdigest()
        export_data["export_metadata"]["export_checksum"] = export_checksum

        # Write export
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(all_entries)} entries to {output_path}")

        return export_data["export_metadata"]

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_entry(self, callback: Callable[[AuditEntry], None]) -> None:
        """Register callback for new entries."""
        self._on_entry.append(callback)

    # =========================================================================
    # STATS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get audit trail statistics."""
        with self._lock:
            # Count by action
            by_action = {}
            for entry in self._entries:
                by_action[entry.action] = by_action.get(entry.action, 0) + 1

            # Count by actor type
            by_actor_type = {"ai": 0, "human": 0, "system": 0}
            for entry in self._entries:
                if entry.actor.startswith("ai:"):
                    by_actor_type["ai"] += 1
                elif entry.actor.startswith("human:"):
                    by_actor_type["human"] += 1
                else:
                    by_actor_type["system"] += 1

            # Verify integrity
            is_valid, _ = self.verify_integrity()

            return {
                "total_entries": self._entry_count,
                "memory_entries": len(self._entries),
                "last_checksum": self._last_checksum[:16] + "..." if self._last_checksum else None,
                "integrity_valid": is_valid,
                "by_action": by_action,
                "by_actor_type": by_actor_type,
            }


# Global audit trail instance
_audit_instance: Optional[AuditTrail] = None


def get_audit_trail(storage_dir: Optional[Path] = None) -> AuditTrail:
    """Get or create the global audit trail instance."""
    global _audit_instance

    if _audit_instance is None:
        if storage_dir is None:
            storage_dir = Path("data/evolution/audit")
        _audit_instance = AuditTrail(storage_dir)

    return _audit_instance


def audit_log(
    action: str,
    actor: str,
    **kwargs,
) -> AuditEntry:
    """
    Convenience function to log an audit entry.

    Usage:
        from app.core.evolution.audit import audit_log

        audit_log(
            action="node_created",
            actor="ai:researcher",
            description="Created new hypothesis",
            node_id="abc123"
        )
    """
    trail = get_audit_trail()
    return trail.log(action, actor, **kwargs)
