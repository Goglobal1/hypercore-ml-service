"""
Alert System Storage Layer - Unified Implementation
Abstract storage interface with in-memory, PostgreSQL, and Redis implementations.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import json
import logging


def get_redis_url() -> str:
    """Get Redis URL from environment variable (Railway provides REDIS_URL)."""
    return os.environ.get('REDIS_URL', 'redis://localhost:6379')

from .models import (
    PatientState,
    EpisodeState,
    AlertEvent,
    AcknowledgmentRecord,
    ClinicalState,
    EventType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ABSTRACT STORAGE INTERFACE
# =============================================================================

class StorageBackend(ABC):
    """Abstract storage interface for alert system persistence."""

    @abstractmethod
    def get_patient_state(self, patient_id: str, risk_domain: str) -> Optional[PatientState]:
        """Retrieve last known state for patient + domain."""
        pass

    @abstractmethod
    def save_patient_state(self, state: PatientState) -> None:
        """Persist patient state."""
        pass

    @abstractmethod
    def get_episode(self, episode_id: str) -> Optional[EpisodeState]:
        """Get episode by ID."""
        pass

    @abstractmethod
    def save_episode(self, episode: EpisodeState) -> None:
        """Save episode state."""
        pass

    @abstractmethod
    def get_open_episodes(self, patient_id: Optional[str] = None) -> List[EpisodeState]:
        """Get all open episodes, optionally filtered by patient."""
        pass

    @abstractmethod
    def log_event(self, event: AlertEvent) -> None:
        """Append event to audit log."""
        pass

    @abstractmethod
    def get_events(
        self,
        patient_id: Optional[str] = None,
        risk_domain: Optional[str] = None,
        event_type: Optional[EventType] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AlertEvent]:
        """Query audit log with filters."""
        pass

    @abstractmethod
    def get_event_by_id(self, event_id: str) -> Optional[AlertEvent]:
        """Get specific event by ID."""
        pass

    @abstractmethod
    def save_acknowledgment(self, ack: AcknowledgmentRecord) -> None:
        """Save acknowledgment record."""
        pass

    @abstractmethod
    def get_acknowledgments(
        self,
        patient_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AcknowledgmentRecord]:
        """Get acknowledgment records with filters."""
        pass

    @abstractmethod
    def clear_patient(self, patient_id: str) -> None:
        """Clear all state for a patient (for testing)."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass


# =============================================================================
# IN-MEMORY IMPLEMENTATION
# =============================================================================

class InMemoryStorage(StorageBackend):
    """
    In-memory storage for development and testing.

    WARNING: Data is lost on restart. Use PostgreSQL for production.
    """

    def __init__(self, max_events: int = 100000, max_acks: int = 10000):
        # patient_id:risk_domain -> PatientState
        self._states: Dict[str, PatientState] = {}
        # episode_id -> EpisodeState
        self._episodes: Dict[str, EpisodeState] = {}
        # Ordered list of all events
        self._audit_log: List[AlertEvent] = []
        self._max_events = max_events
        # Acknowledgments
        self._acknowledgments: List[AcknowledgmentRecord] = []
        self._max_acks = max_acks
        # Event index for fast lookup
        self._event_index: Dict[str, AlertEvent] = {}

    def _key(self, patient_id: str, risk_domain: str) -> str:
        return f"{patient_id}:{risk_domain}"

    def get_patient_state(self, patient_id: str, risk_domain: str) -> Optional[PatientState]:
        return self._states.get(self._key(patient_id, risk_domain))

    def save_patient_state(self, state: PatientState) -> None:
        self._states[self._key(state.patient_id, state.risk_domain)] = state
        # Also save/update episode if present
        if state.episode:
            self._episodes[state.episode.episode_id] = state.episode

    def get_episode(self, episode_id: str) -> Optional[EpisodeState]:
        return self._episodes.get(episode_id)

    def save_episode(self, episode: EpisodeState) -> None:
        self._episodes[episode.episode_id] = episode

    def get_open_episodes(self, patient_id: Optional[str] = None) -> List[EpisodeState]:
        episodes = [e for e in self._episodes.values() if e.is_open]
        if patient_id:
            episodes = [e for e in episodes if e.patient_id == patient_id]
        return sorted(episodes, key=lambda e: e.opened_at, reverse=True)

    def log_event(self, event: AlertEvent) -> None:
        self._audit_log.append(event)
        self._event_index[event.event_id] = event

        # Trim if over limit
        if len(self._audit_log) > self._max_events:
            removed = self._audit_log[:1000]
            self._audit_log = self._audit_log[1000:]
            for evt in removed:
                self._event_index.pop(evt.event_id, None)

    def get_events(
        self,
        patient_id: Optional[str] = None,
        risk_domain: Optional[str] = None,
        event_type: Optional[EventType] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AlertEvent]:
        results = self._audit_log

        if patient_id:
            results = [e for e in results if e.patient_id == patient_id]
        if risk_domain:
            results = [e for e in results if e.risk_domain == risk_domain]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if since:
            results = [e for e in results if e.timestamp >= since]
        if until:
            results = [e for e in results if e.timestamp <= until]

        # Sort by timestamp descending (most recent first)
        results = sorted(results, key=lambda e: e.timestamp, reverse=True)

        # Apply offset and limit
        return results[offset:offset + limit]

    def get_event_by_id(self, event_id: str) -> Optional[AlertEvent]:
        return self._event_index.get(event_id)

    def save_acknowledgment(self, ack: AcknowledgmentRecord) -> None:
        self._acknowledgments.append(ack)

        # Update episode if exists
        episode = self._episodes.get(ack.episode_id)
        if episode:
            episode.acknowledged = True
            episode.acknowledged_by = ack.acknowledged_by
            episode.acknowledged_at = ack.acknowledged_at
            if ack.close_episode:
                episode.closed_at = ack.acknowledged_at
                episode.closed_reason = "acknowledged"

        # Trim if over limit
        if len(self._acknowledgments) > self._max_acks:
            self._acknowledgments = self._acknowledgments[1000:]

    def get_acknowledgments(
        self,
        patient_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AcknowledgmentRecord]:
        results = self._acknowledgments

        if patient_id:
            results = [a for a in results if a.patient_id == patient_id]
        if episode_id:
            results = [a for a in results if a.episode_id == episode_id]
        if since:
            results = [a for a in results if a.acknowledged_at >= since]

        return sorted(results, key=lambda a: a.acknowledged_at, reverse=True)[:limit]

    def clear_patient(self, patient_id: str) -> None:
        keys_to_remove = [k for k in self._states if k.startswith(f"{patient_id}:")]
        for k in keys_to_remove:
            del self._states[k]

        episodes_to_remove = [e for e in self._episodes.values() if e.patient_id == patient_id]
        for e in episodes_to_remove:
            del self._episodes[e.episode_id]

    def get_stats(self) -> Dict[str, Any]:
        fired_count = len([e for e in self._audit_log if e.alert_fired])
        suppressed_count = len([e for e in self._audit_log if not e.alert_fired])

        return {
            "storage_type": "in_memory",
            "patient_states": len(self._states),
            "episodes": len(self._episodes),
            "open_episodes": len([e for e in self._episodes.values() if e.is_open]),
            "events_total": len(self._audit_log),
            "events_fired": fired_count,
            "events_suppressed": suppressed_count,
            "acknowledgments": len(self._acknowledgments),
            "max_events": self._max_events,
        }


# =============================================================================
# POSTGRESQL IMPLEMENTATION (Placeholder)
# =============================================================================

class PostgreSQLStorage(StorageBackend):
    """
    PostgreSQL storage for production deployments.

    Requires: psycopg2 or asyncpg, SQLAlchemy models

    Schema:
    - patient_states: patient_id, risk_domain, current_state, risk_score, ...
    - episodes: episode_id, patient_id, risk_domain, opened_at, closed_at, ...
    - alert_events: event_id, timestamp, patient_id, risk_domain, ...
    - acknowledgments: ack_id, alert_id, patient_id, ...
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._connected = False
        logger.info("PostgreSQL storage initialized (not connected)")

    def connect(self):
        """Establish database connection and create tables if needed."""
        # TODO: Implement actual PostgreSQL connection
        raise NotImplementedError("PostgreSQL storage not yet implemented")

    def get_patient_state(self, patient_id: str, risk_domain: str) -> Optional[PatientState]:
        raise NotImplementedError()

    def save_patient_state(self, state: PatientState) -> None:
        raise NotImplementedError()

    def get_episode(self, episode_id: str) -> Optional[EpisodeState]:
        raise NotImplementedError()

    def save_episode(self, episode: EpisodeState) -> None:
        raise NotImplementedError()

    def get_open_episodes(self, patient_id: Optional[str] = None) -> List[EpisodeState]:
        raise NotImplementedError()

    def log_event(self, event: AlertEvent) -> None:
        raise NotImplementedError()

    def get_events(
        self,
        patient_id: Optional[str] = None,
        risk_domain: Optional[str] = None,
        event_type: Optional[EventType] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AlertEvent]:
        raise NotImplementedError()

    def get_event_by_id(self, event_id: str) -> Optional[AlertEvent]:
        raise NotImplementedError()

    def save_acknowledgment(self, ack: AcknowledgmentRecord) -> None:
        raise NotImplementedError()

    def get_acknowledgments(
        self,
        patient_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AcknowledgmentRecord]:
        raise NotImplementedError()

    def clear_patient(self, patient_id: str) -> None:
        raise NotImplementedError()

    def get_stats(self) -> Dict[str, Any]:
        return {"storage_type": "postgresql", "connected": self._connected}


# =============================================================================
# REDIS CACHE WRAPPER
# =============================================================================

class RedisCacheWrapper(StorageBackend):
    """
    Redis caching wrapper for any storage backend.

    Provides:
    - Fast patient state lookups
    - Real-time pub/sub for alert events
    - TTL-based cache invalidation
    """

    def __init__(self, backend: StorageBackend, redis_url: Optional[str] = None):
        self.backend = backend
        self.redis_url = redis_url or get_redis_url()
        self._cache: Dict[str, Any] = {}  # Simple dict cache for now
        self._cache_ttl_seconds = 300  # 5 minutes
        logger.info(f"Redis cache wrapper initialized with URL configured: {bool(os.environ.get('REDIS_URL'))}")

    def get_patient_state(self, patient_id: str, risk_domain: str) -> Optional[PatientState]:
        # Check cache first
        cache_key = f"state:{patient_id}:{risk_domain}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.now(timezone.utc).timestamp() - cached["time"] < self._cache_ttl_seconds:
                return cached["data"]

        # Cache miss - get from backend
        state = self.backend.get_patient_state(patient_id, risk_domain)
        if state:
            self._cache[cache_key] = {
                "data": state,
                "time": datetime.now(timezone.utc).timestamp()
            }
        return state

    def save_patient_state(self, state: PatientState) -> None:
        # Invalidate cache
        cache_key = f"state:{state.patient_id}:{state.risk_domain}"
        self._cache.pop(cache_key, None)
        # Save to backend
        self.backend.save_patient_state(state)

    def get_episode(self, episode_id: str) -> Optional[EpisodeState]:
        return self.backend.get_episode(episode_id)

    def save_episode(self, episode: EpisodeState) -> None:
        self.backend.save_episode(episode)

    def get_open_episodes(self, patient_id: Optional[str] = None) -> List[EpisodeState]:
        return self.backend.get_open_episodes(patient_id)

    def log_event(self, event: AlertEvent) -> None:
        self.backend.log_event(event)
        # TODO: Publish to Redis pub/sub channel for real-time updates

    def get_events(self, **kwargs) -> List[AlertEvent]:
        return self.backend.get_events(**kwargs)

    def get_event_by_id(self, event_id: str) -> Optional[AlertEvent]:
        return self.backend.get_event_by_id(event_id)

    def save_acknowledgment(self, ack: AcknowledgmentRecord) -> None:
        self.backend.save_acknowledgment(ack)

    def get_acknowledgments(self, **kwargs) -> List[AcknowledgmentRecord]:
        return self.backend.get_acknowledgments(**kwargs)

    def clear_patient(self, patient_id: str) -> None:
        # Clear all cached keys for patient
        keys_to_remove = [k for k in self._cache if f":{patient_id}:" in k]
        for k in keys_to_remove:
            del self._cache[k]
        self.backend.clear_patient(patient_id)

    def get_stats(self) -> Dict[str, Any]:
        backend_stats = self.backend.get_stats()
        return {
            **backend_stats,
            "cache_type": "redis_fallback",
            "cache_entries": len(self._cache),
        }


# =============================================================================
# GLOBAL STORAGE INSTANCE
# =============================================================================

_storage: Optional[StorageBackend] = None


def init_storage(
    backend: str = "memory",
    connection_string: Optional[str] = None,
    use_cache: bool = False,
    redis_url: Optional[str] = None,
) -> StorageBackend:
    """
    Initialize the storage backend.

    Args:
        backend: "memory", "postgresql"
        connection_string: Database connection string (for postgresql)
        use_cache: Whether to wrap with Redis cache
        redis_url: Redis connection URL

    Returns:
        Configured storage backend
    """
    global _storage

    if backend == "memory":
        _storage = InMemoryStorage()
    elif backend == "postgresql":
        if not connection_string:
            raise ValueError("PostgreSQL requires connection_string")
        _storage = PostgreSQLStorage(connection_string)
    else:
        raise ValueError(f"Unknown storage backend: {backend}")

    if use_cache:
        _storage = RedisCacheWrapper(_storage, redis_url or get_redis_url())

    logger.info(f"Storage initialized: {backend}, cache={use_cache}")
    return _storage


def get_storage() -> StorageBackend:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = InMemoryStorage()
    return _storage


def set_storage(storage: StorageBackend) -> None:
    """Set a custom storage backend."""
    global _storage
    _storage = storage
