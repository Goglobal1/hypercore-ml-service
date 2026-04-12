"""
Cognition Store - HyperCore Evolution
======================================

Domain knowledge and prior learning storage.
Adapted from GAIR-NLP ASI-Evolve cognition pattern.

The Cognition Store:
1. Stores domain knowledge (regulatory, clinical, methodological)
2. Retrieves relevant items via embedding similarity
3. Persists lessons from evolution pipeline
4. Provides context for hypothesis generation

Healthcare Additions:
- FDA/EMA regulatory guidance
- Clinical protocol templates
- Drug interaction knowledge
- Safety constraint library
"""

from __future__ import annotations
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..schemas import (
    CognitionItem,
    CognitionItemType,
    DeploymentDomain,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CognitionConfig:
    """Configuration for the Cognition Store."""
    # Storage
    storage_dir: Path = Path("data/evolution/cognition")

    # Capacity
    max_items: int = 100000
    max_items_per_type: int = 20000

    # Retrieval
    default_top_k: int = 10
    similarity_threshold: float = 0.5

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    use_embeddings: bool = True

    # Auto-cleanup
    auto_prune_threshold: int = 90000  # Prune when reaching this
    prune_keep_ratio: float = 0.8  # Keep top 80% by relevance

    def __post_init__(self):
        self.storage_dir = Path(self.storage_dir)


@dataclass
class RetrievalResult:
    """Result of a cognition retrieval."""
    item: CognitionItem
    score: float  # Similarity score [0, 1]
    retrieval_method: str  # "embedding", "keyword", "exact"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item.item_id,
            "title": self.item.title,
            "content": self.item.content[:200] + "..." if len(self.item.content) > 200 else self.item.content,
            "score": self.score,
            "method": self.retrieval_method,
        }


class CognitionStore:
    """
    Domain knowledge and prior learning storage.

    Stores:
    - Regulatory guidance (FDA, EMA, ICH)
    - Clinical protocols and guidelines
    - Prior experiment lessons
    - Safety constraints and rules
    - Methodological best practices

    Retrieval:
    - Embedding-based similarity search
    - Keyword matching fallback
    - Type and domain filtering

    Usage:
        store = CognitionStore()

        # Add knowledge
        store.add(CognitionItem(
            title="FDA AI/ML Guidance",
            content="...",
            item_type=CognitionItemType.REGULATORY,
        ))

        # Retrieve relevant items
        results = store.retrieve(
            query="clinical trial endpoint selection",
            top_k=5,
        )
    """

    def __init__(
        self,
        config: Optional[CognitionConfig] = None,
        embedding_client: Optional[Any] = None,
    ):
        """
        Initialize Cognition Store.

        Args:
            config: Store configuration
            embedding_client: Client for generating embeddings (optional)
        """
        self.config = config or CognitionConfig()
        self.config.storage_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_client = embedding_client

        # In-memory storage
        self._items: Dict[str, CognitionItem] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

        # Indexes
        self._by_type: Dict[CognitionItemType, List[str]] = {t: [] for t in CognitionItemType}
        self._by_domain: Dict[DeploymentDomain, List[str]] = {d: [] for d in DeploymentDomain}
        self._by_category: Dict[str, List[str]] = {}

        # Stats
        self._retrieval_count = 0
        self._add_count = 0

        # Callbacks
        self._on_add: List[Callable[[CognitionItem], None]] = []

        # Load existing data
        self._load()

        logger.info(f"CognitionStore initialized with {len(self._items)} items")

    # =========================================================================
    # ADD / UPDATE
    # =========================================================================

    def add(
        self,
        item: CognitionItem,
        generate_embedding: bool = True,
    ) -> CognitionItem:
        """
        Add a cognition item to the store.

        Args:
            item: The item to add
            generate_embedding: Whether to generate embedding

        Returns:
            The added item (with ID assigned if not present)
        """
        with self._lock:
            # Check capacity
            if len(self._items) >= self.config.max_items:
                self._prune()

            # Check per-type limit
            type_count = len(self._by_type.get(item.item_type, []))
            if type_count >= self.config.max_items_per_type:
                self._prune_type(item.item_type)

            # Store item
            self._items[item.item_id] = item

            # Update indexes
            self._by_type[item.item_type].append(item.item_id)
            self._by_domain[item.domain].append(item.item_id)

            if item.category:
                if item.category not in self._by_category:
                    self._by_category[item.category] = []
                self._by_category[item.category].append(item.item_id)

            self._add_count += 1

        # Generate embedding (outside lock)
        if generate_embedding and self.config.use_embeddings:
            self._generate_embedding(item)

        # Callbacks
        for callback in self._on_add:
            try:
                callback(item)
            except Exception as e:
                logger.error(f"Add callback failed: {e}")

        # Persist
        self._save_item(item)

        logger.debug(f"Added cognition item: {item.item_id} ({item.item_type.value})")

        return item

    def add_many(
        self,
        items: List[CognitionItem],
        generate_embeddings: bool = True,
    ) -> int:
        """Add multiple items efficiently."""
        added = 0
        for item in items:
            try:
                self.add(item, generate_embedding=generate_embeddings)
                added += 1
            except Exception as e:
                logger.error(f"Failed to add item {item.item_id}: {e}")

        logger.info(f"Added {added}/{len(items)} cognition items")
        return added

    def update(self, item: CognitionItem) -> bool:
        """Update an existing item."""
        with self._lock:
            if item.item_id not in self._items:
                return False

            self._items[item.item_id] = item

        # Regenerate embedding
        if self.config.use_embeddings:
            self._generate_embedding(item)

        self._save_item(item)
        return True

    def remove(self, item_id: str) -> bool:
        """Remove an item from the store."""
        with self._lock:
            if item_id not in self._items:
                return False

            item = self._items.pop(item_id)

            # Update indexes
            if item_id in self._by_type[item.item_type]:
                self._by_type[item.item_type].remove(item_id)
            if item_id in self._by_domain[item.domain]:
                self._by_domain[item.domain].remove(item_id)
            if item.category and item_id in self._by_category.get(item.category, []):
                self._by_category[item.category].remove(item_id)

            # Remove embedding
            self._embeddings.pop(item_id, None)

        # Remove from disk
        self._remove_item_file(item_id)

        return True

    # =========================================================================
    # RETRIEVAL
    # =========================================================================

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        item_type: Optional[CognitionItemType] = None,
        domain: Optional[DeploymentDomain] = None,
        category: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant cognition items.

        Args:
            query: Search query
            top_k: Number of results to return
            item_type: Filter by item type
            domain: Filter by domain
            category: Filter by category
            min_score: Minimum similarity score

        Returns:
            List of retrieval results sorted by relevance
        """
        top_k = top_k or self.config.default_top_k
        min_score = min_score or self.config.similarity_threshold

        self._retrieval_count += 1

        # Get candidate items
        candidates = self._get_candidates(item_type, domain, category)

        if not candidates:
            return []

        # Score candidates
        if self.config.use_embeddings and self.embedding_client:
            results = self._retrieve_by_embedding(query, candidates, top_k)
        else:
            results = self._retrieve_by_keyword(query, candidates, top_k)

        # Filter by minimum score
        results = [r for r in results if r.score >= min_score]

        # Update retrieval counts
        for result in results:
            result.item.retrieval_count += 1

        return results

    def retrieve_by_ids(self, item_ids: List[str]) -> List[CognitionItem]:
        """Retrieve items by their IDs."""
        with self._lock:
            return [
                self._items[item_id]
                for item_id in item_ids
                if item_id in self._items
            ]

    def get(self, item_id: str) -> Optional[CognitionItem]:
        """Get a single item by ID."""
        with self._lock:
            return self._items.get(item_id)

    def get_by_type(
        self,
        item_type: CognitionItemType,
        limit: int = 100,
    ) -> List[CognitionItem]:
        """Get all items of a specific type."""
        with self._lock:
            item_ids = self._by_type.get(item_type, [])[:limit]
            return [self._items[item_id] for item_id in item_ids]

    def get_by_domain(
        self,
        domain: DeploymentDomain,
        limit: int = 100,
    ) -> List[CognitionItem]:
        """Get all items for a specific domain."""
        with self._lock:
            item_ids = self._by_domain.get(domain, [])[:limit]
            return [self._items[item_id] for item_id in item_ids]

    def get_recent(self, limit: int = 20) -> List[CognitionItem]:
        """Get most recently added items."""
        with self._lock:
            items = list(self._items.values())

        items.sort(key=lambda x: x.created_at, reverse=True)
        return items[:limit]

    def get_most_retrieved(self, limit: int = 20) -> List[CognitionItem]:
        """Get most frequently retrieved items."""
        with self._lock:
            items = list(self._items.values())

        items.sort(key=lambda x: x.retrieval_count, reverse=True)
        return items[:limit]

    def _get_candidates(
        self,
        item_type: Optional[CognitionItemType],
        domain: Optional[DeploymentDomain],
        category: Optional[str],
    ) -> List[CognitionItem]:
        """Get candidate items based on filters."""
        with self._lock:
            # Start with all items
            if item_type is not None:
                item_ids = set(self._by_type.get(item_type, []))
            else:
                item_ids = set(self._items.keys())

            # Apply domain filter
            if domain is not None:
                domain_ids = set(self._by_domain.get(domain, []))
                item_ids &= domain_ids

            # Apply category filter
            if category is not None:
                category_ids = set(self._by_category.get(category, []))
                item_ids &= category_ids

            return [self._items[item_id] for item_id in item_ids]

    def _retrieve_by_embedding(
        self,
        query: str,
        candidates: List[CognitionItem],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Retrieve using embedding similarity."""
        # Generate query embedding
        query_embedding = self._get_embedding(query)

        if query_embedding is None:
            # Fall back to keyword search
            return self._retrieve_by_keyword(query, candidates, top_k)

        # Calculate similarities
        scored = []
        for item in candidates:
            item_embedding = self._embeddings.get(item.item_id)
            if item_embedding is None:
                continue

            similarity = self._cosine_similarity(query_embedding, item_embedding)
            scored.append((item, similarity))

        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                item=item,
                score=score,
                retrieval_method="embedding",
            )
            for item, score in scored[:top_k]
        ]

    def _retrieve_by_keyword(
        self,
        query: str,
        candidates: List[CognitionItem],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Retrieve using keyword matching."""
        query_words = set(query.lower().split())

        scored = []
        for item in candidates:
            # Combine title and content for matching
            text = f"{item.title} {item.content}".lower()
            text_words = set(text.split())

            # Calculate Jaccard similarity
            if not query_words or not text_words:
                similarity = 0.0
            else:
                intersection = len(query_words & text_words)
                union = len(query_words | text_words)
                similarity = intersection / union if union > 0 else 0.0

            # Boost for title matches
            title_words = set(item.title.lower().split())
            title_overlap = len(query_words & title_words) / len(query_words) if query_words else 0
            similarity += title_overlap * 0.3

            scored.append((item, min(1.0, similarity)))

        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                item=item,
                score=score,
                retrieval_method="keyword",
            )
            for item, score in scored[:top_k]
        ]

    # =========================================================================
    # EMBEDDING
    # =========================================================================

    def _generate_embedding(self, item: CognitionItem) -> None:
        """Generate and store embedding for an item."""
        text = f"{item.title}\n\n{item.content}"
        embedding = self._get_embedding(text)

        if embedding is not None:
            with self._lock:
                self._embeddings[item.item_id] = embedding

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        if not self.embedding_client:
            return None

        try:
            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000]

            response = self.embedding_client.embed(text)
            return response

        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save_item(self, item: CognitionItem) -> None:
        """Save a single item to disk."""
        item_file = self.config.storage_dir / f"{item.item_id}.json"

        data = {
            "item": item.to_dict() if hasattr(item, 'to_dict') else item.__dict__,
            "embedding": self._embeddings.get(item.item_id),
        }

        with open(item_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _remove_item_file(self, item_id: str) -> None:
        """Remove item file from disk."""
        item_file = self.config.storage_dir / f"{item_id}.json"
        if item_file.exists():
            item_file.unlink()

    def _load(self) -> None:
        """Load all items from disk."""
        for item_file in self.config.storage_dir.glob("*.json"):
            try:
                with open(item_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                item_data = data.get("item", {})
                item = CognitionItem(
                    item_id=item_data.get("item_id", item_file.stem),
                    title=item_data.get("title", ""),
                    content=item_data.get("content", ""),
                    source=item_data.get("source", ""),
                    item_type=CognitionItemType(item_data.get("item_type", "lesson")),
                    category=item_data.get("category", ""),
                    domain=DeploymentDomain(item_data.get("domain", "research")),
                    created_at=item_data.get("created_at", ""),
                    metadata=item_data.get("metadata", {}),
                    retrieval_count=item_data.get("retrieval_count", 0),
                )

                self._items[item.item_id] = item

                # Update indexes
                self._by_type[item.item_type].append(item.item_id)
                self._by_domain[item.domain].append(item.item_id)
                if item.category:
                    if item.category not in self._by_category:
                        self._by_category[item.category] = []
                    self._by_category[item.category].append(item.item_id)

                # Load embedding
                embedding = data.get("embedding")
                if embedding:
                    self._embeddings[item.item_id] = embedding

            except Exception as e:
                logger.error(f"Failed to load {item_file}: {e}")

    def save_all(self) -> None:
        """Save all items to disk."""
        with self._lock:
            for item in self._items.values():
                self._save_item(item)

        logger.info(f"Saved {len(self._items)} cognition items")

    # =========================================================================
    # PRUNING
    # =========================================================================

    def _prune(self) -> None:
        """Prune least relevant items to make space."""
        with self._lock:
            items = list(self._items.values())

        # Sort by retrieval count (keep most retrieved)
        items.sort(key=lambda x: x.retrieval_count, reverse=True)

        # Keep top percentage
        keep_count = int(len(items) * self.config.prune_keep_ratio)
        to_remove = items[keep_count:]

        for item in to_remove:
            self.remove(item.item_id)

        logger.info(f"Pruned {len(to_remove)} cognition items")

    def _prune_type(self, item_type: CognitionItemType) -> None:
        """Prune items of a specific type."""
        with self._lock:
            item_ids = self._by_type.get(item_type, [])
            items = [self._items[item_id] for item_id in item_ids]

        items.sort(key=lambda x: x.retrieval_count, reverse=True)

        keep_count = int(len(items) * self.config.prune_keep_ratio)
        to_remove = items[keep_count:]

        for item in to_remove:
            self.remove(item.item_id)

        logger.info(f"Pruned {len(to_remove)} items of type {item_type.value}")

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_add(self, callback: Callable[[CognitionItem], None]) -> None:
        """Register callback for item additions."""
        self._on_add.append(callback)

    # =========================================================================
    # STATS
    # =========================================================================

    def __len__(self) -> int:
        return len(self._items)

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            by_type = {t.value: len(ids) for t, ids in self._by_type.items()}
            by_domain = {d.value: len(ids) for d, ids in self._by_domain.items()}

        return {
            "total_items": len(self._items),
            "total_embeddings": len(self._embeddings),
            "by_type": by_type,
            "by_domain": by_domain,
            "retrieval_count": self._retrieval_count,
            "add_count": self._add_count,
            "categories": list(self._by_category.keys()),
        }


# Global instance
_cognition_instance: Optional[CognitionStore] = None


def get_cognition_store(
    storage_dir: Optional[Path] = None,
    config: Optional[CognitionConfig] = None,
) -> CognitionStore:
    """Get or create the global cognition store instance."""
    global _cognition_instance

    if _cognition_instance is None:
        if config is None:
            config = CognitionConfig()
            if storage_dir:
                config.storage_dir = storage_dir
        _cognition_instance = CognitionStore(config)

    return _cognition_instance
