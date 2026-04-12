"""
UCB1 Sampler - HyperCore
========================

Upper Confidence Bound (UCB1) sampling for parent selection.
Balances exploration (try unvisited nodes) with exploitation
(use proven high-scorers).

Adapted from GAIR-NLP ASI-Evolve with healthcare additions:
- Utility-weighted scoring
- Safety-aware sampling
- Domain filtering

Formula:
    UCB1 = normalized_score + c * sqrt(ln(N) / n_i)

Where:
    - normalized_score: Score normalized to [0, 1]
    - c: Exploration coefficient (default 1.414 = sqrt(2))
    - N: Total visits across all nodes
    - n_i: Visits to this specific node
"""

from __future__ import annotations
import math
import random
import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas import EvolutionNode

logger = logging.getLogger(__name__)


class UCB1Sampler:
    """
    UCB1 sampling for evolution parent selection.

    Balances exploration of unvisited/low-visited nodes with
    exploitation of high-scoring nodes.

    Usage:
        sampler = UCB1Sampler(c=1.414)
        parents = sampler.sample(nodes, n=3)
    """

    def __init__(
        self,
        c: float = 1.414,
        safety_weight: float = 0.2,
        utility_weight: float = 0.3,
    ):
        """
        Initialize UCB1 sampler.

        Args:
            c: Exploration coefficient (higher = more exploration)
            safety_weight: Weight for safety score in ranking
            utility_weight: Weight for utility score in ranking
        """
        self.c = c
        self.safety_weight = safety_weight
        self.utility_weight = utility_weight

        # Stats tracking
        self._total_samples = 0
        self._exploration_samples = 0  # Unvisited nodes
        self._exploitation_samples = 0  # Top-scoring nodes

    def sample(
        self,
        nodes: List["EvolutionNode"],
        n: int,
        domain_filter: Optional[str] = None,
        min_tier: Optional[int] = None,
        require_safety: bool = True,
    ) -> List["EvolutionNode"]:
        """
        Sample n nodes using UCB1.

        Args:
            nodes: Pool of nodes to sample from
            n: Number of nodes to sample
            domain_filter: Only consider nodes in this domain
            min_tier: Minimum capability tier
            require_safety: Only sample nodes that passed safety

        Returns:
            List of sampled nodes
        """
        if not nodes:
            return []

        # Apply filters
        filtered = self._apply_filters(
            nodes, domain_filter, min_tier, require_safety
        )

        if not filtered:
            logger.warning("No nodes passed filters, sampling from all")
            filtered = nodes

        n = min(n, len(filtered))

        # Calculate total visits
        total_visits = sum(node.visit_count for node in filtered)

        if total_visits == 0:
            # No visits yet - random sample
            selected = random.sample(filtered, n)
            self._exploration_samples += n
            self._total_samples += n
            return selected

        # Calculate UCB1 scores
        ucb1_scores = []
        for node in filtered:
            ucb1 = self._calculate_ucb1(node, total_visits)
            ucb1_scores.append((node, ucb1))

        # Sort by UCB1 (descending)
        ucb1_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top n
        selected = [node for node, _ in ucb1_scores[:n]]

        # Track stats
        for node in selected:
            if node.visit_count == 0:
                self._exploration_samples += 1
            else:
                self._exploitation_samples += 1
        self._total_samples += n

        return selected

    def _calculate_ucb1(
        self,
        node: "EvolutionNode",
        total_visits: int,
    ) -> float:
        """Calculate UCB1 score for a node."""
        if node.visit_count == 0:
            # Unvisited nodes get infinite UCB1 (force exploration)
            return float("inf")

        # Get combined score
        base_score = self._get_combined_score(node)

        # Normalize to [0, 1] (assume scores are already roughly in this range)
        normalized_score = max(0.0, min(1.0, base_score))

        # Exploration bonus
        exploration = self.c * math.sqrt(
            math.log(total_visits) / node.visit_count
        )

        return normalized_score + exploration

    def _get_combined_score(self, node: "EvolutionNode") -> float:
        """
        Calculate combined score including safety and utility.

        Weights:
        - Base score: 1 - safety_weight - utility_weight
        - Safety score: safety_weight
        - Utility score: utility_weight
        """
        base_weight = 1.0 - self.safety_weight - self.utility_weight

        combined = node.score * base_weight

        # Add safety component
        if node.evaluation_result:
            safety = 1.0 if node.evaluation_result.safety_passed else 0.0
            # Penalize for warnings
            warning_penalty = len(node.evaluation_result.safety_warnings) * 0.05
            safety = max(0.0, safety - warning_penalty)
            combined += safety * self.safety_weight

        # Add utility component
        if node.utility_breakdown:
            utility = node.utility_breakdown.combined_score()
            combined += utility * self.utility_weight

        return combined

    def _apply_filters(
        self,
        nodes: List["EvolutionNode"],
        domain_filter: Optional[str],
        min_tier: Optional[int],
        require_safety: bool,
    ) -> List["EvolutionNode"]:
        """Apply filters to node pool."""
        filtered = nodes

        if domain_filter:
            filtered = [n for n in filtered if n.domain.value == domain_filter]

        if min_tier is not None:
            filtered = [n for n in filtered if n.capability_tier >= min_tier]

        if require_safety:
            filtered = [
                n for n in filtered
                if n.evaluation_result is None  # Not evaluated yet
                or n.evaluation_result.safety_passed
            ]

        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        exploration_ratio = (
            self._exploration_samples / self._total_samples
            if self._total_samples > 0 else 0
        )

        return {
            "total_samples": self._total_samples,
            "exploration_samples": self._exploration_samples,
            "exploitation_samples": self._exploitation_samples,
            "exploration_ratio": exploration_ratio,
            "c": self.c,
            "safety_weight": self.safety_weight,
            "utility_weight": self.utility_weight,
        }

    def reset_stats(self) -> None:
        """Reset sampling statistics."""
        self._total_samples = 0
        self._exploration_samples = 0
        self._exploitation_samples = 0


class UtilityWeightedSampler:
    """
    Simple utility-weighted sampling.

    Samples proportional to utility score - higher utility
    means higher chance of being selected.

    Use when you want pure exploitation (no exploration).
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize sampler.

        Args:
            temperature: Higher = more uniform, lower = more greedy
        """
        self.temperature = temperature

    def sample(
        self,
        nodes: List["EvolutionNode"],
        n: int,
    ) -> List["EvolutionNode"]:
        """Sample n nodes weighted by utility."""
        if not nodes:
            return []

        n = min(n, len(nodes))

        # Calculate weights
        weights = []
        for node in nodes:
            if node.utility_breakdown:
                weight = node.utility_breakdown.combined_score()
            else:
                weight = node.score

            # Apply temperature
            weight = max(0.001, weight) ** (1.0 / self.temperature)
            weights.append(weight)

        # Normalize
        total = sum(weights)
        if total == 0:
            return random.sample(nodes, n)

        probs = [w / total for w in weights]

        # Sample without replacement
        selected = []
        available = list(zip(nodes, probs))

        for _ in range(n):
            if not available:
                break

            # Weighted random selection
            r = random.random()
            cumsum = 0
            for i, (node, prob) in enumerate(available):
                cumsum += prob
                if r <= cumsum:
                    selected.append(node)
                    available.pop(i)
                    # Renormalize remaining
                    remaining_total = sum(p for _, p in available)
                    if remaining_total > 0:
                        available = [(n, p / remaining_total) for n, p in available]
                    break

        return selected


class DiversitySampler:
    """
    Diversity-aware sampling.

    Samples to maximize diversity in the selected set,
    avoiding similar nodes.

    Uses motivation/analysis text similarity.
    """

    def __init__(self, diversity_weight: float = 0.5):
        """
        Initialize sampler.

        Args:
            diversity_weight: Balance between score and diversity
        """
        self.diversity_weight = diversity_weight

    def sample(
        self,
        nodes: List["EvolutionNode"],
        n: int,
    ) -> List["EvolutionNode"]:
        """Sample n diverse nodes."""
        if not nodes:
            return []

        n = min(n, len(nodes))

        if n == 1:
            # Just take the best
            return [max(nodes, key=lambda x: x.score)]

        # Greedy selection maximizing score + diversity
        selected = []
        remaining = list(nodes)

        # First node: highest score
        best = max(remaining, key=lambda x: x.score)
        selected.append(best)
        remaining.remove(best)

        # Subsequent nodes: balance score and diversity
        for _ in range(n - 1):
            if not remaining:
                break

            best_score = float("-inf")
            best_node = None

            for node in remaining:
                score = node.score * (1 - self.diversity_weight)

                # Diversity: average dissimilarity to selected
                diversity = self._diversity_score(node, selected)
                score += diversity * self.diversity_weight

                if score > best_score:
                    best_score = score
                    best_node = node

            if best_node:
                selected.append(best_node)
                remaining.remove(best_node)

        return selected

    def _diversity_score(
        self,
        node: "EvolutionNode",
        selected: List["EvolutionNode"],
    ) -> float:
        """Calculate diversity score (dissimilarity to selected)."""
        if not selected:
            return 1.0

        # Simple: use motivation text overlap
        node_words = set(node.motivation.lower().split())

        similarities = []
        for other in selected:
            other_words = set(other.motivation.lower().split())

            if not node_words or not other_words:
                similarities.append(0.0)
                continue

            # Jaccard similarity
            intersection = len(node_words & other_words)
            union = len(node_words | other_words)
            similarity = intersection / union if union > 0 else 0

            similarities.append(similarity)

        # Return 1 - average similarity
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity
