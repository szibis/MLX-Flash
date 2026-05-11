"""Expert prefetch from DFlash draft predictions.

Uses draft token IDs to predict which MoE experts will be needed during
target model verification. By running draft embeddings through the router
weights, we can prefetch experts from SSD/cache before they are actually
needed, hiding I/O latency behind the verification compute.

This is prediction-only: incorrect predictions just waste a prefetch I/O
operation with no impact on correctness.

Integration:
    prefetcher = DraftExpertPrefetcher(expert_cache, router_weights)
    runner = DFlashRunner(target, tokenizer, drafter, config,
                          expert_prefetcher=prefetcher)
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import numpy as np


class DraftExpertPrefetcher:
    """Predicts and prefetches MoE experts based on DFlash draft tokens.

    Takes draft token IDs, embeds them, runs through router (gate) weights
    to predict expert assignments, and triggers async prefetch on the
    expert cache.

    Args:
        expert_cache: Cache object with a ``prefetch(layer_idx, expert_ids)``
            method (e.g. LCPCache, FastCacheBindings, or ExpertCacheManager).
            If None, prefetch calls are no-ops (useful for testing).
        router_weights: Dict mapping layer_idx to gate weight matrices.
            Each value should be an mx.array of shape [hidden_size, num_experts]
            or [num_experts, hidden_size] (auto-detected).
        top_k: Number of top experts to prefetch per layer per draft token.
    """

    def __init__(
        self,
        expert_cache: Any,
        router_weights: dict[int, mx.array],
        top_k: int = 2,
    ):
        self._cache = expert_cache
        self._router_weights = router_weights
        self._top_k = top_k

        # Stats
        self._prefetch_requests: int = 0
        self._cache_hits_from_prefetch: int = 0
        self._total_predictions: int = 0

    def prefetch_from_drafts(
        self,
        draft_token_ids: list[int],
        embed_fn: Any,
    ) -> list[tuple[int, int]]:
        """Predict and prefetch experts for draft tokens.

        Args:
            draft_token_ids: List of draft token IDs from the drafter.
            embed_fn: Embedding function (target model's embed_tokens).
                Must accept mx.array of token IDs and return embeddings.

        Returns:
            List of (layer_idx, expert_idx) pairs that were requested
            for prefetch.
        """
        if not draft_token_ids or not self._router_weights:
            return []

        # Embed draft tokens: [1, n_draft, hidden_size]
        token_ids = mx.array([draft_token_ids])
        embeddings = embed_fn(token_ids)
        mx.eval(embeddings)

        # Average pooling across draft positions for a single prediction vector
        # Shape: [hidden_size]
        avg_embedding = mx.mean(embeddings[0], axis=0)

        prefetch_pairs: list[tuple[int, int]] = []

        for layer_idx, gate_weight in self._router_weights.items():
            # gate_weight can be [num_experts, hidden_size] or [hidden_size, num_experts]
            # We need scores = embedding @ gate_weight^T -> [num_experts]
            if gate_weight.shape[0] == avg_embedding.shape[0]:
                # [hidden_size, num_experts] layout
                scores = avg_embedding @ gate_weight
            else:
                # [num_experts, hidden_size] layout
                scores = gate_weight @ avg_embedding

            mx.eval(scores)

            # Get top-K expert indices
            k = min(self._top_k, scores.shape[0])
            if k <= 0:
                continue

            top_k_indices = mx.argpartition(scores, kth=-k)[-k:]
            mx.eval(top_k_indices)
            expert_ids = list(top_k_indices.tolist())  # type: ignore[arg-type]

            self._total_predictions += k

            # Trigger async prefetch
            if self._cache is not None:
                try:
                    self._cache.prefetch(layer_idx, expert_ids)
                    self._prefetch_requests += 1
                except (AttributeError, TypeError):
                    # Cache doesn't support prefetch — silently skip
                    pass

            for eid in expert_ids:
                prefetch_pairs.append((layer_idx, eid))

        return prefetch_pairs

    def get_stats(self) -> dict[str, Any]:
        """Return prefetch statistics.

        Returns:
            Dict with keys:
                prefetch_requests: Number of prefetch() calls made.
                cache_hits_from_prefetch: Number of cache hits attributed
                    to prefetching (must be updated externally).
                total_predictions: Total (layer, expert) pairs predicted.
                accuracy: Hit rate (0.0 if no predictions yet).
        """
        return {
            "prefetch_requests": self._prefetch_requests,
            "cache_hits_from_prefetch": self._cache_hits_from_prefetch,
            "total_predictions": self._total_predictions,
            "accuracy": (
                self._cache_hits_from_prefetch / self._total_predictions if self._total_predictions > 0 else 0.0
            ),
        }

    def record_cache_hit(self, count: int = 1) -> None:
        """Record that prefetched experts were actually used (cache hit).

        Called externally when the target model verification step finds
        that a prefetched expert was in cache.
        """
        self._cache_hits_from_prefetch += count
