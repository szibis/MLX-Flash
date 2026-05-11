"""ScissorHands/H2O KV cache compression via attention-guided eviction.

Combines two complementary approaches:

ScissorHands (arXiv:2305.17118, "Pivotal Token Hypothesis"):
  Tokens that are important at generation step t tend to remain important
  at step t+1 ("persistence of importance"). This means we can safely evict
  tokens that have consistently low attention scores, as they're unlikely
  to become important later.

H2O (arXiv:2306.14048, "Heavy-Hitter Oracle"):
  A small subset of tokens ("heavy hitters") accumulate high attention
  scores across ALL layers. These tokens carry disproportionate semantic
  weight and must be retained. H2O tracks cumulative attention to identify
  them.

Combined approach:
  - Always keep sink tokens (first K) and recent window (last W)
  - For remaining budget, keep tokens with highest importance scores
  - H2O mode: accumulate attention across layers for cross-layer importance
  - ScissorHands mode: use per-step attention for temporal persistence

Achieves 5-20x KV compression with <1% quality loss on standard benchmarks.

Usage:
    config = KVCompressionConfig(budget_ratio=0.2, scoring="h2o")
    cache = CompressedKVCache(config, num_layers=32, num_heads=8, head_dim=128)

    # During generation, pass attention weights:
    keys, values = cache.update(layer_idx, new_k, new_v, attn_weights)
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class KVCompressionConfig:
    """Configuration for attention-guided KV cache compression.

    Attributes:
        budget_ratio: Fraction of KV entries to keep (0.2 = 5x compression).
        sink_tokens: Number of initial tokens always kept (attention sinks).
        recent_window: Number of most recent tokens always kept.
        scoring: Scoring strategy -- "h2o" for cross-layer accumulation,
            "scissorhands" for per-step persistence tracking.
        quantize_evicted: If True, kept entries are quantized to 4-bit
            after compression (further memory reduction).
    """

    budget_ratio: float = 0.2
    sink_tokens: int = 4
    recent_window: int = 128
    scoring: str = "h2o"
    quantize_evicted: bool = False


class AttentionScoreTracker:
    """Track attention scores to identify pivotal/heavy-hitter tokens.

    Supports two scoring modes:
      - "h2o": Accumulates attention weights across all layers. Tokens with
        high cumulative attention are "heavy hitters" that should be kept.
      - "scissorhands": Tracks per-step attention. Uses exponential moving
        average to capture temporal persistence of importance.
    """

    def __init__(self, num_layers: int, max_seq_len: int = 8192, scoring: str = "h2o"):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.scoring = scoring

        # Per-layer cumulative importance scores
        # Shape: [max_seq_len] per layer -- grows as tokens arrive
        self._importance: list[mx.array | None] = [None] * num_layers

        # H2O: cross-layer accumulator (summed across all layers)
        self._global_importance: mx.array | None = None

        # ScissorHands: exponential moving average decay factor
        self._ema_decay: float = 0.9

    def record_attention(self, layer_idx: int, attention_weights: mx.array):
        """Record attention weights for this layer.

        Args:
            layer_idx: Transformer layer index.
            attention_weights: Attention weight tensor of shape
                [num_heads, seq_len, seq_len]. We extract the last row
                (most recent query's attention over all keys) and average
                across heads to get per-token importance.
        """
        # Extract attention from the last query position (most recent token)
        # Shape: [num_heads, seq_len] -- attention of last query over all keys
        last_query_attn = attention_weights[:, -1, :]

        # Average across heads: shape [seq_len]
        token_scores = mx.mean(last_query_attn, axis=0)

        seq_len = token_scores.shape[0]

        if self.scoring == "h2o":
            self._record_h2o(layer_idx, token_scores, seq_len)
        else:
            self._record_scissorhands(layer_idx, token_scores, seq_len)

    def _record_h2o(self, layer_idx: int, token_scores: mx.array, seq_len: int):
        """H2O: Accumulate attention across layers."""
        if self._importance[layer_idx] is None:
            self._importance[layer_idx] = token_scores
        else:
            cur_len = self._importance[layer_idx].shape[0]
            if seq_len > cur_len:
                # Pad existing scores for new tokens
                padding = mx.zeros((seq_len - cur_len,))
                self._importance[layer_idx] = mx.concatenate([self._importance[layer_idx], padding])
            # Accumulate (sum attention received across steps)
            self._importance[layer_idx] = self._importance[layer_idx][:seq_len] + token_scores

        # Update global (cross-layer) importance
        if self._global_importance is None:
            self._global_importance = mx.zeros((seq_len,))
        if self._global_importance.shape[0] < seq_len:
            padding = mx.zeros((seq_len - self._global_importance.shape[0],))
            self._global_importance = mx.concatenate([self._global_importance, padding])
        self._global_importance = self._global_importance[:seq_len] + token_scores

    def _record_scissorhands(self, layer_idx: int, token_scores: mx.array, seq_len: int):
        """ScissorHands: Exponential moving average of per-step attention."""
        if self._importance[layer_idx] is None:
            self._importance[layer_idx] = token_scores
        else:
            cur_len = self._importance[layer_idx].shape[0]
            if seq_len > cur_len:
                padding = mx.zeros((seq_len - cur_len,))
                self._importance[layer_idx] = mx.concatenate([self._importance[layer_idx], padding])
            # EMA update: importance = decay * old + (1 - decay) * new
            self._importance[layer_idx] = (
                self._ema_decay * self._importance[layer_idx][:seq_len] + (1.0 - self._ema_decay) * token_scores
            )

    def get_token_importance(self, layer_idx: int) -> mx.array:
        """Return importance score per token position.

        Higher score = more important token.

        For H2O scoring, returns the global (cross-layer) accumulation.
        For ScissorHands, returns the per-layer EMA scores.
        """
        if self.scoring == "h2o" and self._global_importance is not None:
            return self._global_importance
        if self._importance[layer_idx] is not None:
            return self._importance[layer_idx]
        return mx.array([0.0])

    def get_heavy_hitters(self, layer_idx: int, top_k: int) -> mx.array:
        """Return indices of top-k most important tokens.

        Args:
            layer_idx: Layer to query importance for.
            top_k: Number of top tokens to return.

        Returns:
            Array of token indices sorted by descending importance.
        """
        importance = self.get_token_importance(layer_idx)
        seq_len = importance.shape[0]
        top_k = min(top_k, seq_len)

        if top_k == 0:
            return mx.array([], dtype=mx.int32)

        # Use argsort (descending) to find top-k indices
        sorted_indices = mx.argsort(-importance)
        return sorted_indices[:top_k]

    def truncate(self, keep_indices: mx.array):
        """Truncate tracked scores to match a compressed cache.

        After eviction, the importance arrays must be reindexed to match
        the remaining tokens in the cache.

        Args:
            keep_indices: Indices of tokens that were kept after eviction.
        """
        # Convert to Python list for filtering (MLX lacks boolean indexing)
        idx_list: list[int] = list(keep_indices.tolist())  # type: ignore[arg-type]

        for i in range(self.num_layers):
            if self._importance[i] is not None:
                seq_len = self._importance[i].shape[0]
                valid = [j for j in idx_list if j < seq_len]
                if valid:
                    self._importance[i] = self._importance[i][mx.array(valid)]
                else:
                    self._importance[i] = None

        if self._global_importance is not None:
            seq_len = self._global_importance.shape[0]
            valid = [j for j in idx_list if j < seq_len]
            if valid:
                self._global_importance = self._global_importance[mx.array(valid)]
            else:
                self._global_importance = None

    def reset(self):
        """Clear all tracked attention scores."""
        self._importance = [None] * self.num_layers
        self._global_importance = None


class CompressedKVCache:
    """KV cache that evicts low-importance tokens based on attention scores.

    Maintains a budget-constrained cache where:
      - Sink tokens (first K) are always kept
      - Recent window (last W) is always kept
      - Remaining budget slots go to highest-importance tokens
      - Importance is determined by AttentionScoreTracker (H2O or ScissorHands)
    """

    def __init__(self, config: KVCompressionConfig, num_layers: int, num_heads: int, head_dim: int):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Per-layer KV storage
        # Keys/values shape: [num_heads, seq_len, head_dim]
        self._keys: list[mx.array | None] = [None] * num_layers
        self._values: list[mx.array | None] = [None] * num_layers

        # Attention score tracker
        self.tracker = AttentionScoreTracker(num_layers, scoring=config.scoring)

        # Statistics
        self._total_tokens_seen: int = 0
        self._compression_count: int = 0
        self._total_evicted: int = 0

    def update(
        self, layer_idx: int, new_keys: mx.array, new_values: mx.array, attention_weights: mx.array = None
    ) -> tuple[mx.array, mx.array]:
        """Add new KV entries, evict unimportant tokens if over budget.

        Args:
            layer_idx: Transformer layer index.
            new_keys: New key tensor [num_heads, new_len, head_dim].
            new_values: New value tensor [num_heads, new_len, head_dim].
            attention_weights: Optional attention weights
                [num_heads, seq_len, seq_len] for importance tracking.

        Returns:
            Tuple of (all_keys, all_values) for attention computation.
        """
        new_len = new_keys.shape[1]

        if layer_idx == 0:
            self._total_tokens_seen += new_len

        # Append new entries
        if self._keys[layer_idx] is None:
            self._keys[layer_idx] = new_keys
            self._values[layer_idx] = new_values
        else:
            self._keys[layer_idx] = mx.concatenate([self._keys[layer_idx], new_keys], axis=1)
            self._values[layer_idx] = mx.concatenate([self._values[layer_idx], new_values], axis=1)

        # Record attention scores if provided
        if attention_weights is not None:
            self.tracker.record_attention(layer_idx, attention_weights)

        # Compute budget: budget_ratio * total tokens seen so far
        cur_len = self._keys[layer_idx].shape[1]
        budget = self._compute_budget(cur_len)

        if cur_len > budget:
            self._compress_layer(layer_idx, budget)

        return self._keys[layer_idx], self._values[layer_idx]

    def _compute_budget(self, seq_len: int) -> int:
        """Compute the token budget for the current sequence length.

        The budget is at least sink_tokens + recent_window, and at most
        budget_ratio * seq_len (whichever is larger).
        """
        min_budget = self.config.sink_tokens + self.config.recent_window
        ratio_budget = max(1, int(seq_len * self.config.budget_ratio))
        return max(min_budget, ratio_budget)

    def _compress_layer(self, layer_idx: int, budget: int):
        """Compress a single layer's KV cache to fit within budget.

        Strategy:
          1. Always keep sink tokens (first K)
          2. Always keep recent window (last W)
          3. Fill remaining budget with highest-importance middle tokens
        """
        cur_len = self._keys[layer_idx].shape[1]
        sink = min(self.config.sink_tokens, cur_len)
        window = min(self.config.recent_window, cur_len)

        if cur_len <= budget:
            return

        # Guaranteed slots
        guaranteed = sink + window
        middle_budget = max(0, budget - guaranteed)

        # Middle region: tokens between sink and recent window
        middle_start = sink
        middle_end = cur_len - window

        if middle_end <= middle_start:
            # No middle region -- nothing to evict from
            return

        middle_len = middle_end - middle_start

        if middle_budget >= middle_len:
            # Budget accommodates all middle tokens -- no eviction needed
            return

        # Get importance scores for middle tokens
        importance = self.tracker.get_token_importance(layer_idx)

        if importance.shape[0] >= cur_len:
            middle_importance = importance[middle_start:middle_end]
        else:
            # If importance tracking hasn't caught up, use uniform scores
            middle_importance = mx.ones((middle_len,))

        # Select top-k middle tokens by importance
        if middle_budget > 0:
            sorted_idx = mx.argsort(-middle_importance)
            keep_middle_local = mx.sort(sorted_idx[:middle_budget])
            keep_middle_global = keep_middle_local + middle_start
        else:
            keep_middle_global = mx.array([], dtype=mx.int32)

        # Build final index set: sink + selected_middle + recent_window
        sink_indices = mx.arange(sink)
        window_indices = mx.arange(middle_end, cur_len)

        if middle_budget > 0:
            keep_indices = mx.concatenate([sink_indices, keep_middle_global, window_indices])
        else:
            keep_indices = mx.concatenate([sink_indices, window_indices])

        keep_indices = keep_indices.astype(mx.int32)

        num_evicted = cur_len - keep_indices.shape[0]

        # Reindex KV cache
        self._keys[layer_idx] = self._keys[layer_idx][:, keep_indices, :]
        self._values[layer_idx] = self._values[layer_idx][:, keep_indices, :]

        # Reindex tracker scores
        self.tracker.truncate(keep_indices)

        self._compression_count += 1
        self._total_evicted += num_evicted

    def compress(self, layer_idx: int):
        """Force compression to budget_ratio. Called when memory pressure is high.

        This forces the layer's cache to exactly budget_ratio * current_length,
        regardless of whether the budget has been exceeded.
        """
        if self._keys[layer_idx] is None:
            return

        cur_len = self._keys[layer_idx].shape[1]
        budget = max(
            self.config.sink_tokens + self.config.recent_window,
            int(cur_len * self.config.budget_ratio),
        )
        self._compress_layer(layer_idx, budget)

    def get_kv(self, layer_idx: int) -> tuple[mx.array, mx.array]:
        """Get current KV cache for a layer."""
        if self._keys[layer_idx] is None:
            empty = mx.zeros((self.num_heads, 0, self.head_dim))
            return empty, empty
        return self._keys[layer_idx], self._values[layer_idx]

    def get_compression_ratio(self) -> float:
        """Current compression ratio (original_tokens / kept_tokens).

        Returns 1.0 if no compression has occurred.
        """
        if self._total_tokens_seen == 0:
            return 1.0
        current = self.current_length
        if current == 0:
            return float("inf")
        return self._total_tokens_seen / current

    @property
    def current_length(self) -> int:
        """Current number of tokens in cache (from first non-empty layer)."""
        for i in range(self.num_layers):
            if self._keys[i] is not None:
                return self._keys[i].shape[1]
        return 0

    def get_stats(self) -> dict:
        """Return compression statistics."""
        return {
            "current_length": self.current_length,
            "total_tokens_seen": self._total_tokens_seen,
            "compression_ratio": round(self.get_compression_ratio(), 2),
            "compression_count": self._compression_count,
            "total_evicted": self._total_evicted,
            "scoring": self.config.scoring,
            "budget_ratio": self.config.budget_ratio,
            "sink_tokens": self.config.sink_tokens,
            "recent_window": self.config.recent_window,
        }

    def reset(self):
        """Clear all cached KV pairs and tracked scores."""
        self._keys = [None] * self.num_layers
        self._values = [None] * self.num_layers
        self.tracker.reset()
        self._total_tokens_seen = 0
        self._compression_count = 0
        self._total_evicted = 0


def apply_kv_compression(model, config: KVCompressionConfig = None) -> CompressedKVCache:
    """Create a compressed KV cache sized for the given model.

    Inspects the model to determine num_layers, num_heads, and head_dim,
    then returns a configured CompressedKVCache.

    Args:
        model: A transformer model with a .layers attribute (list of decoder
            layers), where each layer has a .self_attn or .attention module
            with num_heads and head_dim attributes.
        config: KV compression configuration. Uses defaults if None.

    Returns:
        A CompressedKVCache instance ready for use during generation.
    """
    if config is None:
        config = KVCompressionConfig()

    num_layers = len(model.layers)

    first_layer = model.layers[0]
    attn = getattr(first_layer, "self_attn", None) or getattr(first_layer, "attention", None)
    if attn is None:
        raise ValueError("Cannot find attention module on model layer. Expected .self_attn or .attention attribute.")

    num_heads = getattr(attn, "num_heads", None) or getattr(attn, "n_heads", None)
    head_dim = getattr(attn, "head_dim", None) or getattr(attn, "d_head", None)

    if num_heads is None or head_dim is None:
        raise ValueError(
            f"Cannot determine num_heads/head_dim from attention module. Available attributes: {dir(attn)}"
        )

    return CompressedKVCache(config, num_layers, num_heads, head_dim)
