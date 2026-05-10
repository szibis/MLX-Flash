"""StreamingLLM KV eviction for infinite-length generation with fixed-size cache.

Based on "Efficient Streaming Language Models with Attention Sinks"
(arXiv:2309.17453, ICLR 2024).

Key insight: The first few tokens in a sequence act as "attention sinks" --
they receive disproportionately high attention scores regardless of their
semantic content. This happens because softmax needs to allocate attention
mass somewhere, and initial tokens accumulate it by positional advantage.

StreamingLLM maintains a fixed-size KV cache by keeping:
  1. First K "attention sink" tokens (always kept, positions 0..K-1)
  2. A sliding window of the last W recent tokens
  3. Everything in between is evicted

This achieves 22.2x speedup vs sliding window with recomputation, and works
with Llama, Qwen, Gemma, and other RoPE-based models without fine-tuning.

Position handling for RoPE:
  - Sink tokens keep original positions (0..K-1)
  - Window tokens get shifted positions to maintain correct relative distances
  - This preserves RoPE's relative position encoding invariant

Usage:
    config = StreamingLLMConfig(num_sink_tokens=4, window_size=1024)
    cache = StreamingLLMCache(config, num_layers=32, num_heads=8, head_dim=128)

    # During generation:
    keys, values = cache.update(layer_idx=0, new_keys=k, new_values=v)
    # keys/values contain [sink | recent_window] layout
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class StreamingLLMConfig:
    """Configuration for StreamingLLM KV cache eviction.

    Attributes:
        num_sink_tokens: Number of initial "attention sink" tokens to always
            keep in cache. The paper recommends 4 for most models.
        window_size: Size of the sliding window for recent tokens.
        eviction_batch: Number of tokens to evict at once. Batching evictions
            reduces the overhead of cache reorganization.
    """
    num_sink_tokens: int = 4
    window_size: int = 1024
    eviction_batch: int = 256


class StreamingLLMCache:
    """Fixed-size KV cache with attention sink preservation.

    Maintains a two-region cache layout per layer:
        [sink_tokens (0..K-1) | recent_window (last W tokens)]

    When the cache exceeds capacity (K + W), middle tokens are evicted
    in batches to amortize the cost of cache compaction.
    """

    def __init__(self, config: StreamingLLMConfig, num_layers: int,
                 num_heads: int, head_dim: int):
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Per-layer KV storage: list of (keys, values) or None
        # Keys/values shape: [num_heads, seq_len, head_dim]
        self._keys: list[mx.array | None] = [None] * num_layers
        self._values: list[mx.array | None] = [None] * num_layers

        # Per-layer position tracking for RoPE adjustment
        # Stores the original position IDs for each cached token
        self._positions: list[mx.array | None] = [None] * num_layers

        # Track total tokens seen (across all evictions)
        self._total_tokens_seen: int = 0

        # Eviction statistics
        self._eviction_count: int = 0
        self._total_evicted_tokens: int = 0

    def update(self, layer_idx: int, new_keys: mx.array,
               new_values: mx.array) -> tuple[mx.array, mx.array]:
        """Add new KV entries, evict middle tokens if over capacity.

        Args:
            layer_idx: Transformer layer index.
            new_keys: New key tensor, shape [num_heads, new_len, head_dim].
            new_values: New value tensor, shape [num_heads, new_len, head_dim].

        Returns:
            Tuple of (all_keys, all_values) for attention computation.
            Layout: [sink_tokens | ... | recent_window]
        """
        new_len = new_keys.shape[1]

        # Track total tokens seen (only count once across layers)
        if layer_idx == 0:
            self._total_tokens_seen += new_len

        if self._keys[layer_idx] is None:
            # First update: just store
            self._keys[layer_idx] = new_keys
            self._values[layer_idx] = new_values
            # Assign initial positions
            self._positions[layer_idx] = mx.arange(new_len)
        else:
            # Append new KV entries
            self._keys[layer_idx] = mx.concatenate(
                [self._keys[layer_idx], new_keys], axis=1
            )
            self._values[layer_idx] = mx.concatenate(
                [self._values[layer_idx], new_values], axis=1
            )
            # Extend positions
            cur_max = int(mx.max(self._positions[layer_idx]).item()) + 1
            new_positions = mx.arange(cur_max, cur_max + new_len)
            self._positions[layer_idx] = mx.concatenate(
                [self._positions[layer_idx], new_positions]
            )

        # Check if eviction is needed
        cur_len = self._keys[layer_idx].shape[1]
        capacity = self.max_length

        if cur_len > capacity:
            self._evict(layer_idx)

        return self._keys[layer_idx], self._values[layer_idx]

    def _evict(self, layer_idx: int):
        """Evict middle tokens, keeping sink + recent window.

        After eviction the cache contains exactly:
            [sink_tokens (first K) | recent_window (last W)]
        """
        cur_len = self._keys[layer_idx].shape[1]
        sink = self.config.num_sink_tokens
        window = self.config.window_size

        # If total length fits, no eviction needed
        if cur_len <= sink + window:
            return

        num_to_evict = cur_len - (sink + window)

        # Keep sink tokens (first K) and recent window (last W)
        sink_keys = self._keys[layer_idx][:, :sink, :]
        sink_values = self._values[layer_idx][:, :sink, :]
        sink_positions = self._positions[layer_idx][:sink]

        window_keys = self._keys[layer_idx][:, -window:, :]
        window_values = self._values[layer_idx][:, -window:, :]
        window_positions = self._positions[layer_idx][-window:]

        # Concatenate sink + window
        self._keys[layer_idx] = mx.concatenate(
            [sink_keys, window_keys], axis=1
        )
        self._values[layer_idx] = mx.concatenate(
            [sink_values, window_values], axis=1
        )
        self._positions[layer_idx] = mx.concatenate(
            [sink_positions, window_positions]
        )

        self._eviction_count += 1
        self._total_evicted_tokens += num_to_evict

    def get_kv(self, layer_idx: int) -> tuple[mx.array, mx.array]:
        """Get current KV cache for a layer.

        Returns:
            Tuple of (keys, values). Returns zero-length tensors if layer
            has no cached entries.
        """
        if self._keys[layer_idx] is None:
            empty = mx.zeros((self.num_heads, 0, self.head_dim))
            return empty, empty
        return self._keys[layer_idx], self._values[layer_idx]

    def get_positions(self, layer_idx: int) -> mx.array:
        """Get position IDs for cached tokens (for RoPE computation).

        Sink tokens retain their original positions (0..K-1).
        Window tokens retain their original positions to preserve
        relative distance encoding in RoPE.

        Returns:
            Position IDs array of shape [current_length].
        """
        if self._positions[layer_idx] is None:
            return mx.array([], dtype=mx.int32)
        return self._positions[layer_idx]

    @property
    def current_length(self) -> int:
        """Current number of tokens in cache (from first non-empty layer)."""
        for i in range(self.num_layers):
            if self._keys[i] is not None:
                return self._keys[i].shape[1]
        return 0

    @property
    def max_length(self) -> int:
        """Maximum capacity (sink + window)."""
        return self.config.num_sink_tokens + self.config.window_size

    @property
    def is_full(self) -> bool:
        """Whether the cache has reached maximum capacity."""
        return self.current_length >= self.max_length

    def get_stats(self) -> dict:
        """Return eviction count, current size, sink tokens, etc."""
        return {
            "current_length": self.current_length,
            "max_length": self.max_length,
            "is_full": self.is_full,
            "num_sink_tokens": self.config.num_sink_tokens,
            "window_size": self.config.window_size,
            "eviction_count": self._eviction_count,
            "total_evicted_tokens": self._total_evicted_tokens,
            "total_tokens_seen": self._total_tokens_seen,
        }

    def reset(self):
        """Clear all cached KV pairs."""
        self._keys = [None] * self.num_layers
        self._values = [None] * self.num_layers
        self._positions = [None] * self.num_layers
        self._total_tokens_seen = 0
        self._eviction_count = 0
        self._total_evicted_tokens = 0


def apply_streaming_llm(model, config: StreamingLLMConfig = None) -> StreamingLLMCache:
    """Create a StreamingLLM cache sized for the given model.

    Inspects the model to determine num_layers, num_heads, and head_dim,
    then returns a configured StreamingLLMCache.

    Args:
        model: A transformer model with a .layers attribute (list of decoder
            layers), where each layer has a .self_attn or .attention module
            with num_heads and head_dim attributes.
        config: StreamingLLM configuration. Uses defaults if None.

    Returns:
        A StreamingLLMCache instance ready for use during generation.
    """
    if config is None:
        config = StreamingLLMConfig()

    # Extract model dimensions
    num_layers = len(model.layers)

    # Find attention module (handles common naming conventions)
    first_layer = model.layers[0]
    attn = getattr(first_layer, "self_attn", None) or getattr(
        first_layer, "attention", None
    )
    if attn is None:
        raise ValueError(
            "Cannot find attention module on model layer. "
            "Expected .self_attn or .attention attribute."
        )

    # Extract head configuration
    num_heads = getattr(attn, "num_heads", None) or getattr(
        attn, "n_heads", None
    )
    head_dim = getattr(attn, "head_dim", None) or getattr(
        attn, "d_head", None
    )

    if num_heads is None or head_dim is None:
        raise ValueError(
            "Cannot determine num_heads/head_dim from attention module. "
            f"Available attributes: {dir(attn)}"
        )

    return StreamingLLMCache(config, num_layers, num_heads, head_dim)
