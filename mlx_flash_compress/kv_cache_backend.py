"""Unified KV cache backend abstraction.

Provides a common interface for different KV cache strategies:
  - PlainKVCache: passthrough, stores raw K/V (baseline)
  - StreamingKVCache: wraps StreamingLLMCache for attention-sink eviction
  - QuantizedKVCache: wraps QuantizedKVCacheManager for 4-bit storage
  - HybridKVCache: chains quantization + streaming eviction

Factory function ``create_kv_cache()`` instantiates any strategy by name.
``install_kv_cache()`` hooks into a model's attention layers for
observation-only KV capture.

Usage:
    backend = create_kv_cache("hybrid", num_layers=32,
                              num_heads=8, head_dim=128,
                              window_size=1024, key_bits=4)
    handle = install_kv_cache(model, backend)
    # ... run inference ...
    handle.uninstall()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import mlx.core as mx

from mlx_flash_compress.quantized_kv_cache import (
    QuantizedKVCacheManager,
    QuantizedKVConfig,
)
from mlx_flash_compress.streaming_llm import (
    StreamingLLMCache,
    StreamingLLMConfig,
)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class KVCacheBackend(ABC):
    """Abstract interface for KV cache storage strategies.

    All backends accept keys/values of shape ``[num_heads, seq_len, head_dim]``
    (the convention used by StreamingLLMCache).
    """

    @abstractmethod
    def update(self, layer_idx: int, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Store new KV entries for *layer_idx* and return the full cache.

        Args:
            layer_idx: Transformer layer index.
            keys: New key tensor, shape ``[num_heads, new_len, head_dim]``.
            values: New value tensor, same shape as *keys*.

        Returns:
            ``(all_keys, all_values)`` covering the entire cached sequence.
        """

    @abstractmethod
    def get_kv(self, layer_idx: int) -> tuple[mx.array, mx.array]:
        """Retrieve the current KV cache for *layer_idx*."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all cached KV pairs across every layer."""

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return backend-specific statistics."""


# ---------------------------------------------------------------------------
# Plain (passthrough) backend
# ---------------------------------------------------------------------------


class PlainKVCache(KVCacheBackend):
    """Baseline backend that stores raw K/V without compression or eviction."""

    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._keys: list[mx.array | None] = [None] * num_layers
        self._values: list[mx.array | None] = [None] * num_layers

    def update(self, layer_idx: int, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if self._keys[layer_idx] is None:
            self._keys[layer_idx] = keys
            self._values[layer_idx] = values
        else:
            self._keys[layer_idx] = mx.concatenate([self._keys[layer_idx], keys], axis=1)
            self._values[layer_idx] = mx.concatenate([self._values[layer_idx], values], axis=1)
        return self._keys[layer_idx], self._values[layer_idx]

    def get_kv(self, layer_idx: int) -> tuple[mx.array, mx.array]:
        if self._keys[layer_idx] is None:
            empty = mx.zeros((self.num_heads, 0, self.head_dim))
            return empty, empty
        return self._keys[layer_idx], self._values[layer_idx]

    def reset(self) -> None:
        self._keys = [None] * self.num_layers
        self._values = [None] * self.num_layers

    def get_stats(self) -> dict[str, Any]:
        total_tokens = 0
        for k in self._keys:
            if k is not None:
                total_tokens += k.shape[1]
        return {
            "strategy": "plain",
            "num_layers": self.num_layers,
            "total_tokens": total_tokens,
        }


# ---------------------------------------------------------------------------
# Streaming backend (attention-sink eviction)
# ---------------------------------------------------------------------------


class StreamingKVCache(KVCacheBackend):
    """Wraps :class:`StreamingLLMCache` behind the unified interface."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        *,
        num_sink_tokens: int = 4,
        window_size: int = 1024,
        eviction_batch: int = 256,
    ):
        config = StreamingLLMConfig(
            num_sink_tokens=num_sink_tokens,
            window_size=window_size,
            eviction_batch=eviction_batch,
        )
        self._cache = StreamingLLMCache(config, num_layers, num_heads, head_dim)

    def update(self, layer_idx: int, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        return self._cache.update(layer_idx, keys, values)

    def get_kv(self, layer_idx: int) -> tuple[mx.array, mx.array]:
        return self._cache.get_kv(layer_idx)

    def reset(self) -> None:
        self._cache.reset()

    def get_stats(self) -> dict[str, Any]:
        stats = self._cache.get_stats()
        stats["strategy"] = "streaming"
        return stats


# ---------------------------------------------------------------------------
# Quantized backend (4-bit KV storage)
# ---------------------------------------------------------------------------


class QuantizedKVCache(KVCacheBackend):
    """Wraps :class:`QuantizedKVCacheManager` behind the unified interface.

    Note: ``QuantizedKVCacheManager`` expects tensors shaped
    ``[seq_len, num_heads, head_dim]`` while the unified interface uses
    ``[num_heads, seq_len, head_dim]``.  This wrapper transposes on the
    boundary so callers always use the ``[heads, seq, dim]`` convention.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        *,
        key_bits: int = 4,
        value_bits: int = 4,
        group_size: int = 64,
        calibration_tokens: int = 32,
    ):
        config = QuantizedKVConfig(
            key_bits=key_bits,
            value_bits=value_bits,
            group_size=group_size,
            calibration_tokens=calibration_tokens,
        )
        self._manager = QuantizedKVCacheManager(config, num_layers, num_heads, head_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def update(self, layer_idx: int, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        # Transpose [heads, seq, dim] -> [seq, heads, dim] for the manager
        keys_t = mx.transpose(keys, axes=(1, 0, 2))
        values_t = mx.transpose(values, axes=(1, 0, 2))
        self._manager.update(layer_idx, keys_t, values_t)
        return self.get_kv(layer_idx)

    def get_kv(self, layer_idx: int) -> tuple[mx.array, mx.array]:
        try:
            k, v = self._manager.get_kv(layer_idx)
        except ValueError:
            empty = mx.zeros((self.num_heads, 0, self.head_dim))
            return empty, empty
        # Transpose [seq, heads, dim] -> [heads, seq, dim]
        return mx.transpose(k, axes=(1, 0, 2)), mx.transpose(v, axes=(1, 0, 2))

    def reset(self) -> None:
        self._manager.reset()

    def get_stats(self) -> dict[str, Any]:
        stats = self._manager.get_stats()
        stats["strategy"] = "quantized"
        return stats


# ---------------------------------------------------------------------------
# Hybrid backend (quantize + streaming eviction)
# ---------------------------------------------------------------------------


class HybridKVCache(KVCacheBackend):
    """Chains quantized storage with streaming-LLM eviction.

    New KV pairs are first quantized (via :class:`QuantizedKVCacheManager`)
    and then the eviction policy of :class:`StreamingLLMCache` decides
    which tokens to keep.  The effective pipeline is:

        incoming K/V -> quantize -> store -> evict (sink + window policy)

    Internally the hybrid backend stores *dequantized* values in the
    streaming cache so that eviction operates on uniform-shape tensors.
    This means memory savings come from the quantize/dequantize round-trip
    (reduced precision) while the eviction bounds the sequence length.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        *,
        num_sink_tokens: int = 4,
        window_size: int = 1024,
        eviction_batch: int = 256,
        key_bits: int = 4,
        value_bits: int = 4,
        group_size: int = 64,
        calibration_tokens: int = 32,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Quantization layer — used to compress each incoming chunk
        self._quant_config = QuantizedKVConfig(
            key_bits=key_bits,
            value_bits=value_bits,
            group_size=group_size,
            calibration_tokens=calibration_tokens,
        )
        self._quant_manager = QuantizedKVCacheManager(self._quant_config, num_layers, num_heads, head_dim)

        # Eviction layer — streaming cache that decides what to keep
        streaming_config = StreamingLLMConfig(
            num_sink_tokens=num_sink_tokens,
            window_size=window_size,
            eviction_batch=eviction_batch,
        )
        self._stream_cache = StreamingLLMCache(streaming_config, num_layers, num_heads, head_dim)

        self._update_count = 0

    def update(self, layer_idx: int, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if layer_idx == 0:
            self._update_count += 1

        # Step 1: quantize + dequantize to simulate lossy compression
        keys_t = mx.transpose(keys, axes=(1, 0, 2))
        values_t = mx.transpose(values, axes=(1, 0, 2))
        self._quant_manager.update(layer_idx, keys_t, values_t)
        qk, qv = self._quant_manager.get_kv(layer_idx)
        # Back to [heads, seq, dim]
        qk = mx.transpose(qk, axes=(1, 0, 2))
        qv = mx.transpose(qv, axes=(1, 0, 2))

        # We only want the *new* tokens (last chunk) after dequantization.
        # The quantized manager stores the full history; extract the tail.
        new_len = keys.shape[1]
        qk_new = qk[:, -new_len:, :]
        qv_new = qv[:, -new_len:, :]

        # Step 2: feed the (lossy) chunk into the streaming cache for eviction
        return self._stream_cache.update(layer_idx, qk_new, qv_new)

    def get_kv(self, layer_idx: int) -> tuple[mx.array, mx.array]:
        return self._stream_cache.get_kv(layer_idx)

    def reset(self) -> None:
        self._quant_manager.reset()
        self._stream_cache.reset()
        self._update_count = 0

    def get_stats(self) -> dict[str, Any]:
        stream_stats = self._stream_cache.get_stats()
        quant_stats = self._quant_manager.get_stats()
        return {
            "strategy": "hybrid",
            "update_count": self._update_count,
            "streaming": stream_stats,
            "quantization": quant_stats,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_kv_cache(
    strategy: str = "plain",
    num_layers: int = 32,
    num_heads: int = 8,
    head_dim: int = 128,
    **kwargs: Any,
) -> KVCacheBackend:
    """Create a KV cache backend by strategy name.

    Args:
        strategy: One of ``"plain"``, ``"streaming"``, ``"quantized"``,
            or ``"hybrid"``.
        num_layers: Number of transformer layers.
        num_heads: Number of KV heads (may equal num_attention_heads for MHA).
        head_dim: Dimension per head.
        **kwargs: Strategy-specific options forwarded to the backend
            constructor (e.g. ``window_size``, ``key_bits``).

    Returns:
        A :class:`KVCacheBackend` instance.
    """
    _streaming_keys = {"num_sink_tokens", "window_size", "eviction_batch"}
    _quantized_keys = {"key_bits", "value_bits", "group_size", "calibration_tokens"}

    if strategy == "plain":
        return PlainKVCache(num_layers, num_heads, head_dim)
    if strategy == "streaming":
        filtered = {k: v for k, v in kwargs.items() if k in _streaming_keys}
        return StreamingKVCache(num_layers, num_heads, head_dim, **filtered)
    if strategy == "quantized":
        filtered = {k: v for k, v in kwargs.items() if k in _quantized_keys}
        return QuantizedKVCache(num_layers, num_heads, head_dim, **filtered)
    if strategy == "hybrid":
        filtered = {k: v for k, v in kwargs.items() if k in _streaming_keys | _quantized_keys}
        return HybridKVCache(num_layers, num_heads, head_dim, **filtered)
    raise ValueError(f"Unknown KV cache strategy {strategy!r}. Choose from: plain, streaming, quantized, hybrid")


# ---------------------------------------------------------------------------
# Model installation (observation-only)
# ---------------------------------------------------------------------------


class _AttentionWrapper:
    """Callable wrapper that intercepts attention output to capture KV pairs."""

    def __init__(self, original_attn: Any, layer_idx: int, backend: KVCacheBackend):
        self._original = original_attn
        self._layer_idx = layer_idx
        self._backend = backend

    def __call__(self, *args, **kwargs):
        result = self._original(*args, **kwargs)
        # Try to extract K/V from the result.  Many MLX attention
        # modules return (output, (keys, values)) or just output.
        # We observe opportunistically -- if we can't find KV, skip.
        if isinstance(result, tuple) and len(result) >= 2:
            maybe_kv = result[1]
            if isinstance(maybe_kv, tuple) and len(maybe_kv) == 2:
                k, v = maybe_kv
                if isinstance(k, mx.array) and isinstance(v, mx.array):
                    self._backend.update(self._layer_idx, k, v)
        return result

    def __getattr__(self, name):
        # Proxy all other attribute access to the original attention module
        return getattr(self._original, name)


class _KVCacheHandle:
    """Handle returned by :func:`install_kv_cache` to allow uninstallation."""

    def __init__(self):
        self._originals: list[tuple[Any, str, Any]] = []

    def _record(self, layer: Any, attr_name: str, original_attn: Any) -> None:
        self._originals.append((layer, attr_name, original_attn))

    def uninstall(self) -> None:
        """Restore original attention modules on all patched layers."""
        for layer, attr_name, original_attn in self._originals:
            setattr(layer, attr_name, original_attn)
        self._originals.clear()


def _get_layers(model):
    """Extract transformer layers, handling ``model.model.layers`` nesting."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    return None


def install_kv_cache(model: Any, backend: KVCacheBackend) -> _KVCacheHandle:
    """Hook into a model's attention layers for observation-only KV capture.

    Replaces each layer's attention module with a thin wrapper that,
    after the original forward pass, captures the returned K/V tensors
    and passes them to ``backend.update(layer_idx, ...)``.  The model's
    own KV cache is unaffected -- this is purely observational.

    Args:
        model: A transformer model with a ``layers`` attribute.
        backend: The :class:`KVCacheBackend` to feed captured KV data into.

    Returns:
        A :class:`_KVCacheHandle` whose ``.uninstall()`` method restores
        the original attention modules.
    """
    layers = _get_layers(model)
    if layers is None:
        raise ValueError("Cannot find model layers. Expected model.layers or model.model.layers.")

    handle = _KVCacheHandle()

    for layer_idx, layer in enumerate(layers):
        attr_name = None
        original_attn = None
        for candidate in ("self_attn", "attention"):
            if hasattr(layer, candidate):
                attr_name = candidate
                original_attn = getattr(layer, candidate)
                break

        if original_attn is None:
            continue

        wrapper = _AttentionWrapper(original_attn, layer_idx, backend)
        setattr(layer, attr_name, wrapper)
        handle._record(layer, attr_name, original_attn)

    return handle
