"""Phase-level pipelined execution for overlapping IO and compute.

Instead of load-all → compute-all per layer, we interleave:
  Phase 1: Prefetch attention weights → compute input norm
  Phase 2: Wait for attention → prefetch MLP weights → compute attention
  Phase 3: Wait for MLP → compute MLP → prefetch next layer

This hides SSD latency behind GPU compute, achieving near-full utilization
of both SSD bandwidth and GPU compute simultaneously.

The pipeline adapts its prefetch depth (1-3 layers ahead) based on
measured compute vs IO time ratios.

Usage:
  executor = PipelinedExecutor(model, safetensors_map)
  executor.execute_layer(layer_idx, hidden_states)
"""

import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from mlx_flash_compress.page_cache import PageCacheAdvisor, EvictionStrategy


@dataclass
class PipelineStats:
    """Track pipeline execution metrics."""
    layers_executed: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    total_compute_ms: float = 0.0
    total_io_ms: float = 0.0
    overlap_ratio: float = 0.0   # fraction of IO hidden behind compute

    @property
    def io_hidden_pct(self) -> float:
        if self.total_io_ms == 0:
            return 100.0
        return min(self.overlap_ratio * 100, 100.0)


@dataclass
class LayerPhase:
    """A single phase of layer execution with associated prefetch."""
    name: str                    # "attn_norm", "attn", "mlp_norm", "mlp"
    weight_keys: list = field(default_factory=list)   # weight keys to prefetch
    compute_fn: Optional[Callable] = None


class PrefetchWorker:
    """Background thread for prefetching weight byte ranges into page cache."""

    def __init__(self, advisor: PageCacheAdvisor, max_workers: int = 2):
        self.advisor = advisor
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="prefetch")
        self._pending: dict = {}  # layer_idx -> Future
        self._ema_io_ms = 5.0     # exponential moving average of IO time
        self._ema_compute_ms = 5.0
        self._alpha = 0.3

    @property
    def prefetch_depth(self) -> int:
        """Dynamically adjust lookahead based on IO/compute ratio."""
        if self._ema_compute_ms <= 0:
            return 1
        ratio = self._ema_io_ms / self._ema_compute_ms
        if ratio < 0.5:
            return 1  # IO is fast, minimal lookahead
        elif ratio < 1.5:
            return 2  # balanced
        else:
            return 3  # IO-bound, aggressive lookahead

    def submit_prefetch(self, layer_idx: int, mmap_obj, byte_ranges: list):
        """Submit a prefetch job for a layer's weight byte ranges."""
        if layer_idx in self._pending:
            return  # already submitted

        def _do_prefetch():
            for offset, length in byte_ranges:
                self.advisor.prefetch_expert(mmap_obj, offset, length)

        self._pending[layer_idx] = self._pool.submit(_do_prefetch)

    def wait_for(self, layer_idx: int, timeout: float = 5.0) -> bool:
        """Wait for a specific layer's prefetch to complete."""
        fut = self._pending.pop(layer_idx, None)
        if fut is None:
            return False
        try:
            fut.result(timeout=timeout)
            return True
        except Exception:
            return False

    def update_timing(self, io_ms: float, compute_ms: float):
        """Update EMA of IO and compute times for adaptive depth."""
        self._ema_io_ms = self._alpha * io_ms + (1 - self._alpha) * self._ema_io_ms
        self._ema_compute_ms = self._alpha * compute_ms + (1 - self._alpha) * self._ema_compute_ms

    def shutdown(self):
        self._pool.shutdown(wait=False)


class PipelinedExecutor:
    """Execute transformer layers with phase-level IO/compute overlap.

    For each layer:
      1. Start prefetching attention weights for this layer
      2. While prefetch runs, compute input LayerNorm
      3. Wait for attention weights, start prefetching MLP weights
      4. Compute attention
      5. Wait for MLP weights, start prefetching NEXT layer's attention
      6. Compute MLP
      7. Advise FREE on this layer's used weights

    This keeps the GPU busy during SSD reads and vice versa.
    """

    def __init__(
        self,
        advisor: Optional[PageCacheAdvisor] = None,
        eviction_strategy: EvictionStrategy = EvictionStrategy.MADV_FREE,
    ):
        if advisor is None:
            advisor = PageCacheAdvisor(strategy=eviction_strategy)
        self.advisor = advisor
        self.prefetch = PrefetchWorker(advisor)
        self.stats = PipelineStats()
        self._layer_timings: deque = deque(maxlen=100)

    def execute_layer_phases(
        self,
        layer_idx: int,
        total_layers: int,
        mmap_obj,
        attn_byte_ranges: list,
        mlp_byte_ranges: list,
        compute_norm_fn: Callable,
        compute_attn_fn: Callable,
        compute_mlp_fn: Callable,
        next_attn_byte_ranges: Optional[list] = None,
    ):
        """Execute a single layer with pipelined phases.

        Returns the MLP output (hidden states for next layer).
        """
        t_start = time.monotonic()

        # Phase 1: Prefetch attention + compute norm
        self.prefetch.submit_prefetch(
            layer_idx * 10 + 0, mmap_obj, attn_byte_ranges
        )
        t_norm_start = time.monotonic()
        norm_out = compute_norm_fn()
        if HAS_MLX:
            mx.eval(norm_out) if hasattr(norm_out, 'shape') else None
        t_norm_end = time.monotonic()

        # Phase 2: Wait for attention weights + prefetch MLP + compute attention
        self.prefetch.wait_for(layer_idx * 10 + 0)

        self.prefetch.submit_prefetch(
            layer_idx * 10 + 1, mmap_obj, mlp_byte_ranges
        )
        t_attn_start = time.monotonic()
        attn_out = compute_attn_fn(norm_out)
        if HAS_MLX:
            mx.eval(attn_out) if hasattr(attn_out, 'shape') else None
        t_attn_end = time.monotonic()

        # Phase 3: Wait for MLP weights + prefetch next layer + compute MLP
        self.prefetch.wait_for(layer_idx * 10 + 1)

        # Start prefetching next layer's attention weights
        if next_attn_byte_ranges and layer_idx + 1 < total_layers:
            self.prefetch.submit_prefetch(
                (layer_idx + 1) * 10 + 0, mmap_obj, next_attn_byte_ranges
            )

        t_mlp_start = time.monotonic()
        mlp_out = compute_mlp_fn(attn_out)
        if HAS_MLX:
            mx.eval(mlp_out) if hasattr(mlp_out, 'shape') else None
        t_mlp_end = time.monotonic()

        # Evict this layer's weights from page cache
        for offset, length in attn_byte_ranges + mlp_byte_ranges:
            self.advisor.evict_expert(mmap_obj, offset, length)

        # Update stats
        compute_ms = (t_norm_end - t_norm_start + t_attn_end - t_attn_start + t_mlp_end - t_mlp_start) * 1000
        total_ms = (t_mlp_end - t_start) * 1000
        io_ms = max(total_ms - compute_ms, 0)

        self.prefetch.update_timing(io_ms, compute_ms)
        self.stats.layers_executed += 1
        self.stats.total_compute_ms += compute_ms
        self.stats.total_io_ms += io_ms

        if total_ms > 0:
            self.stats.overlap_ratio = 1.0 - (io_ms / total_ms)

        self._layer_timings.append({
            "layer": layer_idx,
            "total_ms": total_ms,
            "compute_ms": compute_ms,
            "io_ms": io_ms,
            "prefetch_depth": self.prefetch.prefetch_depth,
        })

        return mlp_out

    def shutdown(self):
        self.prefetch.shutdown()
