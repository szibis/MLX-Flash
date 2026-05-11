"""LCP (Least Critical Priority) eviction + async prefetch pipeline.

LCP eviction from mlx-moe (mu-hashmi/mlx-moe):
  Priority(expert) = frequency × 0.25^(steps_since_last_use / 128)

This combines frequency and recency with exponential decay — experts that
are both frequently AND recently used have highest priority. The 0.25 base
and 128 decay constant were tuned by mlx-moe for MoE routing patterns.

Async prefetch pipeline (the unfilled gap on Apple Silicon):
  Layer N:  GPU computes attention + routing
            ↕ PARALLEL (concurrent.futures)
            Background thread reads layer N+1 experts from disk
            Decompresses into staging buffer

  Layer N+1: Experts already in staging (zero wait on cache hit)
             ↕ PARALLEL
             Background prefetches layer N+2

Skip-fallback: When an expert is not cached and not prefetched in time,
zero its routing score and renormalize remaining experts. This trades
a small quality degradation for zero-latency guarantee.

Two-stage dendritic loading: Load only the gate projection (W1) first.
If the gate activation magnitude is below threshold, skip loading the
remaining projections (W2, W3) — saving ~66% of I/O for low-impact experts.
"""

import math
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class LCPEntry:
    """Cache entry with LCP priority tracking."""

    data: bytes
    layer_idx: int
    expert_id: int
    frequency: int = 0
    last_step: int = 0
    size_bytes: int = 0
    # For two-stage dendritic loading
    is_partial: bool = False  # True if only W1 (gate_proj) loaded
    gate_magnitude: float = 0.0


@dataclass
class PrefetchRequest:
    """A pending prefetch of expert data."""

    layer_idx: int
    expert_id: int
    future: Optional[Future] = None


@dataclass
class PipelineStats:
    """Statistics for the async prefetch pipeline."""

    cache_hits: int = 0
    prefetch_hits: int = 0
    cold_loads: int = 0
    skip_fallbacks: int = 0
    dendritic_skips: int = 0  # experts where W2/W3 loading was skipped
    total_requests: int = 0
    prefetch_accuracy: float = 0.0
    avg_priority: float = 0.0

    @property
    def hit_rate(self):
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits + self.prefetch_hits) / self.total_requests

    @property
    def skip_rate(self):
        if self.total_requests == 0:
            return 0.0
        return self.skip_fallbacks / self.total_requests


class LCPCache:
    """Expert cache with LCP eviction + async prefetch + dendritic loading.

    Usage:
        cache = LCPCache(
            expert_dir="path/to/experts/",
            capacity_bytes=2 * 1024**3,  # 2GB
            num_prefetch_workers=2,
        )

        # During inference:
        for token in range(max_tokens):
            cache.advance_step()
            for layer in range(num_layers):
                # Get routing decision
                expert_ids = router(hidden_state)

                # Fetch experts (cache hit, prefetch hit, or cold load)
                expert_data = cache.fetch(layer, expert_ids)

                # Kick off prefetch for next layer
                if layer < num_layers - 1:
                    predicted = cache.predict_next(layer, expert_ids)
                    cache.prefetch(layer + 1, predicted)
    """

    def __init__(
        self,
        expert_dir: str,
        capacity_bytes: int = 2 * 1024**3,
        num_prefetch_workers: int = 2,
        lcp_base: float = 0.25,
        lcp_decay: int = 128,
        dendritic_threshold: float = 0.01,
        enable_skip_fallback: bool = True,
        enable_dendritic: bool = True,
        simulated_ssd_latency_ms: float = 0.0,
    ):
        self.expert_dir = Path(expert_dir)
        self.capacity = capacity_bytes
        self.lcp_base = lcp_base
        self.lcp_decay = lcp_decay
        self.dendritic_threshold = dendritic_threshold
        self.enable_skip = enable_skip_fallback
        self.enable_dendritic = enable_dendritic
        self._ssd_latency_ms = simulated_ssd_latency_ms

        # Cache storage
        self._cache: dict[tuple[int, int], LCPEntry] = {}
        self._current_bytes: int = 0

        # Step counter
        self._step: int = 0

        # Prefetch state
        self._prefetch_pool = ThreadPoolExecutor(max_workers=num_prefetch_workers)
        self._prefetch_pending: dict[tuple[int, int], Future] = {}
        self._lock = threading.Lock()

        # Co-occurrence tracker for prediction
        self._cooccurrence: dict[int, dict[int, dict[int, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self._prev_experts: dict[int, list[int]] = {}

        # Stats
        self.stats = PipelineStats()

    def advance_step(self):
        """Call once per generated token."""
        self._step += 1

    def _priority(self, entry: LCPEntry) -> float:
        """LCP priority: frequency × base^(steps_since_last / decay)."""
        steps_since = self._step - entry.last_step
        return entry.frequency * (self.lcp_base ** (steps_since / self.lcp_decay))

    def _expert_path(self, layer_idx: int, expert_id: int) -> Path:
        return self.expert_dir / f"layer_{layer_idx:03d}" / f"expert_{expert_id:04d}.bin"

    def _read_expert(self, layer_idx: int, expert_id: int) -> bytes:
        """Read expert weights from SSD.

        Args:
            layer_idx: Transformer layer index.
            expert_id: Expert index within the layer.

        Returns:
            Raw bytes of the expert weight file.

        Raises:
            FileNotFoundError: If the expert weight file does not exist.
            OSError: If the file cannot be read (permission denied, I/O error).
        """
        path = self._expert_path(layer_idx, expert_id)
        if not path.exists():
            raise FileNotFoundError(f"Expert weight file not found: {path} (layer={layer_idx}, expert={expert_id})")
        try:
            data = path.read_bytes()
        except OSError as e:
            raise OSError(f"Failed to read expert weights from {path}: {e}") from e
        if self._ssd_latency_ms > 0:
            scale = len(data) / (2 * 1024 * 1024)
            time.sleep(self._ssd_latency_ms * scale / 1000)
        return data

    def _read_expert_partial(self, layer_idx: int, expert_id: int) -> bytes:
        """Dendritic: read only the first 1/3 of expert (gate projection).

        Args:
            layer_idx: Transformer layer index.
            expert_id: Expert index within the layer.

        Returns:
            Raw bytes of the first third of the expert weight file.

        Raises:
            FileNotFoundError: If the expert weight file does not exist.
            OSError: If the file cannot be read.
        """
        path = self._expert_path(layer_idx, expert_id)
        if not path.exists():
            raise FileNotFoundError(f"Expert weight file not found: {path} (layer={layer_idx}, expert={expert_id})")
        try:
            file_size = path.stat().st_size
        except OSError as e:
            raise OSError(f"Cannot stat expert weight file {path}: {e}") from e
        if file_size == 0:
            return b""
        partial_size = file_size // 3  # gate_proj is ~1/3 of total
        try:
            with open(path, "rb") as f:
                data = f.read(partial_size)
        except OSError as e:
            raise OSError(f"Failed to read partial expert weights from {path}: {e}") from e
        if self._ssd_latency_ms > 0:
            scale = len(data) / (2 * 1024 * 1024)
            time.sleep(self._ssd_latency_ms * scale / 1000)
        return data

    def _evict_until_free(self, needed_bytes: int):
        """Evict lowest-priority entries until needed_bytes are free."""
        while self._current_bytes + needed_bytes > self.capacity and self._cache:
            # Find lowest priority entry
            min_key = None
            min_priority = float("inf")
            for key, entry in self._cache.items():
                p = self._priority(entry)
                if p < min_priority:
                    min_priority = p
                    min_key = key

            if min_key is None:
                break

            evicted = self._cache.pop(min_key)
            self._current_bytes -= evicted.size_bytes

    def _insert(self, layer_idx: int, expert_id: int, data: bytes, is_partial: bool = False):
        """Insert into cache, evicting if necessary.

        Thread-safe: all cache mutations happen under self._lock.

        Args:
            layer_idx: Transformer layer index.
            expert_id: Expert index within the layer.
            data: Raw expert weight bytes to cache.
            is_partial: True if only gate projection was loaded (dendritic).
        """
        key = (layer_idx, expert_id)
        with self._lock:
            if key in self._cache:
                old = self._cache[key]
                self._current_bytes -= old.size_bytes

            self._evict_until_free(len(data))
            entry = LCPEntry(
                data=data,
                layer_idx=layer_idx,
                expert_id=expert_id,
                frequency=1,
                last_step=self._step,
                size_bytes=len(data),
                is_partial=is_partial,
            )
            self._cache[key] = entry
            self._current_bytes += len(data)

    def fetch(
        self,
        layer_idx: int,
        expert_ids: list[int],
        allow_skip: bool = True,
    ) -> list[tuple[bytes, str]]:
        """Fetch experts with priority: cache > prefetch > cold load > skip.

        Thread-safe for concurrent access from multiple inference threads.

        Args:
            layer_idx: Transformer layer index.
            expert_ids: List of expert indices to fetch.
            allow_skip: If True, allow skip-fallback when expert is unavailable.

        Returns:
            List of (data_bytes, source) where source is one of:
            'cache', 'prefetch', 'cold', 'skip', 'dendritic_skip'.
        """
        self.stats.total_requests += len(expert_ids)
        results = []

        for eid in expert_ids:
            key = (layer_idx, eid)

            # 1. Cache hit (fast path)
            with self._lock:
                if key in self._cache:
                    entry = self._cache[key]
                    entry.frequency += 1
                    entry.last_step = self._step
                    self.stats.cache_hits += 1
                    results.append((entry.data, "cache"))
                    continue

            # 2. Prefetch hit (background read completed)
            future = None
            with self._lock:
                if key in self._prefetch_pending:
                    future = self._prefetch_pending.pop(key)

            if future is not None:
                if future.done():
                    try:
                        data = future.result(timeout=0)
                        self._insert(layer_idx, eid, data)
                        self.stats.prefetch_hits += 1
                        results.append((data, "prefetch"))
                        continue
                    except Exception:
                        pass
                else:
                    # Prefetch still running -- wait briefly or skip
                    try:
                        data = future.result(timeout=0.001)  # 1ms max wait
                        self._insert(layer_idx, eid, data)
                        self.stats.prefetch_hits += 1
                        results.append((data, "prefetch"))
                        continue
                    except Exception:
                        pass

            # 3. Dendritic two-stage loading
            if self.enable_dendritic:
                try:
                    partial = self._read_expert_partial(layer_idx, eid)
                except (FileNotFoundError, OSError):
                    partial = b""
                if partial:
                    # Estimate gate magnitude from partial data
                    sample = partial[: min(1024, len(partial))]
                    gate_mag = np.frombuffer(sample, dtype=np.uint8).astype(np.float32).std()
                    if gate_mag < self.dendritic_threshold:
                        # Expert would contribute negligibly -- skip W2/W3
                        self._insert(layer_idx, eid, partial, is_partial=True)
                        self.stats.dendritic_skips += 1
                        results.append((partial, "dendritic_skip"))
                        continue

            # 4. Skip fallback (zero routing score, renormalize)
            if self.enable_skip and allow_skip:
                self.stats.skip_fallbacks += 1
                results.append((b"", "skip"))
                continue

            # 5. Cold load (synchronous -- last resort)
            try:
                data = self._read_expert(layer_idx, eid)
            except (FileNotFoundError, OSError):
                # Expert file missing or unreadable -- treat as skip
                self.stats.skip_fallbacks += 1
                results.append((b"", "skip"))
                continue
            self._insert(layer_idx, eid, data)
            self.stats.cold_loads += 1
            results.append((data, "cold"))

        # Update co-occurrence (once per fetch call, not per expert)
        self._update_cooccurrence(layer_idx, expert_ids)

        return results

    def prefetch(self, layer_idx: int, expert_ids: list[int]):
        """Async prefetch: kick off background reads for predicted experts.

        Thread-safe: checks and updates prefetch state under lock.

        Args:
            layer_idx: Layer index to prefetch experts for.
            expert_ids: Expert indices to prefetch.
        """
        for eid in expert_ids:
            key = (layer_idx, eid)
            with self._lock:
                if key in self._cache or key in self._prefetch_pending:
                    continue  # already cached or being fetched
                future = self._prefetch_pool.submit(self._read_expert, layer_idx, eid)
                self._prefetch_pending[key] = future

    def _update_cooccurrence(self, layer_idx: int, expert_ids: list[int]):
        """Update co-occurrence matrix for prediction."""
        if layer_idx > 0 and (layer_idx - 1) in self._prev_experts:
            prev = self._prev_experts[layer_idx - 1]
            for p in prev:
                for c in expert_ids:
                    self._cooccurrence[layer_idx - 1][p][c] += 1
        self._prev_experts[layer_idx] = expert_ids

    def predict_next(self, current_layer: int, current_experts: list[int], top_k: int = 4) -> list[int]:
        """Predict next layer's experts from co-occurrence statistics.

        Uses the co-occurrence matrix built from previous fetch() calls
        to predict which experts at ``current_layer + 1`` are most likely
        needed next.

        Args:
            current_layer: The layer currently being processed.
            current_experts: Expert IDs selected at the current layer.
            top_k: Maximum number of experts to predict.

        Returns:
            List of predicted expert IDs for the next layer, sorted by
            likelihood. Empty if no co-occurrence data is available.
        """
        if current_layer not in self._cooccurrence:
            return []

        scores: defaultdict[int, float] = defaultdict(float)
        for eid in current_experts:
            if eid in self._cooccurrence[current_layer]:
                for next_eid, count in self._cooccurrence[current_layer][eid].items():
                    scores[next_eid] += count

        if not scores:
            return []

        sorted_experts = sorted(scores.items(), key=lambda x: -x[1])
        return [eid for eid, _ in sorted_experts[:top_k]]

    def get_cache_summary(self) -> dict:
        """Get cache state summary including priority statistics.

        Thread-safe: reads cache state under lock.

        Returns:
            Dictionary with entries count, memory usage, capacity,
            utilization ratio, and priority statistics.
        """
        with self._lock:
            priorities = [self._priority(e) for e in self._cache.values()]
            num_entries = len(self._cache)
            current_bytes = self._current_bytes
        return {
            "entries": num_entries,
            "bytes_used_mb": current_bytes / 1e6,
            "capacity_mb": self.capacity / 1e6,
            "utilization": current_bytes / self.capacity if self.capacity > 0 else 0,
            "avg_priority": float(np.mean(priorities)) if priorities else 0.0,
            "min_priority": float(min(priorities)) if priorities else 0.0,
            "max_priority": float(max(priorities)) if priorities else 0.0,
        }

    def shutdown(self):
        """Shut down the prefetch thread pool and release resources."""
        self._prefetch_pool.shutdown(wait=False)
