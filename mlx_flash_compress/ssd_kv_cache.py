"""2-tier KV cache: hot in RAM, cold on SSD.

When the RAM cache exceeds ``max_ram_tokens`` the oldest entries are
evicted to SSD as safetensors files.  On a cache miss the cold entries
are loaded back and merged with the hot portion, avoiding recomputation.

Each inference session gets its own SSD subdirectory keyed by a hash of
the prompt prefix, so concurrent sessions never collide.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_flash_compress.kv_cache_backend import KVCacheBackend, PlainKVCache


def _session_dir(base: Path, prompt_prefix: str) -> Path:
    h = hashlib.sha256(prompt_prefix.encode()).hexdigest()[:16]
    return base / h


class SSDKVCache(KVCacheBackend):
    """Hot/cold KV cache with SSD spill.

    Parameters
    ----------
    num_layers:
        Number of transformer layers.
    num_heads:
        Number of KV attention heads.
    head_dim:
        Dimension per head.
    max_ram_tokens:
        Maximum tokens kept in the hot (RAM) tier per layer before
        eviction to SSD.
    cache_dir:
        Root directory for cold storage.  A session subdirectory is
        created underneath.
    prompt_prefix:
        Used to derive a session-unique subdirectory hash.
    inner_backend:
        Optional pre-built backend for the hot tier.  When *None* a
        :class:`PlainKVCache` is created.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        *,
        max_ram_tokens: int = 2048,
        cache_dir: str | Path | None = None,
        prompt_prefix: str = "",
        inner_backend: KVCacheBackend | None = None,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_ram_tokens = max_ram_tokens

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "mlx-flash" / "kv-cache"
        self._base_dir = Path(cache_dir)
        self._session_dir = _session_dir(self._base_dir, prompt_prefix)
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._hot: KVCacheBackend = inner_backend or PlainKVCache(num_layers, num_heads, head_dim)

        # Per-layer cold token count (only used for stats / merge decisions)
        self._cold_tokens: list[int] = [0] * num_layers

        self._lock = threading.Lock()

        self._ram_hits = 0
        self._ssd_hits = 0
        self._ssd_writes = 0
        self._ssd_reads = 0
        self._bytes_on_ssd = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ssd_path(self, layer_idx: int) -> Path:
        return self._session_dir / f"layer_{layer_idx:04d}.safetensors"

    def _evict_to_ssd(self, layer_idx: int, keys: mx.array, values: mx.array) -> None:
        """Append *keys*/*values* to the cold file for *layer_idx*."""
        path = self._ssd_path(layer_idx)
        with self._lock:
            if path.exists():
                existing = mx.load(str(path))
                old_k = existing["keys"]
                old_v = existing["values"]
                keys = mx.concatenate([old_k, keys], axis=1)
                values = mx.concatenate([old_v, values], axis=1)

            mx.save_safetensors(str(path), {"keys": keys, "values": values})
            self._cold_tokens[layer_idx] = keys.shape[1]
            self._ssd_writes += 1
            self._bytes_on_ssd = sum(
                p.stat().st_size for p in self._session_dir.iterdir() if p.suffix == ".safetensors"
            )

    def _load_cold(self, layer_idx: int) -> tuple[mx.array, mx.array] | None:
        path = self._ssd_path(layer_idx)
        if not path.exists():
            return None
        with self._lock:
            data = mx.load(str(path))
            self._ssd_reads += 1
            self._ssd_hits += 1
            return data["keys"], data["values"]

    # ------------------------------------------------------------------
    # KVCacheBackend interface
    # ------------------------------------------------------------------

    def update(self, layer_idx: int, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        all_k, all_v = self._hot.update(layer_idx, keys, values)

        seq_len = all_k.shape[1]
        if seq_len > self.max_ram_tokens:
            overflow = seq_len - self.max_ram_tokens
            evict_k = all_k[:, :overflow, :]
            evict_v = all_v[:, :overflow, :]
            self._evict_to_ssd(layer_idx, evict_k, evict_v)

            keep_k = all_k[:, overflow:, :]
            keep_v = all_v[:, overflow:, :]

            # Snapshot all layers before reset so we don't lose them
            saved: list[tuple[mx.array, mx.array] | None] = []
            for i in range(self.num_layers):
                if i == layer_idx:
                    saved.append((keep_k, keep_v))
                else:
                    ik, iv = self._hot.get_kv(i)
                    saved.append((ik, iv) if ik.shape[1] > 0 else None)

            self._hot.reset()
            for i, pair in enumerate(saved):
                if pair is not None:
                    self._hot.update(i, pair[0], pair[1])

            all_k, all_v = keep_k, keep_v

        return all_k, all_v

    def get_kv(self, layer_idx: int) -> tuple[mx.array, mx.array]:
        hot_k, hot_v = self._hot.get_kv(layer_idx)
        self._ram_hits += 1

        cold = self._load_cold(layer_idx)
        if cold is None:
            return hot_k, hot_v

        cold_k, cold_v = cold
        if hot_k.shape[1] == 0:
            return cold_k, cold_v
        return (
            mx.concatenate([cold_k, hot_k], axis=1),
            mx.concatenate([cold_v, hot_v], axis=1),
        )

    def reset(self) -> None:
        self._hot.reset()
        self._cold_tokens = [0] * self.num_layers
        if self._session_dir.exists():
            shutil.rmtree(self._session_dir, ignore_errors=True)
            self._session_dir.mkdir(parents=True, exist_ok=True)
        self._ram_hits = 0
        self._ssd_hits = 0
        self._ssd_writes = 0
        self._ssd_reads = 0
        self._bytes_on_ssd = 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "strategy": "ssd",
            "num_layers": self.num_layers,
            "max_ram_tokens": self.max_ram_tokens,
            "ram_hits": self._ram_hits,
            "ssd_hits": self._ssd_hits,
            "ssd_writes": self._ssd_writes,
            "ssd_reads": self._ssd_reads,
            "bytes_on_ssd": self._bytes_on_ssd,
            "cold_tokens_per_layer": list(self._cold_tokens),
        }
