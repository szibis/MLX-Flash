"""Real expert streaming: GPU lookup table + pre-stacked weight tensors.

Replaces MLX's QuantizedSwitchLinear with a cached version that:
1. Keeps only `capacity` experts per layer in GPU memory (pre-stacked tensors)
2. Uses a GPU lookup table to map global expert IDs to local cache slots
3. Updates cache between tokens (not during forward pass)
4. Zero-eval fallback: missing experts get score=0, remaining renormalized

This is the technique that makes 50GB+ models run on 36GB Macs.

Architecture:
  Forward pass (GPU only, no CPU/disk):
    router(x) -> expert_ids -> lookup[expert_ids] -> local_ids
    gather_qmm(x, stacked_weights, local_ids) -> output

  Between tokens (CPU + disk):
    Record which experts were activated
    Evict cold experts (LCP priority)
    Load new experts from safetensors via mmap
    Update stacked tensors + lookup table

Usage:
  from mlx_flash_compress.expert_streaming import enable_expert_streaming

  model, tokenizer = mlx_lm.load("model_name", lazy=True)
  streaming = enable_expert_streaming(model, capacity_per_layer=200)
  streaming.warmup()

  # Generate normally
  output = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=100)
  streaming.update()  # call between generation steps
"""

import json
import math
import mmap
import os
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import mlx.core as mx
import mlx.nn as nn


# -- Safetensors mmap reader --

@dataclass
class TensorInfo:
    dtype: str
    shape: list
    data_offset: int
    data_size: int


class SafetensorsMap:
    """Memory-mapped access to safetensors files for per-expert slicing."""

    def __init__(self, shard_paths: list):
        self._mmaps = {}
        self._tensor_map = {}
        self._data_offsets = {}
        for path in shard_paths:
            self._index_shard(path)

    def _index_shard(self, path):
        with open(path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)
            data_start = 8 + header_size

        header = json.loads(header_bytes)
        self._data_offsets[path] = data_start

        for key, info in header.items():
            if key == "__metadata__":
                continue
            offsets = info.get("data_offsets", [0, 0])
            self._tensor_map[key] = (path, TensorInfo(
                dtype=info.get("dtype", "F16"),
                shape=info.get("shape", []),
                data_offset=offsets[0],
                data_size=offsets[1] - offsets[0],
            ))

    def _get_mmap(self, path):
        if path not in self._mmaps:
            f = open(path, "rb")
            self._mmaps[path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmaps[path]

    _NP_DTYPES = {"F16": np.float16, "BF16": np.float16, "F32": np.float32,
                   "U32": np.uint32, "I32": np.int32, "U8": np.uint8}

    def get_expert_slice(self, key, expert_ids):
        """Load specific expert rows from stacked [E, ...] or per-expert format."""
        # Try stacked format first (key = "...switch_mlp.gate_proj.weight" shape [E, ...])
        if key in self._tensor_map:
            path, info = self._tensor_map[key]
            if len(info.shape) >= 3:
                # Stacked format: slice rows directly
                num_experts = info.shape[0]
                row_size = info.data_size // num_experts
                mm = self._get_mmap(path)
                base = self._data_offsets[path] + info.data_offset
                np_dtype = self._NP_DTYPES.get(info.dtype, np.float16)
                row_shape = info.shape[1:]

                slices = []
                for eid in expert_ids:
                    offset = base + eid * row_size
                    raw = mm[offset:offset + row_size]
                    arr = np.frombuffer(raw, dtype=np_dtype).reshape(row_shape)
                    slices.append(arr)
                return mx.array(np.stack(slices))

        # Per-expert format: "model.layers.N.mlp.experts.E.proj.attr"
        # Derive the per-expert key pattern from the stacked key
        # e.g. "model.layers.0.mlp.switch_mlp.gate_proj.weight"
        #    -> "model.layers.0.mlp.experts.{E}.gate_proj.weight"
        per_expert_key = key.replace(".switch_mlp.", ".experts.0.")
        if per_expert_key in self._tensor_map:
            slices = []
            for eid in expert_ids:
                eid_key = key.replace(".switch_mlp.", f".experts.{eid}.")
                if eid_key not in self._tensor_map:
                    continue
                path, info = self._tensor_map[eid_key]
                mm = self._get_mmap(path)
                offset = self._data_offsets[path] + info.data_offset
                np_dtype = self._NP_DTYPES.get(info.dtype, np.float16)
                raw = mm[offset:offset + info.data_size]
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(info.shape)
                slices.append(arr)
            if slices:
                return mx.array(np.stack(slices))

        return None

    def has_key(self, key):
        return key in self._tensor_map

    def close(self):
        for mm in self._mmaps.values():
            mm.close()
        self._mmaps.clear()


# -- LCP eviction tracker --

class LCPTracker:
    def __init__(self, num_experts, lcp_base=0.25, lcp_decay=128.0):
        self.num_experts = num_experts
        self.lcp_base = lcp_base
        self.lcp_decay = lcp_decay
        self.frequency = np.zeros(num_experts, dtype=np.int64)
        self.last_used = np.zeros(num_experts, dtype=np.int64)
        self.step = 0

    def record(self, expert_ids):
        self.step += 1
        for eid in expert_ids:
            if 0 <= eid < self.num_experts:
                self.frequency[eid] += 1
                self.last_used[eid] = self.step

    def priority(self, eid):
        age = self.step - self.last_used[eid]
        return self.frequency[eid] * (self.lcp_base ** (age / self.lcp_decay))

    def coldest(self, among, n):
        priorities = [(eid, self.priority(eid)) for eid in among]
        priorities.sort(key=lambda x: x[1])
        return [eid for eid, _ in priorities[:n]]


# -- Cached QuantizedSwitchLinear replacement --

class CachedSwitchLinear(nn.Module):
    """Drop-in replacement using GPU lookup table + pre-stacked weights."""

    def __init__(self, cache, proj_name, group_size, bits, mode=0):
        super().__init__()
        self._cache = cache
        self._proj_name = proj_name
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    def __call__(self, x, indices, sorted_indices=False):
        if self._proj_name == "gate_proj":
            self._cache._indices_buffer.append(indices)

        local_indices = self._cache.lookup[indices]

        return mx.gather_qmm(
            x,
            self._cache.weights[self._proj_name],
            self._cache.scales[self._proj_name],
            self._cache.biases.get(self._proj_name),
            rhs_indices=local_indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )


# -- Expert cache per layer --

class ExpertCache:
    """Pre-stacked expert weights in GPU with LCP eviction."""

    def __init__(self, layer_idx, num_experts, capacity, st_map, weight_keys):
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.capacity = min(capacity, num_experts)
        self.st_map = st_map
        self.weight_keys = weight_keys
        self.cached_ids = []
        self.lookup = mx.zeros(num_experts, dtype=mx.int32)
        self.hit_mask = mx.zeros(num_experts)
        self.weights = {}
        self.scales = {}
        self.biases = {}
        self.tracker = LCPTracker(num_experts)
        self._indices_buffer = []
        self.total_tokens = 0
        self.cache_updates = 0

    def initial_fill(self, expert_ids=None):
        if expert_ids is None:
            expert_ids = list(range(self.capacity))
        self.cached_ids = expert_ids[:self.capacity]

        for full_key_name, st_key in self.weight_keys.items():
            data = self.st_map.get_expert_slice(st_key, self.cached_ids)
            if data is None:
                continue
            parts = full_key_name.split(".")
            proj = parts[0]  # gate_proj, up_proj, down_proj
            attr = parts[1]  # weight, scales, biases
            target = self.weights if attr == "weight" else self.scales if attr == "scales" else self.biases
            target[proj] = data

        self._rebuild_lookup()

    def _rebuild_lookup(self):
        lookup_np = np.zeros(self.num_experts, dtype=np.int32)
        hit_np = np.zeros(self.num_experts, dtype=np.float32)
        for slot, eid in enumerate(self.cached_ids):
            lookup_np[eid] = slot
            hit_np[eid] = 1.0
        self.lookup = mx.array(lookup_np)
        self.hit_mask = mx.array(hit_np)

    def update_between_tokens(self):
        if not self._indices_buffer:
            return
        activated = set()
        for indices in self._indices_buffer:
            ids = np.array(indices).flatten().tolist()
            activated.update(ids)
            self.tracker.record(ids)
        self._indices_buffer.clear()
        self.total_tokens += 1

        cached_set = set(self.cached_ids)
        misses = [eid for eid in activated if eid not in cached_set]
        if not misses:
            return

        evict_candidates = [eid for eid in self.cached_ids if eid not in activated]
        n_to_evict = min(len(misses), len(evict_candidates))
        to_evict = self.tracker.coldest(evict_candidates, n_to_evict)
        if not to_evict:
            return

        new_ids = misses[:len(to_evict)]
        evict_set = set(to_evict)
        new_cached = [eid for eid in self.cached_ids if eid not in evict_set]
        new_cached.extend(new_ids)
        self.cached_ids = new_cached[:self.capacity]

        # Reload changed expert slots from disk
        for full_key_name, st_key in self.weight_keys.items():
            data = self.st_map.get_expert_slice(st_key, new_ids)
            if data is None:
                continue
            parts = full_key_name.split(".")
            proj = parts[0]
            attr = parts[1]
            target = self.weights if attr == "weight" else self.scales if attr == "scales" else self.biases
            if proj in target:
                # Find slot indices for new experts
                slot_map = {eid: i for i, eid in enumerate(self.cached_ids)}
                slots = [slot_map[eid] for eid in new_ids if eid in slot_map]
                if slots:
                    slot_idx = mx.array(slots[:data.shape[0]])
                    target[proj][slot_idx] = data

        self._rebuild_lookup()
        self.cache_updates += 1

    def stats(self):
        return {
            "layer": self.layer_idx,
            "cached": len(self.cached_ids),
            "capacity": self.capacity,
            "coverage": len(self.cached_ids) / max(self.num_experts, 1),
            "tokens": self.total_tokens,
            "updates": self.cache_updates,
        }


# -- Streaming state --

@dataclass
class StreamingState:
    caches: list = field(default_factory=list)
    st_map: object = None
    model: object = None

    def warmup(self):
        for cache in self.caches:
            cache.initial_fill()
        arrays = []
        for cache in self.caches:
            arrays.extend(cache.weights.values())
            arrays.extend(cache.scales.values())
            arrays.extend(cache.biases.values())
        if arrays:
            mx.eval(*arrays)

    def update(self):
        for cache in self.caches:
            cache.update_between_tokens()

    def stats(self):
        return [c.stats() for c in self.caches]

    def total_cached(self):
        return sum(len(c.cached_ids) for c in self.caches)

    def avg_coverage(self):
        if not self.caches:
            return 0
        return sum(c.stats()["coverage"] for c in self.caches) / len(self.caches)

    def cleanup(self):
        if self.st_map:
            self.st_map.close()


# -- Skip-fallback: zero missing expert scores --

def enable_skip_fallback(model, caches: list):
    """Monkey-patch MoE blocks to zero scores for uncached experts.

    When an expert is not in the cache, its routing score is set to 0
    and remaining scores are renormalized. This avoids computing with
    stale/placeholder weights for uncached experts.
    """
    import types

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        return

    cache_by_layer = {c.layer_idx: c for c in caches}

    for layer_idx, layer in enumerate(layers):
        if layer_idx not in cache_by_layer:
            continue
        if not hasattr(layer, "mlp"):
            continue

        cache = cache_by_layer[layer_idx]
        mlp = layer.mlp
        original_call = type(mlp).__call__

        def make_patched(orig, cache_ref):
            def patched_call(self, x):
                gates = self.gate(x)
                gates = mx.softmax(gates, axis=-1, precise=True)

                k = self.top_k
                inds = mx.stop_gradient(
                    mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
                )
                scores = mx.take_along_axis(gates, inds, axis=-1)

                # Zero out scores for uncached experts
                mask = cache_ref.hit_mask[inds]  # 1.0 cached, 0.0 not
                scores = scores * mask
                score_sum = scores.sum(axis=-1, keepdims=True)
                scores = mx.where(score_sum > 0, scores / score_sum, scores)

                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2)

                if hasattr(self, "shared_expert"):
                    shared = self.shared_expert(x)
                    if hasattr(self, "shared_expert_gate"):
                        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
                    y = y + shared

                return y
            return patched_call

        mlp.__call__ = types.MethodType(make_patched(original_call, cache), mlp)


# -- Profile-based warmup --

def get_warmup_experts(task: str = "general", num_layers: int = 24,
                       num_experts: int = 60, top_n: int = 30) -> list:
    """Get hot experts for a task from pre-computed profiles.

    Uses task_profiler's predefined profiles to determine which experts
    to pre-load for a given workload type.
    """
    try:
        from mlx_flash_compress.task_profiler import get_predefined_profile
        profile = get_predefined_profile(task, num_layers=num_layers, num_experts=num_experts)
        hot_experts = profile.get_hot_experts(top_pct=top_n / num_experts)

        # Convert to flat list of expert IDs per layer
        result = []
        for layer_idx in range(num_layers):
            layer_key = str(layer_idx)
            if layer_key in hot_experts:
                result.append(hot_experts[layer_key][:top_n])
            else:
                result.append(list(range(top_n)))
        return result
    except Exception:
        # Fallback: first N experts
        return [list(range(top_n)) for _ in range(num_layers)]


# -- Public API --

def enable_expert_streaming(model, capacity_per_layer=200, model_path=None):
    """Enable expert streaming on a loaded MLX MoE model.

    Replaces QuantizedSwitchLinear with cached versions using
    pre-stacked GPU tensors and lookup tables.
    """
    state = StreamingState(model=model)

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    if layers is None:
        raise ValueError("Cannot find model layers")

    # Find safetensors
    if model_path:
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            # Try HuggingFace cache
            import glob
            cache_pattern = os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{model_path.replace('/', '--')}*/snapshots/*/")
            cache_dirs = sorted(glob.glob(cache_pattern))
            if cache_dirs:
                model_dir = Path(cache_dirs[-1])

        shards = sorted(model_dir.glob("*.safetensors"))
        if shards:
            state.st_map = SafetensorsMap([str(s) for s in shards])

    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue
        mlp = layer.mlp

        # Detect switch_mlp (Qwen/DeepSeek) or block_sparse_moe (Mixtral)
        switch = getattr(mlp, "switch_mlp", None)
        if switch is None:
            switch = getattr(mlp, "block_sparse_moe", None)
            if switch is not None:
                switch = getattr(switch, "switch_mlp", None)
        if switch is None:
            continue

        num_experts = 0
        proj_modules = {}
        weight_keys = {}

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(switch, proj_name, None)
            if proj is None:
                continue
            w = getattr(proj, "weight", None)
            if w is not None and len(w.shape) == 3:
                num_experts = w.shape[0]
                proj_modules[proj_name] = proj
                for suffix in ["weight", "scales", "biases"]:
                    wk = f"model.layers.{layer_idx}.mlp.switch_mlp.{proj_name}.{suffix}"
                    weight_keys[f"{proj_name}.{suffix}"] = wk

        if num_experts == 0:
            continue

        cache = ExpertCache(
            layer_idx=layer_idx,
            num_experts=num_experts,
            capacity=min(capacity_per_layer, num_experts),
            st_map=state.st_map,
            weight_keys=weight_keys,
        )
        state.caches.append(cache)

        for proj_name, proj in proj_modules.items():
            replacement = CachedSwitchLinear(
                cache=cache,
                proj_name=proj_name,
                group_size=getattr(proj, "group_size", 64),
                bits=getattr(proj, "bits", 4),
                mode=getattr(proj, "mode", 0),
            )
            setattr(switch, proj_name, replacement)

    return state
