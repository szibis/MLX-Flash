"""Shared expert pinning: keep always-on experts in the hot cache tier.

DeepSeek-V2/V3 and Qwen3-MoE models have "shared experts" that are
activated for every token regardless of routing. These experts carry
general-purpose knowledge and should never be evicted from cache.

Currently mlx-flash's ExpertCacheManager treats all experts uniformly,
which means shared experts can be evicted during cache pressure. This
module detects shared experts (from model config or runtime observation)
and pins them in the hot cache tier so they are never evicted.

Detection strategies:
  1. Config-based: Check model config for ``num_shared_experts``,
     ``shared_expert_gate``, ``n_shared_experts`` fields.
  2. Observation-based: If an expert is activated >95% of the time,
     classify it as shared.

Integration: SharedExpertPinner provides ``should_evict()`` that
returns False for shared experts, which can be checked by the
LCPTracker or LeastStalePolicy before eviction.

Usage:
    from mlx_flash_compress.shared_expert_pinning import detect_and_pin_shared_experts

    pinner = detect_and_pin_shared_experts(model)
    if not pinner.should_evict(layer_idx=5, expert_id=0):
        print("Expert 0 at layer 5 is pinned — do not evict")
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


# Known config keys for shared experts across model architectures
_SHARED_EXPERT_CONFIG_KEYS = [
    "num_shared_experts",       # DeepSeek-V2/V3
    "n_shared_experts",         # Some DeepSeek variants
    "shared_expert_num",        # Alternative naming
    "num_shared_expert",        # Singular variant
]

# Keys that indicate shared expert gate presence (DeepSeek pattern:
# shared experts have a gating mechanism too)
_SHARED_GATE_CONFIG_KEYS = [
    "shared_expert_gate",
    "has_shared_expert_gate",
]

# Keys that hold explicit shared expert ID lists
_SHARED_ID_CONFIG_KEYS = [
    "shared_expert_ids",
    "shared_expert_indices",
]


@dataclass
class SharedExpertInfo:
    """Metadata about detected shared experts."""
    layer_idx: int
    expert_ids: list[int]
    detection_method: str  # "config" or "observation"
    activation_rate: float = 1.0  # fraction of tokens this expert is activated


class SharedExpertDetector:
    """Detect which experts are shared (always-on) from model config or runtime observation.

    Shared experts are identified by:
    1. Model config fields (``num_shared_experts``, etc.) — these appear in
       DeepSeek-V2/V3 and Qwen3-MoE architectures.
    2. Runtime observation: if an expert is activated for >95% of tokens,
       it behaves as a shared expert.
    """

    def __init__(self, model=None, config: Optional[dict] = None):
        self._model = model
        self._config = config
        self._shared_experts: dict[int, list[int]] = {}
        self._detection_method: str = "none"

    def detect_from_config(self, model_config: dict) -> dict[int, list[int]]:
        """Check model config for shared expert declarations.

        Looks for ``num_shared_experts``, ``shared_expert_ids``, and
        similar fields that DeepSeek-V2/V3 and Qwen3-MoE models expose.

        Args:
            model_config: Model configuration dictionary (e.g., from
                ``config.json`` or ``model.config``).

        Returns:
            ``{layer_idx: [shared_expert_ids]}``. If the config declares
            shared experts globally (not per-layer), the same IDs are
            applied to all MoE layers.
        """
        result: dict[int, list[int]] = {}

        # 1. Check for explicit shared expert IDs
        explicit_ids = None
        for key in _SHARED_ID_CONFIG_KEYS:
            if key in model_config:
                val = model_config[key]
                if isinstance(val, list):
                    explicit_ids = val
                    break

        # 2. Check for num_shared_experts count
        num_shared = 0
        for key in _SHARED_EXPERT_CONFIG_KEYS:
            if key in model_config:
                val = model_config[key]
                if isinstance(val, int) and val > 0:
                    num_shared = val
                    break

        # 3. Determine shared expert IDs
        if explicit_ids is not None:
            shared_ids = explicit_ids
        elif num_shared > 0:
            # Convention: shared experts are the first N expert IDs.
            # DeepSeek-V2 uses this convention. The shared experts
            # are separate modules (not part of the routed expert pool),
            # but they occupy the first N slots in the expert index space.
            shared_ids = list(range(num_shared))
        else:
            return result

        # 4. Apply to all MoE layers
        num_layers = model_config.get(
            "num_hidden_layers",
            model_config.get("num_layers", model_config.get("n_layers", 0)),
        )

        # Check if model specifies which layers are MoE
        # DeepSeek-V2 has first_k_dense_replace (first K layers are dense, rest MoE)
        first_moe_layer = model_config.get("first_k_dense_replace", 0)
        moe_layer_freq = model_config.get("moe_layer_freq", 1)

        if num_layers > 0:
            for layer_idx in range(num_layers):
                if layer_idx < first_moe_layer:
                    continue
                if moe_layer_freq > 1 and (layer_idx - first_moe_layer) % moe_layer_freq != 0:
                    continue
                result[layer_idx] = list(shared_ids)
        else:
            # No layer count in config; store as layer -1 (global)
            result[-1] = list(shared_ids)

        self._shared_experts = result
        self._detection_method = "config"
        return result

    def detect_from_observation(
        self,
        routing_log: list,
        threshold: float = 0.95,
    ) -> dict[int, list[int]]:
        """Detect shared experts from runtime routing observations.

        An expert activated for more than ``threshold`` fraction of all
        tokens is classified as a shared expert. This handles models
        that don't declare shared experts in config but have experts
        that behave as shared in practice.

        Args:
            routing_log: List of routing events. Each event should have
                ``layer_idx``, ``expert_ids`` attributes (matching the
                ``RoutingEvent`` dataclass from ``router_hook.py``).
            threshold: Activation frequency threshold. An expert activated
                for >95% of tokens is considered shared.

        Returns:
            ``{layer_idx: [shared_expert_ids]}``.
        """
        if not routing_log:
            return {}

        # Count activations per (layer, expert)
        activation_counts: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        token_counts_per_layer: dict[int, int] = defaultdict(int)

        # Group by token to count unique tokens per layer
        tokens_per_layer: dict[int, set] = defaultdict(set)

        for event in routing_log:
            layer_idx = event.layer_idx
            token_idx = event.token_idx
            tokens_per_layer[layer_idx].add(token_idx)
            for eid in event.expert_ids:
                activation_counts[layer_idx][eid] += 1

        # Identify shared experts: activation_rate > threshold
        result: dict[int, list[int]] = {}
        for layer_idx, expert_counts in activation_counts.items():
            total_tokens = len(tokens_per_layer[layer_idx])
            if total_tokens == 0:
                continue

            shared_ids = []
            for eid, count in sorted(expert_counts.items()):
                rate = count / total_tokens
                if rate >= threshold:
                    shared_ids.append(eid)

            if shared_ids:
                result[layer_idx] = shared_ids

        self._shared_experts = result
        self._detection_method = "observation"
        return result

    def get_shared_experts(self) -> dict[int, list[int]]:
        """Return detected shared experts.

        Returns:
            ``{layer_idx: [shared_expert_ids]}``. Empty if no detection
            has been run yet.
        """
        return dict(self._shared_experts)


class SharedExpertPinner:
    """Pin shared experts in the hot cache tier to prevent eviction.

    Integrates with ExpertCacheManager / LCPTracker / LeastStalePolicy
    by providing ``should_evict()`` and ``is_pinned()`` checks. The
    eviction policy should call ``should_evict()`` before evicting
    any expert — if it returns False, the expert must not be evicted.
    """

    def __init__(self, shared_experts: dict[int, list[int]]):
        """Initialize with detected shared experts.

        Args:
            shared_experts: ``{layer_idx: [shared_expert_ids]}``.
                Use ``SharedExpertDetector`` to produce this mapping.
        """
        self._shared_experts = dict(shared_experts)

        # Build fast lookup set: {(layer_idx, expert_id)}
        self._pinned_set: set[tuple[int, int]] = set()
        for layer_idx, expert_ids in self._shared_experts.items():
            for eid in expert_ids:
                self._pinned_set.add((layer_idx, eid))

        # Stats tracking
        self._eviction_blocks: int = 0
        self._eviction_checks: int = 0

    def is_pinned(self, layer_idx: int, expert_id: int) -> bool:
        """Check if an expert is pinned (shared).

        Args:
            layer_idx: The layer index.
            expert_id: The expert ID within that layer.

        Returns:
            True if the expert is pinned and should not be evicted.
        """
        # Check layer-specific pinning
        if (layer_idx, expert_id) in self._pinned_set:
            return True
        # Check global pinning (layer -1 means all layers)
        if (-1, expert_id) in self._pinned_set:
            return True
        return False

    def should_evict(self, layer_idx: int, expert_id: int) -> bool:
        """Check whether an expert is allowed to be evicted.

        This is the primary integration point with eviction policies.
        Returns False for shared (pinned) experts.

        Args:
            layer_idx: The layer index.
            expert_id: The expert ID within that layer.

        Returns:
            True if the expert can be evicted, False if it is pinned.
        """
        self._eviction_checks += 1
        if self.is_pinned(layer_idx, expert_id):
            self._eviction_blocks += 1
            return False
        return True

    def filter_eviction_candidates(
        self,
        candidates: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Filter a list of eviction candidates, removing pinned experts.

        Convenience method for batch eviction: given a list of
        ``(layer_idx, expert_id)`` candidates, return only those
        that are eligible for eviction.

        Args:
            candidates: List of ``(layer_idx, expert_id)`` tuples.

        Returns:
            Filtered list with pinned experts removed.
        """
        return [
            (layer_idx, expert_id)
            for layer_idx, expert_id in candidates
            if self.should_evict(layer_idx, expert_id)
        ]

    def get_pinned_count(self) -> int:
        """Return the total number of pinned expert slots.

        Returns:
            Count of unique (layer, expert) pairs that are pinned.
        """
        return len(self._pinned_set)

    def get_pinned_experts(self) -> dict[int, list[int]]:
        """Return the pinned expert mapping.

        Returns:
            ``{layer_idx: [expert_ids]}`` for all pinned experts.
        """
        return dict(self._shared_experts)

    def get_stats(self) -> dict:
        """Return pinning statistics.

        Returns:
            Dictionary with pinning metrics including pinned count,
            eviction checks, and block rate.
        """
        block_rate = (
            self._eviction_blocks / self._eviction_checks
            if self._eviction_checks > 0
            else 0.0
        )
        return {
            "pinned_count": self.get_pinned_count(),
            "pinned_layers": len(self._shared_experts),
            "eviction_checks": self._eviction_checks,
            "eviction_blocks": self._eviction_blocks,
            "block_rate": round(block_rate, 4),
            "pinned_experts": {
                str(layer_idx): expert_ids
                for layer_idx, expert_ids in self._shared_experts.items()
            },
        }


def detect_and_pin_shared_experts(
    model,
    cache_manager=None,
) -> SharedExpertPinner:
    """One-line convenience: detect shared experts and configure pinning.

    Attempts config-based detection first (fast, deterministic). If the
    model config doesn't declare shared experts, falls back to a no-op
    pinner (no experts pinned).

    For observation-based detection, use ``SharedExpertDetector`` directly
    with a routing log from ``RouterHook``.

    Args:
        model: An MLX MoE model. Should have ``model.config`` or
            ``model.args`` with MoE configuration fields.
        cache_manager: Optional cache manager to integrate with.
            Currently unused; reserved for future integration with
            ``ExpertCacheManager``.

    Returns:
        A ``SharedExpertPinner`` instance configured for the model's
        shared experts.
    """
    detector = SharedExpertDetector(model=model)

    # Try to extract model config
    model_config = _extract_model_config(model)

    if model_config:
        shared = detector.detect_from_config(model_config)
        if shared:
            return SharedExpertPinner(shared)

    # No shared experts found in config — return empty pinner
    return SharedExpertPinner({})


def _extract_model_config(model) -> Optional[dict]:
    """Extract model configuration dictionary from various model formats.

    Handles multiple model wrapper patterns:
    - ``model.config`` (dict or object with ``__dict__``)
    - ``model.args`` (mlx_lm pattern)
    - ``model.model.config``
    """
    config_sources = []

    # Try model.config
    if hasattr(model, "config"):
        config_sources.append(model.config)
    # Try model.args (mlx_lm pattern)
    if hasattr(model, "args"):
        config_sources.append(model.args)
    # Try model.model.config
    if hasattr(model, "model") and hasattr(model.model, "config"):
        config_sources.append(model.model.config)
    # Try model.model.args
    if hasattr(model, "model") and hasattr(model.model, "args"):
        config_sources.append(model.model.args)

    def _to_dict(source):
        """Convert a config source to a dict, handling class-level attrs."""
        if isinstance(source, dict):
            return source
        if hasattr(source, "__dict__"):
            # Merge class-level attributes with instance attributes
            # so that class Cfg: num_shared_experts = 3 works
            d = {}
            for cls in reversed(type(source).__mro__):
                for k, v in vars(cls).items():
                    if not k.startswith("_") and not callable(v):
                        d[k] = v
            d.update(source.__dict__)
            return d
        return None

    for source in config_sources:
        d = _to_dict(source)
        if d is None:
            continue
        for key in _SHARED_EXPERT_CONFIG_KEYS + _SHARED_ID_CONFIG_KEYS:
            if key in d:
                return d

    # Return whatever we found, even if it doesn't have shared expert keys
    for source in config_sources:
        d = _to_dict(source)
        if d is not None:
            return d

    return None
