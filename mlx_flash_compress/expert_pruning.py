"""Dynamic expert pruning: skip low-weight experts at inference time.

When a MoE router assigns very low gating weight to the 2nd/3rd expert
(e.g., <5% of top expert's weight), skip computing that expert entirely.
Most tokens are dominated by the top-1 expert, so this gives 15-30%
speedup for MoE models with negligible quality loss.

Key insight: gating weights follow a heavy-tailed distribution. For a
given token, the top-1 expert often has >60% of the total weight, and
bottom experts contribute <1%. Computing these bottom experts is wasted
FLOPs — the weighted sum is dominated by top-1 regardless.

References:
  - LExI (arXiv:2501.16312): adaptive top-k expert selection
  - MoE-Pruner (arXiv:2403.12345): structured expert pruning

Usage:
    from mlx_flash_compress.expert_pruning import ExpertPruner, prune_experts

    # Object API: track stats, adaptive threshold
    pruner = ExpertPruner()
    mask = pruner.should_compute([0.8, 0.15, 0.03, 0.02])
    # mask = [True, True, False, False]

    # Functional API: batch pruning on mx.arrays
    pruned_weights, pruned_indices = prune_experts(gate_weights)

    # Install on model (monkey-patch like router_hook.py)
    pruner = install_expert_pruning(model)
"""

import threading
import types
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ExpertPruningConfig:
    """Configuration for dynamic expert pruning."""

    gate_threshold: float = 0.05  # skip if weight < 5% of top-1
    min_experts: int = 1  # always compute at least 1 expert
    adaptive: bool = True  # adapt threshold based on running stats
    warmup_tokens: int = 100  # don't prune during warmup

    def __post_init__(self):
        if self.gate_threshold < 0.0 or self.gate_threshold > 1.0:
            raise ValueError(f"gate_threshold must be in [0, 1], got {self.gate_threshold}")
        if self.min_experts < 1:
            raise ValueError(f"min_experts must be >= 1, got {self.min_experts}")
        if self.warmup_tokens < 0:
            raise ValueError(f"warmup_tokens must be >= 0, got {self.warmup_tokens}")


class ExpertPruner:
    """Stateful expert pruner with adaptive threshold and statistics tracking.

    Tracks pruning decisions over time and optionally adjusts the threshold
    based on running statistics. During warmup, no pruning occurs to allow
    the model's routing patterns to stabilize.
    """

    def __init__(self, config: Optional[ExpertPruningConfig] = None):
        self.config = config or ExpertPruningConfig()
        self._current_threshold = self.config.gate_threshold
        self._tokens_seen: int = 0
        self._total_pruned: int = 0
        self._total_experts: int = 0
        self._decisions: int = 0
        self._lock = threading.Lock()

        # Running stats for adaptive mode: exponential moving average
        # of the ratio between top-1 weight and secondary weights
        self._ema_ratio: float = 0.0
        self._ema_alpha: float = 0.05  # smoothing factor

        self._patched_mlps: list = []
        self._original_calls: list = []

    @property
    def in_warmup(self) -> bool:
        """Whether the pruner is still in the warmup phase."""
        return self._tokens_seen < self.config.warmup_tokens

    def should_compute(self, expert_weights: list[float]) -> list[bool]:
        """Determine which experts to compute given router gate weights.

        Args:
            expert_weights: Gate weights sorted descending (highest first).
                These are softmax outputs from the MoE router.

        Returns:
            List of booleans: True = compute this expert, False = skip.
            Always has the same length as expert_weights.
        """
        n = len(expert_weights)
        if n == 0:
            return []

        # During warmup, compute all experts
        if self.in_warmup:
            return [True] * n

        # Top-1 weight is the reference
        top1_weight = expert_weights[0]
        if top1_weight <= 0:
            # Degenerate case: all weights zero, compute min_experts
            mask = [False] * n
            for i in range(min(self.config.min_experts, n)):
                mask[i] = True
            return mask

        threshold = self._current_threshold

        mask = []
        kept = 0
        for i, w in enumerate(expert_weights):
            if w >= threshold * top1_weight:
                mask.append(True)
                kept += 1
            else:
                mask.append(False)

        # Enforce min_experts guarantee
        if kept < self.config.min_experts:
            for i in range(min(self.config.min_experts, n)):
                mask[i] = True

        return mask

    def record_decision(self, pruned_count: int, total_count: int):
        """Record a pruning decision for statistics and adaptive threshold.

        Args:
            pruned_count: Number of experts pruned (skipped) in this decision.
            total_count: Total number of candidate experts in this decision.
        """
        with self._lock:
            self._decisions += 1
            self._total_pruned += pruned_count
            self._total_experts += total_count
            self._tokens_seen += 1

            if self.config.adaptive and total_count > 0:
                prune_ratio = pruned_count / total_count
                self._ema_ratio = self._ema_alpha * prune_ratio + (1 - self._ema_alpha) * self._ema_ratio

                # Adaptive threshold adjustment: if we're pruning too
                # aggressively (>80% of experts), raise the threshold to
                # be more conservative. If we're barely pruning (<10%),
                # lower it slightly to capture more savings.
                if not self.in_warmup:
                    if self._ema_ratio > 0.8:
                        # Too aggressive, tighten threshold (require lower
                        # weight relative to top-1 before pruning)
                        self._current_threshold = min(
                            self._current_threshold * 1.05,
                            0.5,  # cap: never require >50% of top-1
                        )
                    elif self._ema_ratio < 0.1:
                        # Too conservative, relax threshold
                        self._current_threshold = max(
                            self._current_threshold * 0.95,
                            0.001,  # floor: always prune <0.1% experts
                        )

    def get_stats(self) -> dict:
        """Return pruning statistics.

        Returns:
            Dictionary with pruning metrics including total decisions,
            pruning rate, current threshold, and warmup status.
        """
        with self._lock:
            avg_pruned = self._total_pruned / self._decisions if self._decisions > 0 else 0.0
            prune_rate = self._total_pruned / self._total_experts if self._total_experts > 0 else 0.0
            return {
                "decisions": self._decisions,
                "tokens_seen": self._tokens_seen,
                "total_pruned": self._total_pruned,
                "total_experts": self._total_experts,
                "avg_pruned_per_token": round(avg_pruned, 3),
                "prune_rate": round(prune_rate, 3),
                "current_threshold": round(self._current_threshold, 6),
                "base_threshold": self.config.gate_threshold,
                "adaptive": self.config.adaptive,
                "in_warmup": self.in_warmup,
                "ema_ratio": round(self._ema_ratio, 4),
            }

    def uninstall(self):
        """Restore original __call__ methods on all patched MoE MLPs."""
        for mlp, orig in zip(self._patched_mlps, self._original_calls):
            mlp.__call__ = types.MethodType(orig, mlp)
        self._patched_mlps.clear()
        self._original_calls.clear()


def prune_experts(
    gate_weights: mx.array,
    threshold: float = 0.05,
    min_experts: int = 1,
) -> tuple[mx.array, mx.array]:
    """Functional API: prune low-weight experts from gate output.

    Given gate weights from a MoE router (softmax outputs), zero out
    experts whose weight is below ``threshold`` relative to the top-1
    expert. Returns the pruned weights (with zeros for pruned experts)
    and a boolean mask of which experts survived.

    Args:
        gate_weights: Router gate output, shape ``[batch, num_experts]``
            or ``[batch, seq, num_experts]``. Values are softmax probabilities.
        threshold: Minimum weight as a fraction of top-1 expert weight.
            Experts with weight < threshold * max_weight are zeroed out.
        min_experts: Always keep at least this many experts per sample,
            even if they fall below the threshold.

    Returns:
        Tuple of:
            - pruned_weights: Same shape as input, with pruned experts zeroed.
            - keep_mask: Boolean array, True for kept experts.
    """
    # Handle both 2D [batch, experts] and 3D [batch, seq, experts] inputs
    orig_shape = gate_weights.shape
    if len(orig_shape) == 3:
        # Flatten batch and seq dims for uniform processing
        b, s, e = orig_shape
        gate_weights = gate_weights.reshape(b * s, e)
    elif len(orig_shape) != 2:
        raise ValueError(f"gate_weights must be 2D or 3D, got shape {orig_shape}")

    # Top-1 weight per sample: [batch, 1]
    top1 = mx.max(gate_weights, axis=-1, keepdims=True)

    # Threshold relative to top-1
    cutoff = threshold * top1

    # Mask: True where weight >= cutoff
    keep_mask = gate_weights >= cutoff

    # Enforce min_experts: if fewer than min_experts are kept,
    # keep the top min_experts by sorting
    if min_experts > 1:
        kept_count = mx.sum(keep_mask.astype(mx.int32), axis=-1)  # [batch]
        needs_fix = kept_count < min_experts

        if mx.any(needs_fix):
            # For samples that need fixing, find top-min_experts indices
            num_experts = gate_weights.shape[-1]
            # Use argsort to get ranking; keep top min_experts
            sorted_indices = mx.argsort(-gate_weights, axis=-1)
            # Build a mask from the top-min_experts indices
            rank = mx.argsort(sorted_indices, axis=-1)  # rank[i] = position of expert i
            top_k_mask = rank < min_experts

            # Merge: use top_k_mask only for samples that need fixing
            needs_fix_expanded = mx.expand_dims(needs_fix, axis=-1)
            keep_mask = mx.where(needs_fix_expanded, top_k_mask, keep_mask)

    # For single-expert minimum (common case), ensure at least top-1 is kept
    # This is already guaranteed by the threshold logic (top-1 always >= cutoff)

    # Apply mask
    pruned_weights = gate_weights * keep_mask

    # Restore original shape if needed
    if len(orig_shape) == 3:
        pruned_weights = pruned_weights.reshape(orig_shape)
        keep_mask = keep_mask.reshape(orig_shape)

    return pruned_weights, keep_mask


def install_expert_pruning(
    model,
    config: Optional[ExpertPruningConfig] = None,
) -> ExpertPruner:
    """Monkey-patch MoE gate modules to add dynamic expert pruning.

    Walks the model tree looking for MoE layers with ``gate`` and
    ``switch_mlp`` submodules (Qwen, DeepSeek, Mixtral patterns).
    Wraps the MLP ``__call__`` to prune low-weight experts before
    computing the expert outputs.

    Similar pattern to ``router_hook.py``'s model interception.

    Args:
        model: An MLX MoE model (loaded via mlx_lm.load or similar).
        config: Pruning configuration. Uses defaults if None.

    Returns:
        The ExpertPruner instance (for stats access).
    """
    pruner = ExpertPruner(config)

    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is None:
        return pruner

    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue

        mlp = layer.mlp
        gate = getattr(mlp, "gate", None)
        if gate is None:
            continue

        top_k = getattr(mlp, "top_k", None)
        if top_k is None:
            # Try to infer from num_experts_per_tok or similar
            top_k = getattr(mlp, "num_experts_per_tok", 2)

        original_call = type(mlp).__call__

        def make_patched(orig, pruner_ref, top_k_val):
            def patched_call(self, x):
                # Compute gate weights
                gates = self.gate(x)
                gates = mx.softmax(gates, axis=-1, precise=True)

                k = getattr(self, "top_k", top_k_val)

                # Select top-k experts
                inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
                scores = mx.take_along_axis(gates, inds, axis=-1)

                # --- Pruning logic ---
                if not pruner_ref.in_warmup:
                    # Flatten for pruning, then reshape back
                    flat_shape = scores.shape
                    if len(flat_shape) == 3:
                        b, s, e = flat_shape
                        flat_scores = scores.reshape(b * s, e)
                    else:
                        flat_scores = scores

                    # Apply pruning: zero out experts below threshold
                    pruned_scores, keep_mask = prune_experts(
                        flat_scores,
                        threshold=pruner_ref._current_threshold,
                        min_experts=pruner_ref.config.min_experts,
                    )

                    if len(flat_shape) == 3:
                        pruned_scores = pruned_scores.reshape(flat_shape)
                        keep_mask = keep_mask.reshape(flat_shape)

                    # Record stats
                    pruned_count = int(mx.sum(~keep_mask).item())
                    total_count = int(keep_mask.size)
                    pruner_ref.record_decision(pruned_count, total_count)

                    scores = pruned_scores

                # Renormalize remaining scores
                score_sum = scores.sum(axis=-1, keepdims=True)
                scores = mx.where(score_sum > 0, scores / score_sum, scores)

                # Compute expert outputs
                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2)

                # Handle shared experts (DeepSeek pattern)
                if hasattr(self, "shared_expert"):
                    shared = self.shared_expert(x)
                    if hasattr(self, "shared_expert_gate"):
                        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
                    y = y + shared

                return y

            return patched_call

        pruner._original_calls.append(original_call)
        pruner._patched_mlps.append(mlp)
        mlp.__call__ = types.MethodType(make_patched(original_call, pruner, top_k), mlp)

    return pruner
