"""Real MLX router interception for high-accuracy expert prefetching.

Instead of simulating routing decisions, this hooks into the actual MLX
model's gate module to capture real routing decisions during inference.
This is the difference between 6.6% prediction accuracy (simulated)
and 93-97% accuracy (real, per Speculating Experts paper).

Approach: monkey-patch the model's MoE gate forward pass to capture
which experts are selected per layer per token. Feed these into the
co-occurrence predictor for speculative prefetch.

Works with:
  - Qwen MoE models (switch_mlp with gate)
  - Mixtral models (block_sparse_moe with gate)
  - DeepSeek MoE models (gate module)
  - Any MLX model using nn.Module with a 'gate' submodule

Usage:
  model, tokenizer = load("mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit")
  hook = RouterHook(model, num_layers=24, num_experts=60, top_k=4)
  hook.install()

  # Run inference normally — hook captures routing decisions
  output = generate(model, tokenizer, prompt="Hello", max_tokens=100)

  # Get captured routing data
  routing_log = hook.get_routing_log()
  accuracy = hook.measure_prediction_accuracy()
  hook.uninstall()
"""

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@dataclass
class RoutingEvent:
    """A single routing decision captured from the model."""

    token_idx: int
    layer_idx: int
    expert_ids: list[int]
    expert_weights: list[float]


@dataclass
class RouterHookStats:
    """Statistics from router interception."""

    total_events: int = 0
    total_tokens: int = 0
    expert_frequency: dict = field(default_factory=lambda: defaultdict(int))
    layer_expert_frequency: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    prediction_correct: int = 0
    prediction_total: int = 0

    @property
    def prediction_accuracy(self) -> float:
        if self.prediction_total == 0:
            return 0.0
        return self.prediction_correct / self.prediction_total


class RouterHook:
    """Intercepts MLX MoE router decisions during inference.

    Monkey-patches the model's gate modules to capture which experts
    are activated per layer per token. This data feeds the co-occurrence
    predictor for speculative prefetch.
    """

    def __init__(
        self, model, num_layers: int = 24, num_experts: int = 60, top_k: int = 4, on_event: Optional[Callable] = None
    ):
        if not HAS_MLX:
            raise RuntimeError("MLX required for router hooking")

        self.model = model
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self._on_event = on_event  # Optional callback for routing events (e.g., prefetcher)

        self._routing_log: list[RoutingEvent] = []
        self._token_counter: int = 0
        self._installed: bool = False
        self._lock = threading.Lock()

        # Gate replacement tracking (object replacement pattern, same as expert_pruning.py)
        self._patched_mlps: list = []  # MLP objects whose .gate was replaced
        self._original_gates: list = []  # original gate objects for uninstall

        # Co-occurrence predictor
        self._cooccurrence = np.zeros((num_layers, num_experts, num_experts), dtype=np.float32)
        self._prev_experts: dict[int, list[int]] = {}

        self.stats = RouterHookStats()

    def install(self):
        """Install hooks on all MoE gate modules."""
        if self._installed:
            return

        self._find_and_hook_gates(self.model)
        self._installed = True

    def uninstall(self):
        """Remove all hooks, restore original gate modules."""
        for mlp, orig_gate in zip(self._patched_mlps, self._original_gates):
            mlp.gate = orig_gate
        self._patched_mlps.clear()
        self._original_gates.clear()
        self._installed = False

    def _find_and_hook_gates(self, model, prefix=""):
        """Walk model tree and hook all gate modules in MoE layers."""
        self._gate_modules = {}

        # Find gate modules: model.layers[i].mlp.gate
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            return

        for i, layer in enumerate(layers):
            gate = None
            parent = None  # the object whose .gate attribute we replace

            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                gate = layer.mlp.gate
                parent = layer.mlp
            elif hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "gate"):
                gate = layer.block_sparse_moe.gate
                parent = layer.block_sparse_moe

            if gate is not None and parent is not None:
                self._gate_modules[i] = gate
                self._original_gates.append(gate)
                self._patched_mlps.append(parent)
                parent.gate = _HookedGateWrapper(gate, self, i)

    def _record_routing(self, layer_idx: int, expert_ids: list[int], weights: list[float]):
        """Record a routing decision (thread-safe)."""
        with self._lock:
            event = RoutingEvent(
                token_idx=self._token_counter,
                layer_idx=layer_idx,
                expert_ids=expert_ids,
                expert_weights=weights,
            )
            self._routing_log.append(event)
            self.stats.total_events += 1

            # Update co-occurrence
            if layer_idx > 0 and (layer_idx - 1) in self._prev_experts:
                prev = self._prev_experts[layer_idx - 1]
                for p in prev:
                    for c in expert_ids:
                        if p < self.num_experts and c < self.num_experts:
                            self._cooccurrence[layer_idx - 1, p, c] += 1

            self._prev_experts[layer_idx] = expert_ids

            # Update frequency
            for eid in expert_ids:
                self.stats.expert_frequency[eid] += 1
                self.stats.layer_expert_frequency[layer_idx][eid] += 1

        # Notify external handler (e.g., AdvancedPrefetcher) outside the lock
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                pass  # Never block inference due to callback errors

    def advance_token(self):
        """Call between tokens to advance the counter."""
        self._token_counter += 1
        self.stats.total_tokens += 1
        self._prev_experts.clear()

    def predict_next(self, layer_idx: int, current_experts: list[int]) -> list[int]:
        """Predict next layer's experts using co-occurrence data."""
        if layer_idx >= self.num_layers - 1:
            return []

        scores = np.zeros(self.num_experts, dtype=np.float32)
        for eid in current_experts:
            if eid < self.num_experts:
                scores += self._cooccurrence[layer_idx, eid, :]

        if scores.sum() == 0:
            return []

        top_indices = np.argsort(scores)[-self.top_k :][::-1]
        return top_indices.tolist()

    def get_routing_log(self) -> list[RoutingEvent]:
        return self._routing_log.copy()

    def get_expert_heatmap(self) -> np.ndarray:
        """Get (num_layers, num_experts) heatmap of expert activation frequency."""
        heatmap = np.zeros((self.num_layers, self.num_experts), dtype=np.float32)
        for event in self._routing_log:
            for eid in event.expert_ids:
                if event.layer_idx < self.num_layers and eid < self.num_experts:
                    heatmap[event.layer_idx, eid] += 1

        # Normalize per layer
        row_sums = heatmap.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return heatmap / row_sums

    def measure_prediction_accuracy(self) -> float:
        """Measure how well the co-occurrence predictor works on captured data.

        Replays the routing log and checks if predictions match actual selections.
        """
        if len(self._routing_log) < 2:
            return 0.0

        correct = 0
        total = 0
        prev_by_layer: dict[int, list[int]] = {}

        for event in self._routing_log:
            if event.layer_idx > 0 and (event.layer_idx - 1) in prev_by_layer:
                predicted = self.predict_next(event.layer_idx - 1, prev_by_layer[event.layer_idx - 1])
                actual_set = set(event.expert_ids)
                hits = sum(1 for p in predicted if p in actual_set)
                correct += hits
                total += len(event.expert_ids)

            prev_by_layer[event.layer_idx] = event.expert_ids

        self.stats.prediction_correct = correct
        self.stats.prediction_total = total
        return correct / total if total > 0 else 0.0

    def get_hot_experts(self, threshold: float = 0.05) -> dict[int, list[int]]:
        """Get hot experts per layer (frequency above threshold).

        Returns: {layer_idx: [expert_ids sorted by frequency]}
        """
        hot = {}
        for layer_idx in range(self.num_layers):
            freqs = self.stats.layer_expert_frequency.get(layer_idx, {})
            total = sum(freqs.values()) if freqs else 1
            hot_experts = [
                eid for eid, count in sorted(freqs.items(), key=lambda x: -x[1]) if count / total >= threshold
            ]
            if hot_experts:
                hot[layer_idx] = hot_experts
        return hot


class _HookedGateWrapper:
    """Wrapper that replaces a gate module to observe routing decisions.

    Uses the same object-replacement pattern as ``_GateWrapper`` in
    ``expert_pruning.py``.  Python's call protocol resolves ``obj()`` via
    ``type(obj).__call__``, so patching an instance ``__call__`` does not
    work.  Instead we replace the gate object on the parent MLP and
    delegate attribute access to the original gate via ``__getattr__``.

    The wrapper is observation-only: it calls the original gate, extracts
    the top-k expert indices from the logits, records a ``RoutingEvent``
    on the ``RouterHook``, and returns the *unmodified* logits so that
    model behaviour is not affected.
    """

    def __init__(self, original_gate, hook: RouterHook, layer_idx: int):
        self._original_gate = original_gate
        self._hook = hook
        self._layer_idx = layer_idx

    def __call__(self, *args, **kwargs):
        logits = self._original_gate(*args, **kwargs)

        # Capture routing decision (observation only, never block inference)
        try:
            if isinstance(logits, mx.array):
                logits_np = np.array(logits)
                if logits_np.ndim >= 2:
                    # Apply softmax to get expert probabilities
                    last_logits = logits_np.reshape(-1, logits_np.shape[-1])[-1]
                    # Numerically stable softmax
                    shifted = last_logits - last_logits.max()
                    exp_vals = np.exp(shifted)
                    probs = exp_vals / exp_vals.sum()

                    k = min(self._hook.top_k, len(probs))
                    top_k_idx = np.argsort(probs)[-k:][::-1]
                    top_k_weights = probs[top_k_idx]

                    self._hook._record_routing(
                        self._layer_idx,
                        top_k_idx.tolist(),
                        top_k_weights.tolist(),
                    )

                    # Update co-occurrence for experts selected together
                    expert_ids = top_k_idx.tolist()
                    for ii in range(len(expert_ids)):
                        for jj in range(ii + 1, len(expert_ids)):
                            ei, ej = expert_ids[ii], expert_ids[jj]
                            if (
                                ei < self._hook.num_experts
                                and ej < self._hook.num_experts
                                and self._layer_idx < self._hook.num_layers
                            ):
                                self._hook._cooccurrence[self._layer_idx, ei, ej] += 1
                                self._hook._cooccurrence[self._layer_idx, ej, ei] += 1
        except Exception:
            pass  # Never block inference

        return logits

    def __getattr__(self, name):
        return getattr(self._original_gate, name)
