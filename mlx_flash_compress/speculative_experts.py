"""Speculative expert techniques: residual predictor, forward eviction, spec execution.

Three advanced techniques from recent MoE inference papers (2025-2026):

1. Residual-Stream Predictor (Speculating Experts, arXiv:2603.19289):
   Uses the pre-MoE hidden state (residual stream) to predict next-layer
   experts via a simple linear projection. Achieves 97-99% accuracy vs
   ~90% for MLP-based shadow models. Zero additional GPU overhead.

2. Forward-Looking Eviction (MoE-SpeQ, arXiv:2511.14102):
   Belady-optimal approximation: never evict experts predicted to be
   needed in the next N steps. Integrates predictions into LCP eviction
   to achieve near-optimal cache hit rates.

3. Speculative Expert Execution (MoE-SpAc, arXiv:2603.09983):
   Execute predicted experts before the router confirms, then verify.
   Accept if correct (>90%), discard otherwise. 14-42% TPOT reduction
   on unified memory (no PCIe penalty for speculative loads).
"""

import numpy as np
from dataclasses import dataclass, field


# -- Residual-stream predictor --

class ResidualPredictor:
    """Predict next-layer experts from the residual stream via linear projection.

    Instead of training an MLP on routing decisions (shadow model approach),
    this uses the pre-MoE hidden state directly. Since adjacent layers have
    >97% cosine similarity in gate inputs (FATE paper), a simple linear
    projection from hidden_dim -> num_experts suffices.

    Key advantage: no additional forward pass — the hidden state is already
    computed as part of normal inference.
    """

    def __init__(self, num_layers: int, num_experts: int, hidden_dim: int = 256,
                 top_k: int = 4, lr: float = 0.005, seed: int = 0):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.lr = lr

        rng = np.random.default_rng(seed)

        # Per-layer linear projection: hidden_dim -> num_experts
        # Much simpler than ShadowPredictor's 2-layer MLP
        self._W = {}
        self._b = {}
        for layer in range(max(num_layers - 1, 1)):
            scale = np.sqrt(2.0 / hidden_dim)
            self._W[layer] = rng.standard_normal((hidden_dim, num_experts)).astype(np.float32) * scale
            self._b[layer] = np.zeros(num_experts, dtype=np.float32)

        self._prev_hidden: dict[int, np.ndarray] = {}
        self._training_steps = 0
        self._total_loss = 0.0

    def observe(self, layer_idx: int, expert_ids: list[int],
                hidden_state: np.ndarray = None):
        """Record routing decision and optionally train from hidden state.

        If hidden_state is provided, trains the predictor online.
        If not, falls back to recording expert IDs only.
        """
        if (hidden_state is not None and layer_idx > 0
                and (layer_idx - 1) in self._prev_hidden):
            pair_idx = layer_idx - 1
            if pair_idx < max(self.num_layers - 1, 1):
                x = self._prev_hidden[pair_idx]
                # Truncate/pad to hidden_dim
                if len(x) > self.hidden_dim:
                    x = x[:self.hidden_dim]
                elif len(x) < self.hidden_dim:
                    x = np.pad(x, (0, self.hidden_dim - len(x)))

                # Target: multi-hot of actual experts
                target = np.zeros(self.num_experts, dtype=np.float32)
                for eid in expert_ids:
                    if 0 <= eid < self.num_experts:
                        target[eid] = 1.0
                target_sum = target.sum()
                if target_sum > 0:
                    target /= target_sum

                # Forward
                logits = x @ self._W[pair_idx] + self._b[pair_idx]
                logits -= logits.max()
                probs = np.exp(logits) / (np.exp(logits).sum() + 1e-8)

                # SGD on cross-entropy
                d_logits = probs - target
                dW = np.outer(x, d_logits)
                db = d_logits

                self._W[pair_idx] -= self.lr * dW
                self._b[pair_idx] -= self.lr * db

                loss = -np.sum(target * np.log(probs + 1e-8))
                self._total_loss += loss
                self._training_steps += 1

        if hidden_state is not None:
            self._prev_hidden[layer_idx] = hidden_state.flatten()[:self.hidden_dim].copy()

    def predict(self, layer_idx: int, hidden_state: np.ndarray = None) -> list[int]:
        """Predict next layer's experts from hidden state."""
        if layer_idx >= self.num_layers - 1:
            return []
        if hidden_state is None:
            return list(range(min(self.top_k, self.num_experts)))

        x = hidden_state.flatten()[:self.hidden_dim]
        if len(x) < self.hidden_dim:
            x = np.pad(x, (0, self.hidden_dim - len(x)))

        logits = x @ self._W[layer_idx] + self._b[layer_idx]
        top_indices = np.argsort(logits)[-self.top_k:][::-1]
        return top_indices.tolist()

    def accuracy(self, predicted: list[int], actual: list[int]) -> float:
        if not actual:
            return 1.0
        hits = sum(1 for a in actual if a in set(predicted))
        return hits / len(actual)

    def stats(self):
        avg_loss = self._total_loss / max(self._training_steps, 1)
        return {
            "training_steps": self._training_steps,
            "avg_loss": round(avg_loss, 4),
            "params_per_layer": self.hidden_dim * self.num_experts,
        }


# -- Forward-looking eviction (Belady-optimal approximation) --

class ForwardLookingEvictor:
    """Belady-optimal eviction: don't evict what you'll need soon.

    Standard LCP looks backward (frequency * decay). This integrator
    looks forward: if the predictor says expert E will be needed in
    the next N steps, protect it from eviction regardless of LCP score.

    Combined score: LCP_score + future_bonus
    where future_bonus = protection_weight / (predicted_distance + 1)
    """

    def __init__(self, num_experts: int, lookahead_steps: int = 3,
                 protection_weight: float = 10.0):
        self.num_experts = num_experts
        self.lookahead_steps = lookahead_steps
        self.protection_weight = protection_weight
        # Track predicted future accesses: expert_id -> steps_until_needed
        self._predicted_needs: dict[int, int] = {}

    def update_predictions(self, predicted_experts: list[int], steps_ahead: int = 1):
        """Update which experts are predicted to be needed."""
        for eid in predicted_experts:
            # Keep the closest prediction
            if eid not in self._predicted_needs or steps_ahead < self._predicted_needs[eid]:
                self._predicted_needs[eid] = steps_ahead

    def clear_predictions(self):
        """Clear stale predictions after each token."""
        # Decrement all distances, remove expired ones
        new_preds = {}
        for eid, dist in self._predicted_needs.items():
            if dist > 1:
                new_preds[eid] = dist - 1
        self._predicted_needs = new_preds

    def eviction_score(self, eid: int, lcp_priority: float) -> float:
        """Compute combined eviction score (higher = keep).

        Adds a future-awareness bonus to the LCP priority.
        """
        bonus = 0.0
        if eid in self._predicted_needs:
            dist = self._predicted_needs[eid]
            bonus = self.protection_weight / (dist + 1)
        return lcp_priority + bonus

    def select_eviction(self, candidates: list[int],
                        lcp_priorities: dict[int, float],
                        n: int) -> list[int]:
        """Select n experts to evict, protecting predicted-needed ones."""
        scored = []
        for eid in candidates:
            lcp_score = lcp_priorities.get(eid, 0.0)
            total_score = self.eviction_score(eid, lcp_score)
            scored.append((total_score, eid))
        scored.sort()  # lowest score first = evict first
        return [eid for _, eid in scored[:n]]


# -- Speculative expert execution --

@dataclass
class SpeculativeResult:
    """Result of speculative expert execution."""
    predicted_experts: list = field(default_factory=list)
    actual_experts: list = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    speculative_saves_ms: float = 0.0

    @property
    def accuracy(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)


class SpeculativeExecutor:
    """Execute predicted experts before router confirms.

    Flow:
    1. Predictor says experts [A, B, C, D] will be needed
    2. Start computing with [A, B, C, D] immediately
    3. Router confirms actual experts [A, B, C, E]
    4. Accept results for A, B, C (hits). Discard D, compute E (miss).

    On unified memory (Apple Silicon), the speculation cost is just
    redundant GPU compute (~0.1ms per expert), not a PCIe transfer.
    """

    def __init__(self, hit_cost_ms: float = 0.0, miss_cost_ms: float = 0.5):
        self.hit_cost_ms = hit_cost_ms  # cost of a correct speculation
        self.miss_cost_ms = miss_cost_ms  # cost of loading a missed expert
        self._total_hits = 0
        self._total_misses = 0
        self._total_saved_ms = 0.0

    def evaluate_speculation(self, predicted: list[int],
                             actual: list[int]) -> SpeculativeResult:
        """Evaluate how well speculation matched reality."""
        pred_set = set(predicted)
        actual_set = set(actual)

        hits = len(pred_set & actual_set)
        misses = len(actual_set - pred_set)
        wasted = len(pred_set - actual_set)

        # Time saved: hits avoid the load latency
        saved = hits * self.miss_cost_ms - wasted * self.hit_cost_ms

        self._total_hits += hits
        self._total_misses += misses
        self._total_saved_ms += max(saved, 0)

        return SpeculativeResult(
            predicted_experts=predicted,
            actual_experts=actual,
            hits=hits,
            misses=misses,
            speculative_saves_ms=round(max(saved, 0), 3),
        )

    def stats(self) -> dict:
        total = self._total_hits + self._total_misses
        return {
            "total_speculations": total,
            "hits": self._total_hits,
            "misses": self._total_misses,
            "accuracy": round(self._total_hits / max(total, 1), 4),
            "total_saved_ms": round(self._total_saved_ms, 2),
        }


# -- Simulation --

def simulate_speculative_pipeline(num_layers: int = 24, num_experts: int = 60,
                                   num_tokens: int = 100, top_k: int = 4,
                                   seed: int = 42) -> dict:
    """Simulate the full speculative pipeline with all three techniques."""
    rng = np.random.default_rng(seed)

    expert_probs = np.array([(1.0 / (i + 1)) ** 0.8 for i in range(num_experts)])
    expert_probs /= expert_probs.sum()

    predictor = ResidualPredictor(num_layers, num_experts, hidden_dim=64,
                                   top_k=top_k, seed=seed)
    evictor = ForwardLookingEvictor(num_experts)
    executor = SpeculativeExecutor()

    warmup = min(20, num_tokens // 2)

    for token in range(num_tokens):
        prev_experts = None
        for layer in range(num_layers):
            actual = rng.choice(num_experts, size=top_k, replace=False,
                               p=expert_probs).tolist()
            hidden = rng.standard_normal(64).astype(np.float32)

            if token >= warmup and prev_experts is not None:
                predicted = predictor.predict(layer - 1, hidden)
                executor.evaluate_speculation(predicted, actual)
                evictor.update_predictions(predicted, steps_ahead=1)

            predictor.observe(layer, actual, hidden_state=hidden)
            prev_experts = actual

        evictor.clear_predictions()

    return {
        "predictor": predictor.stats(),
        "executor": executor.stats(),
        "tokens": num_tokens,
        "warmup": warmup,
    }
