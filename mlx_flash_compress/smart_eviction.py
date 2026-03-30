"""SpecMD-inspired Least-Stale cache eviction policy.

Standard LRU/LFU eviction is suboptimal for MoE expert caching because
expert access patterns are predictable (routing follows power-law distribution
with temporal locality across nearby tokens).

The Least-Stale policy exploits this by estimating which cached experts
are most likely to be needed soon, and evicting the ones with the lowest
predicted future access probability.

Key insight from SpecMD paper (arxiv.org/abs/2602.03921):
  - 85x fewer collision misses vs LRU
  - 88%+ hit rates at only 5% cache capacity
  - 10.7-34.7% TTFT reduction

Our implementation uses three signals:
  1. Frequency (how often this expert is routed overall)
  2. Recency (how recently it was last used)
  3. Layer affinity (experts in earlier layers are accessed more predictably)
"""

import time
import math
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

import numpy as np


@dataclass
class ExpertAccessRecord:
    """Per-expert access history for eviction scoring."""
    total_count: int = 0
    recent_count: int = 0  # accesses in last N tokens
    last_token_pos: int = 0
    layer_idx: int = 0
    expert_id: int = 0


class LeastStalePolicy:
    """SpecMD-inspired eviction policy combining frequency, recency, and prediction.

    Score(expert) = frequency_weight * freq + recency_weight * recency + layer_weight * layer_score

    Higher score = more likely to be needed = KEEP.
    Evict the expert with the LOWEST score.
    """

    def __init__(
        self,
        frequency_weight: float = 0.4,
        recency_weight: float = 0.4,
        layer_weight: float = 0.2,
        recency_window: int = 100,  # tokens
        num_layers: int = 24,
    ):
        self.freq_w = frequency_weight
        self.rec_w = recency_weight
        self.layer_w = layer_weight
        self.recency_window = recency_window
        self.num_layers = num_layers

        # Access tracking
        self._records: dict[tuple[int, int], ExpertAccessRecord] = {}
        self._total_tokens: int = 0
        self._recent_accesses: list[tuple[int, int]] = []  # ring buffer

    def record_access(self, layer_idx: int, expert_id: int):
        """Record an expert access event."""
        key = (layer_idx, expert_id)
        if key not in self._records:
            self._records[key] = ExpertAccessRecord(
                layer_idx=layer_idx, expert_id=expert_id
            )
        rec = self._records[key]
        rec.total_count += 1
        rec.last_token_pos = self._total_tokens

        self._recent_accesses.append(key)
        if len(self._recent_accesses) > self.recency_window * 4:
            # Trim to keep memory bounded
            self._recent_accesses = self._recent_accesses[-self.recency_window:]

    def advance_token(self):
        """Call once per generated token to advance the clock."""
        self._total_tokens += 1

    def score(self, layer_idx: int, expert_id: int) -> float:
        """Compute retention score for an expert (higher = keep)."""
        key = (layer_idx, expert_id)
        rec = self._records.get(key)

        if rec is None:
            return 0.0

        # Frequency component: fraction of total tokens that used this expert
        freq = rec.total_count / max(self._total_tokens, 1)

        # Recency component: exponential decay based on tokens since last access
        tokens_since = self._total_tokens - rec.last_token_pos
        recency = math.exp(-tokens_since / max(self.recency_window, 1))

        # Layer affinity: earlier layers have more predictable routing
        # (tokens processed sequentially, router decisions correlate across positions)
        layer_score = 1.0 - (layer_idx / max(self.num_layers, 1))

        return (
            self.freq_w * freq +
            self.rec_w * recency +
            self.layer_w * layer_score
        )

    def select_eviction(
        self,
        cached_keys: list[tuple[int, int]],
    ) -> tuple[int, int]:
        """Select the best expert to evict (lowest score)."""
        if not cached_keys:
            raise ValueError("No cached keys to evict")

        min_score = float('inf')
        min_key = cached_keys[0]

        for key in cached_keys:
            s = self.score(key[0], key[1])
            if s < min_score:
                min_score = s
                min_key = key

        return min_key

    def batch_evict(
        self,
        cached_keys: list[tuple[int, int]],
        num_to_evict: int,
    ) -> list[tuple[int, int]]:
        """Select multiple experts to evict, sorted by ascending score."""
        scored = [(self.score(k[0], k[1]), k) for k in cached_keys]
        scored.sort(key=lambda x: x[0])
        return [k for _, k in scored[:num_to_evict]]


class RoutingPredictor:
    """Lightweight expert routing predictor for speculative prefetching.

    Uses a simple approach: for each layer, track the co-occurrence matrix
    between current-layer experts and next-layer experts. When we know
    which experts layer L activated, predict layer L+1's experts using
    the co-occurrence statistics.

    Reference: Speculating Experts (arxiv.org/abs/2603.19289) achieves
    93-97% prediction accuracy with this approach.
    """

    def __init__(self, num_layers: int, num_experts: int, top_k: int = 4):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.top_k = top_k

        # Co-occurrence matrix: for each layer pair (L, L+1),
        # count how often expert i at layer L co-occurs with expert j at L+1
        # Shape: (num_layers-1, num_experts, num_experts)
        self._cooccurrence = np.zeros(
            (num_layers - 1, num_experts, num_experts), dtype=np.float32
        )
        self._prev_experts: dict[int, list[int]] = {}
        self._total_observations = 0

    def observe(self, layer_idx: int, expert_ids: list[int]):
        """Record which experts were activated at this layer.

        If we have the previous layer's experts, update co-occurrence.
        """
        if layer_idx > 0 and (layer_idx - 1) in self._prev_experts:
            prev = self._prev_experts[layer_idx - 1]
            for p in prev:
                for c in expert_ids:
                    self._cooccurrence[layer_idx - 1, p, c] += 1

        self._prev_experts[layer_idx] = expert_ids
        self._total_observations += 1

    def predict(self, layer_idx: int, current_experts: list[int]) -> list[int]:
        """Predict which experts layer_idx+1 will activate.

        Uses the co-occurrence matrix to score all next-layer experts,
        then returns top-K.
        """
        if layer_idx >= self.num_layers - 1:
            return []

        # Sum co-occurrence scores from all current experts
        scores = np.zeros(self.num_experts, dtype=np.float32)
        for eid in current_experts:
            scores += self._cooccurrence[layer_idx, eid, :]

        # Return top-K predicted experts
        if scores.sum() == 0:
            # No data yet — return most globally frequent experts
            return list(range(min(self.top_k, self.num_experts)))

        top_indices = np.argsort(scores)[-self.top_k:][::-1]
        return top_indices.tolist()

    def accuracy(self, predicted: list[int], actual: list[int]) -> float:
        """Measure prediction accuracy (fraction of actual experts that were predicted)."""
        if not actual:
            return 1.0
        pred_set = set(predicted)
        hits = sum(1 for a in actual if a in pred_set)
        return hits / len(actual)


@dataclass
class PrefetchResult:
    """Statistics from a prefetch simulation run."""
    total_predictions: int = 0
    correct_predictions: int = 0
    total_experts_prefetched: int = 0
    wasted_prefetches: int = 0
    avg_accuracy: float = 0.0
    prefetch_hit_rate: float = 0.0


def simulate_prefetch(
    num_layers: int,
    num_experts: int,
    num_tokens: int,
    top_k: int = 4,
    seed: int = 42,
) -> PrefetchResult:
    """Simulate speculative expert prefetching over a token generation sequence.

    Uses power-law routing (Zipf) to simulate realistic expert selection,
    trains the predictor online, and measures prediction accuracy.
    """
    rng = np.random.default_rng(seed)
    predictor = RoutingPredictor(num_layers, num_experts, top_k)

    expert_probs = np.array([(1.0 / (i + 1)) ** 0.8 for i in range(num_experts)])
    expert_probs /= expert_probs.sum()

    result = PrefetchResult()
    accuracies = []

    # Warmup phase: first 10 tokens to build co-occurrence stats
    warmup = min(10, num_tokens)
    for token in range(warmup):
        for layer in range(num_layers):
            experts = rng.choice(num_experts, size=top_k, replace=False, p=expert_probs).tolist()
            predictor.observe(layer, experts)

    # Evaluation phase
    for token in range(warmup, num_tokens):
        for layer in range(num_layers):
            actual = rng.choice(num_experts, size=top_k, replace=False, p=expert_probs).tolist()

            # Predict from previous layer's experts
            if layer > 0:
                predicted = predictor.predict(layer - 1, prev_experts)
                acc = predictor.accuracy(predicted, actual)
                accuracies.append(acc)
                result.total_predictions += 1
                result.correct_predictions += sum(1 for a in actual if a in predicted)
                result.total_experts_prefetched += len(predicted)
                result.wasted_prefetches += sum(1 for p in predicted if p not in actual)

            predictor.observe(layer, actual)
            prev_experts = actual

    if accuracies:
        result.avg_accuracy = float(np.mean(accuracies))
    if result.total_experts_prefetched > 0:
        result.prefetch_hit_rate = result.correct_predictions / result.total_experts_prefetched

    return result
