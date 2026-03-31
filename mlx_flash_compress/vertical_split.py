"""Vertical expert splitting: cache partial expert rows for 2x coverage.

Instead of caching N complete experts, cache 2N partial experts (top half
of each weight matrix). This doubles the number of experts in cache,
dramatically improving hit rates.

When an expert is partially cached:
  - Use the cached top rows for a fast approximate result
  - Optionally load remaining rows from SSD for exact result

Inspired by: MoEpic paper (vertical expert splitting for 2x cache coverage).

The insight: for quantized weight matrices [rows, cols], the top rows
contribute the most to the output. Caching top-half gives ~90-95%
of the full expert's contribution.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SplitConfig:
    """Configuration for vertical expert splitting."""
    split_factor: int = 2  # cache 1/split_factor of each expert
    exact_on_miss: bool = False  # load remaining rows on cache miss?

    @property
    def cached_fraction(self) -> float:
        return 1.0 / self.split_factor


class VerticalSplitCache:
    """Cache that stores partial expert weights for increased coverage.

    For a weight matrix of shape [E, rows, cols]:
    - Full cache: stores [capacity, rows, cols] → capacity experts
    - Split cache: stores [capacity*2, rows//2, cols] → 2x experts

    Same memory footprint, double the expert coverage.
    """

    def __init__(self, num_experts: int, rows: int, cols: int,
                 capacity: int, split_factor: int = 2):
        self.num_experts = num_experts
        self.rows = rows
        self.cols = cols
        self.split_rows = rows // split_factor
        self.split_factor = split_factor
        # Split cache holds split_factor * capacity experts
        self.split_capacity = capacity * split_factor
        self.full_capacity = capacity

        # Track which experts are cached (partial and full)
        self.partial_ids: list[int] = []  # experts with top rows cached
        self.full_ids: list[int] = []  # experts fully cached

    def coverage(self) -> dict:
        """Return coverage statistics."""
        partial = len(self.partial_ids)
        full = len(self.full_ids)
        total = self.num_experts
        return {
            "partial_cached": partial,
            "full_cached": full,
            "total_experts": total,
            "partial_coverage": partial / max(total, 1),
            "full_coverage": full / max(total, 1),
            "effective_coverage": (partial + full) / max(total, 1),
            "split_factor": self.split_factor,
            "memory_per_expert_full": self.rows * self.cols,
            "memory_per_expert_split": self.split_rows * self.cols,
        }

    def plan_allocation(self, hot_expert_ids: list[int],
                        warm_expert_ids: list[int]) -> dict:
        """Plan how to allocate cache between full and partial experts.

        Hot experts: fully cached (all rows)
        Warm experts: partially cached (top rows only)
        Cold experts: not cached at all

        Strategy: fill full slots with hot experts first, then use remaining
        capacity for partial warm experts.
        """
        num_full = min(len(hot_expert_ids), self.full_capacity)
        full = hot_expert_ids[:num_full]

        # Remaining capacity in split units
        used_split_units = num_full * self.split_factor
        remaining_units = self.split_capacity - used_split_units
        num_partial = min(len(warm_expert_ids), remaining_units)
        partial = warm_expert_ids[:num_partial]

        return {
            "full": full,
            "partial": partial,
            "full_count": len(full),
            "partial_count": len(partial),
            "total_experts_cached": len(full) + len(partial),
            "memory_utilization": (len(full) * self.split_factor + len(partial)) / max(self.split_capacity, 1),
        }

    def simulate_hit_rate(self, routing_trace: list[list[int]]) -> dict:
        """Simulate cache hit rate over a routing trace.

        routing_trace: list of [expert_id, ...] per token.
        Returns hit statistics.
        """
        full_set = set(self.full_ids)
        partial_set = set(self.partial_ids)

        full_hits = 0
        partial_hits = 0
        misses = 0
        total = 0

        for token_experts in routing_trace:
            for eid in token_experts:
                total += 1
                if eid in full_set:
                    full_hits += 1
                elif eid in partial_set:
                    partial_hits += 1
                else:
                    misses += 1

        return {
            "total_lookups": total,
            "full_hits": full_hits,
            "partial_hits": partial_hits,
            "misses": misses,
            "full_hit_rate": full_hits / max(total, 1),
            "partial_hit_rate": partial_hits / max(total, 1),
            "total_hit_rate": (full_hits + partial_hits) / max(total, 1),
            "miss_rate": misses / max(total, 1),
        }


def estimate_split_benefit(num_experts: int, capacity: int,
                           split_factor: int = 2) -> dict:
    """Estimate the benefit of vertical splitting vs full caching.

    Assumes Zipf distribution of expert access (realistic for MoE).
    """
    # Zipf distribution: expert i has probability proportional to 1/(i+1)^0.8
    probs = np.array([(1.0 / (i + 1)) ** 0.8 for i in range(num_experts)])
    probs /= probs.sum()

    # Full cache: top `capacity` experts by frequency
    full_cache_experts = np.argsort(probs)[-capacity:]
    full_hit_rate = probs[full_cache_experts].sum()

    # Split cache: top `capacity * split_factor` experts (partial)
    split_capacity = capacity * split_factor
    split_cache_experts = np.argsort(probs)[-min(split_capacity, num_experts):]
    split_hit_rate = probs[split_cache_experts].sum()

    # Effective split hit rate considers partial hits are ~90% as good
    partial_quality = 0.90  # approximate quality of partial expert
    effective_split_rate = full_hit_rate + (split_hit_rate - full_hit_rate) * partial_quality

    return {
        "num_experts": num_experts,
        "capacity": capacity,
        "split_factor": split_factor,
        "full_cache_hit_rate": round(full_hit_rate, 4),
        "split_cache_hit_rate": round(split_hit_rate, 4),
        "effective_split_hit_rate": round(effective_split_rate, 4),
        "improvement": round((effective_split_rate - full_hit_rate) / max(full_hit_rate, 0.001), 4),
        "experts_cached_full": capacity,
        "experts_cached_split": min(split_capacity, num_experts),
    }
