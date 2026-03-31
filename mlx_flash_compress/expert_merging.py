"""Offline expert merging: cluster similar experts into super-experts.

When multiple experts have similar weight matrices (high cosine similarity),
they can be merged into a single "super-expert" with averaged weights.
The routing table maps multiple original expert IDs to the same merged expert,
reducing the number of distinct expert parameters to store/load.

This is different from pruning (which removes experts entirely) — merged
experts preserve coverage while reducing cache pressure by 15-30%.

Inspired by: DEK, EEP, Task-Aware Expert Merging (arXiv:2509.19781).

Usage:
    plan = plan_expert_merges(weight_dict, threshold=0.95)
    merged_weights = apply_merges(weight_dict, plan)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class MergePlan:
    """Plan for merging similar experts."""
    # Mapping: merged_id -> list of original expert IDs
    clusters: dict = field(default_factory=dict)
    # Mapping: original_id -> merged_id
    redirect: dict = field(default_factory=dict)
    # Stats
    original_count: int = 0
    merged_count: int = 0
    similarity_threshold: float = 0.0

    @property
    def reduction(self) -> float:
        if self.original_count == 0:
            return 0.0
        return 1.0 - self.merged_count / self.original_count


def cosine_similarity_matrix(weights: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine similarity between expert weight matrices.

    Each weight matrix is flattened to a vector for comparison.
    """
    n = len(weights)
    flat = [w.flatten().astype(np.float32) for w in weights]
    norms = [np.linalg.norm(f) for f in flat]

    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            if norms[i] > 0 and norms[j] > 0:
                s = np.dot(flat[i], flat[j]) / (norms[i] * norms[j])
                sim[i, j] = s
                sim[j, i] = s
            else:
                sim[i, j] = 0.0
                sim[j, i] = 0.0
    return sim


def plan_expert_merges(expert_weights: list[np.ndarray],
                       threshold: float = 0.95) -> MergePlan:
    """Plan which experts to merge based on weight similarity.

    Args:
        expert_weights: List of weight matrices, one per expert.
        threshold: Cosine similarity threshold for merging (0.95 = very similar).

    Returns:
        MergePlan with cluster assignments and redirect mapping.
    """
    n = len(expert_weights)
    if n == 0:
        return MergePlan()

    sim = cosine_similarity_matrix(expert_weights)

    # Greedy clustering: assign each expert to the first cluster
    # with similarity >= threshold to the cluster representative
    assigned = [False] * n
    clusters = {}
    redirect = {}
    cluster_id = 0

    for i in range(n):
        if assigned[i]:
            continue

        # Start new cluster with expert i as representative
        cluster = [i]
        assigned[i] = True
        redirect[i] = cluster_id

        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if sim[i, j] >= threshold:
                cluster.append(j)
                assigned[j] = True
                redirect[j] = cluster_id

        clusters[cluster_id] = cluster
        cluster_id += 1

    return MergePlan(
        clusters=clusters,
        redirect=redirect,
        original_count=n,
        merged_count=cluster_id,
        similarity_threshold=threshold,
    )


def apply_merges(expert_weights: list[np.ndarray],
                 plan: MergePlan) -> list[np.ndarray]:
    """Apply merge plan: average weights within each cluster.

    Returns a list of merged weight matrices (one per cluster).
    """
    merged = []
    for cluster_id in sorted(plan.clusters.keys()):
        members = plan.clusters[cluster_id]
        if len(members) == 1:
            merged.append(expert_weights[members[0]].copy())
        else:
            avg = np.mean([expert_weights[m] for m in members], axis=0)
            merged.append(avg)
    return merged


def estimate_merge_savings(num_experts: int, threshold: float = 0.95,
                           seed: int = 42) -> dict:
    """Estimate merge savings using synthetic experts with controlled similarity.

    Creates fake expert weights where some are near-duplicates to simulate
    real-world expert redundancy.
    """
    rng = np.random.default_rng(seed)

    # Create num_experts weight matrices, some similar
    base_dim = 64
    weights = []
    num_unique = int(num_experts * 0.7)  # 70% unique
    bases = [rng.standard_normal((base_dim, base_dim)).astype(np.float32)
             for _ in range(num_unique)]

    for i in range(num_experts):
        if i < num_unique:
            weights.append(bases[i])
        else:
            # Clone a random base with small perturbation
            base = bases[rng.integers(0, num_unique)]
            noise_scale = rng.uniform(0.001, 0.05)
            noise = rng.standard_normal(base.shape).astype(np.float32) * noise_scale
            weights.append(base + noise)

    plan = plan_expert_merges(weights, threshold=threshold)

    return {
        "original_experts": num_experts,
        "merged_experts": plan.merged_count,
        "reduction_pct": round(plan.reduction * 100, 1),
        "clusters": len(plan.clusters),
        "largest_cluster": max(len(c) for c in plan.clusters.values()),
        "threshold": threshold,
    }
