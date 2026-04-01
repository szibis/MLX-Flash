"""PT-MoE KV-cache sharing: reuse KV cache between adjacent transformer blocks.

Apple's PT-MoE architecture (Tech Report 2025, arXiv:2507.13575) uses a
"parallel track" design where Block 2 reuses Block 1's KV cache instead of
computing its own. This saves 37.5% of KV memory with zero quality loss.

The insight: in many transformer architectures, adjacent blocks produce
very similar KV projections. Sharing one block's KV cache with the next
eliminates redundant computation and memory.

Sharing patterns:
  - Pair sharing: block[i] shares with block[i+1] (37.5% savings)
  - Group sharing: every N blocks share (higher savings, slight quality loss)
  - Selective: only share between blocks with high KV similarity

Usage:
  plan = plan_kv_sharing(num_layers=32, strategy="pair")
  # plan.donor_layers = [0, 2, 4, ...] — these compute KV
  # plan.receiver_layers = [1, 3, 5, ...] — these reuse donor's KV
"""

from dataclasses import dataclass, field


@dataclass
class KVSharingPlan:
    """Plan for which layers share KV caches."""
    num_layers: int = 0
    strategy: str = "pair"
    # Donor layers: compute their own KV cache
    donor_layers: list = field(default_factory=list)
    # Receiver layers: reuse a donor's KV cache
    receiver_layers: list = field(default_factory=list)
    # Mapping: receiver_layer -> donor_layer
    sharing_map: dict = field(default_factory=dict)

    @property
    def num_donors(self) -> int:
        return len(self.donor_layers)

    @property
    def num_receivers(self) -> int:
        return len(self.receiver_layers)

    @property
    def memory_savings_pct(self) -> float:
        if self.num_layers == 0:
            return 0.0
        return self.num_receivers / self.num_layers * 100

    @property
    def kv_caches_needed(self) -> int:
        """Number of unique KV caches needed (vs num_layers without sharing)."""
        return self.num_donors

    def stats(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "strategy": self.strategy,
            "donors": self.num_donors,
            "receivers": self.num_receivers,
            "kv_caches_needed": self.kv_caches_needed,
            "memory_savings_pct": round(self.memory_savings_pct, 1),
        }


def plan_kv_sharing(num_layers: int, strategy: str = "pair",
                     group_size: int = 2) -> KVSharingPlan:
    """Plan KV-cache sharing between transformer layers.

    Strategies:
      - "pair": adjacent pairs share (block 0→1, 2→3, etc.) — 50% savings
      - "group": groups of N share one KV cache — (N-1)/N savings
      - "none": no sharing (baseline)
    """
    plan = KVSharingPlan(num_layers=num_layers, strategy=strategy)

    if strategy == "none":
        plan.donor_layers = list(range(num_layers))
        return plan

    if strategy == "pair":
        group_size = 2

    for start in range(0, num_layers, group_size):
        group = list(range(start, min(start + group_size, num_layers)))
        if not group:
            continue
        donor = group[0]
        plan.donor_layers.append(donor)
        for layer in group[1:]:
            plan.receiver_layers.append(layer)
            plan.sharing_map[layer] = donor

    return plan


def estimate_kv_memory(num_layers: int, hidden_dim: int = 4096,
                        num_heads: int = 32, head_dim: int = 128,
                        max_seq_len: int = 4096, kv_bits: int = 16,
                        strategy: str = "pair") -> dict:
    """Estimate KV cache memory with and without sharing.

    Returns memory estimates in GB.
    """
    # KV cache per layer: 2 (K+V) * num_heads * head_dim * max_seq_len * bytes
    bytes_per_element = kv_bits / 8
    kv_per_layer = 2 * num_heads * head_dim * max_seq_len * bytes_per_element
    kv_per_layer_gb = kv_per_layer / (1024 ** 3)

    # Without sharing
    total_no_sharing = kv_per_layer_gb * num_layers

    # With sharing
    plan = plan_kv_sharing(num_layers, strategy=strategy)
    total_with_sharing = kv_per_layer_gb * plan.kv_caches_needed

    savings_gb = total_no_sharing - total_with_sharing

    return {
        "num_layers": num_layers,
        "strategy": strategy,
        "kv_per_layer_mb": round(kv_per_layer / (1024 ** 2), 1),
        "total_no_sharing_gb": round(total_no_sharing, 2),
        "total_with_sharing_gb": round(total_with_sharing, 2),
        "savings_gb": round(savings_gb, 2),
        "savings_pct": round(plan.memory_savings_pct, 1),
        "kv_bits": kv_bits,
        "max_seq_len": max_seq_len,
    }
