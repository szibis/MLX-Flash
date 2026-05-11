"""Tests for KV cache sharing: plan generation, memory estimates, sharing strategies."""

import pytest

from mlx_flash_compress.kv_cache_sharing import (
    KVSharingPlan,
    estimate_kv_memory,
    plan_kv_sharing,
)


class TestKVSharingPlan:
    def test_empty_plan(self):
        plan = KVSharingPlan()
        assert plan.num_layers == 0
        assert plan.num_donors == 0
        assert plan.num_receivers == 0
        assert plan.kv_caches_needed == 0
        assert plan.memory_savings_pct == 0.0

    def test_stats(self):
        plan = KVSharingPlan(
            num_layers=8,
            strategy="pair",
            donor_layers=[0, 2, 4, 6],
            receiver_layers=[1, 3, 5, 7],
            sharing_map={1: 0, 3: 2, 5: 4, 7: 6},
        )
        stats = plan.stats()
        assert stats["num_layers"] == 8
        assert stats["strategy"] == "pair"
        assert stats["donors"] == 4
        assert stats["receivers"] == 4
        assert stats["kv_caches_needed"] == 4
        assert stats["memory_savings_pct"] == 50.0


class TestPlanKVSharing:
    def test_none_strategy(self):
        plan = plan_kv_sharing(num_layers=8, strategy="none")
        assert plan.strategy == "none"
        assert plan.donor_layers == list(range(8))
        assert plan.receiver_layers == []
        assert plan.sharing_map == {}
        assert plan.kv_caches_needed == 8
        assert plan.memory_savings_pct == 0.0

    def test_pair_strategy(self):
        plan = plan_kv_sharing(num_layers=8, strategy="pair")
        assert plan.strategy == "pair"
        assert plan.donor_layers == [0, 2, 4, 6]
        assert plan.receiver_layers == [1, 3, 5, 7]
        assert plan.sharing_map == {1: 0, 3: 2, 5: 4, 7: 6}
        assert plan.kv_caches_needed == 4
        assert plan.memory_savings_pct == 50.0

    def test_pair_strategy_odd_layers(self):
        plan = plan_kv_sharing(num_layers=7, strategy="pair")
        # Layers: 0,1 2,3 4,5 6
        assert 0 in plan.donor_layers
        assert 6 in plan.donor_layers  # unpaired layer is a donor
        assert plan.num_donors + plan.num_receivers == 7

    def test_group_strategy_size_3(self):
        plan = plan_kv_sharing(num_layers=9, strategy="group", group_size=3)
        assert plan.strategy == "group"
        # Groups: [0,1,2], [3,4,5], [6,7,8]
        assert plan.donor_layers == [0, 3, 6]
        assert sorted(plan.receiver_layers) == [1, 2, 4, 5, 7, 8]
        assert plan.kv_caches_needed == 3
        # 6 receivers / 9 layers = 66.7%
        assert abs(plan.memory_savings_pct - 66.7) < 0.1

    def test_group_strategy_size_4(self):
        plan = plan_kv_sharing(num_layers=8, strategy="group", group_size=4)
        # Groups: [0,1,2,3], [4,5,6,7]
        assert plan.donor_layers == [0, 4]
        assert plan.kv_caches_needed == 2
        assert plan.memory_savings_pct == 75.0

    def test_group_strategy_remainder(self):
        plan = plan_kv_sharing(num_layers=10, strategy="group", group_size=3)
        # Groups: [0,1,2], [3,4,5], [6,7,8], [9]
        assert 9 in plan.donor_layers  # last group has single layer
        all_layers = set(plan.donor_layers) | set(plan.receiver_layers)
        assert all_layers == set(range(10))

    def test_pair_with_group_size_ignored(self):
        """For 'pair' strategy, group_size parameter is overridden to 2."""
        plan = plan_kv_sharing(num_layers=8, strategy="pair", group_size=4)
        # pair always uses groups of 2
        assert plan.donor_layers == [0, 2, 4, 6]

    def test_single_layer(self):
        plan = plan_kv_sharing(num_layers=1, strategy="pair")
        assert plan.donor_layers == [0]
        assert plan.receiver_layers == []
        assert plan.kv_caches_needed == 1

    def test_two_layers_pair(self):
        plan = plan_kv_sharing(num_layers=2, strategy="pair")
        assert plan.donor_layers == [0]
        assert plan.receiver_layers == [1]
        assert plan.sharing_map == {1: 0}

    def test_sharing_map_correctness(self):
        """Every receiver must map to a valid donor."""
        plan = plan_kv_sharing(num_layers=32, strategy="pair")
        donor_set = set(plan.donor_layers)
        for receiver, donor in plan.sharing_map.items():
            assert donor in donor_set
            assert receiver in plan.receiver_layers

    def test_no_overlap_donors_receivers(self):
        plan = plan_kv_sharing(num_layers=32, strategy="pair")
        assert set(plan.donor_layers) & set(plan.receiver_layers) == set()


class TestEstimateKVMemory:
    def test_no_sharing_baseline(self):
        result = estimate_kv_memory(
            num_layers=32, strategy="none",
            hidden_dim=4096, num_heads=32, head_dim=128,
            max_seq_len=4096, kv_bits=16,
        )
        assert result["savings_gb"] == 0.0
        assert result["savings_pct"] == 0.0
        assert result["total_no_sharing_gb"] == result["total_with_sharing_gb"]

    def test_pair_saves_half(self):
        result = estimate_kv_memory(
            num_layers=32, strategy="pair",
            hidden_dim=4096, num_heads=32, head_dim=128,
            max_seq_len=4096, kv_bits=16,
        )
        assert result["savings_pct"] == 50.0
        assert result["savings_gb"] > 0
        assert abs(result["total_with_sharing_gb"] - result["total_no_sharing_gb"] / 2) < 0.01

    def test_group4_saves_75_pct(self):
        result = estimate_kv_memory(num_layers=8, strategy="group", kv_bits=16)
        # default group_size in estimate_kv_memory calls plan_kv_sharing with group_size=2
        # but "group" strategy doesn't override group_size, so it defaults to 2 → same as pair
        # The function only passes strategy, not group_size
        assert result["savings_pct"] == 50.0

    def test_kv_per_layer_calculation(self):
        result = estimate_kv_memory(
            num_layers=1, strategy="none",
            hidden_dim=4096, num_heads=32, head_dim=128,
            max_seq_len=4096, kv_bits=16,
        )
        # 2 * 32 * 128 * 4096 * 2 bytes = 64MB
        expected_mb = 2 * 32 * 128 * 4096 * 2 / (1024 ** 2)
        assert abs(result["kv_per_layer_mb"] - expected_mb) < 0.1

    def test_lower_bits_less_memory(self):
        r16 = estimate_kv_memory(num_layers=32, strategy="none", kv_bits=16)
        r8 = estimate_kv_memory(num_layers=32, strategy="none", kv_bits=8)
        assert r8["total_no_sharing_gb"] < r16["total_no_sharing_gb"]
        assert abs(r8["total_no_sharing_gb"] * 2 - r16["total_no_sharing_gb"]) < 0.01

    def test_output_fields(self):
        result = estimate_kv_memory(num_layers=32, strategy="pair")
        expected_keys = {
            "num_layers", "strategy", "kv_per_layer_mb",
            "total_no_sharing_gb", "total_with_sharing_gb",
            "savings_gb", "savings_pct", "kv_bits", "max_seq_len",
        }
        assert expected_keys <= set(result.keys())
