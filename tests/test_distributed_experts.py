"""Tests for distributed expert parallelism: sharding, placement, speedup estimates."""

import os

import pytest

from mlx_flash_compress.distributed_experts import (
    DistributedConfig,
    ExpertShard,
    estimate_distributed_speedup,
    plan_expert_distribution,
)


class TestExpertShard:
    def test_single_node_owns_all(self):
        shard = ExpertShard(rank=0, world_size=1, total_experts=60)
        assert shard.owned_experts == list(range(60))
        assert len(shard.owned_experts) == 60

    def test_two_nodes_even_split(self):
        s0 = ExpertShard(rank=0, world_size=2, total_experts=60)
        s1 = ExpertShard(rank=1, world_size=2, total_experts=60)
        assert s0.owned_experts == list(range(0, 30))
        assert s1.owned_experts == list(range(30, 60))

    def test_three_nodes_remainder(self):
        """Last node picks up remainder experts."""
        s0 = ExpertShard(rank=0, world_size=3, total_experts=10)
        s1 = ExpertShard(rank=1, world_size=3, total_experts=10)
        s2 = ExpertShard(rank=2, world_size=3, total_experts=10)
        assert s0.owned_experts == [0, 1, 2]
        assert s1.owned_experts == [3, 4, 5]
        assert s2.owned_experts == list(range(6, 10))  # gets remainder
        # Total coverage
        all_experts = s0.owned_experts + s1.owned_experts + s2.owned_experts
        assert sorted(all_experts) == list(range(10))

    def test_is_local(self):
        shard = ExpertShard(rank=0, world_size=2, total_experts=10)
        assert shard.is_local(0) is True
        assert shard.is_local(4) is True
        assert shard.is_local(5) is False
        assert shard.is_local(9) is False

    def test_owner_of(self):
        shard = ExpertShard(rank=0, world_size=2, total_experts=10)
        assert shard.owner_of(0) == 0
        assert shard.owner_of(4) == 0
        assert shard.owner_of(5) == 1
        assert shard.owner_of(9) == 1

    def test_owner_of_last_expert(self):
        shard = ExpertShard(rank=0, world_size=3, total_experts=10)
        # Expert 9: 9 // 3 = 3, but clamped to world_size-1 = 2
        assert shard.owner_of(9) == 2

    def test_owner_of_zero_experts_per_node(self):
        shard = ExpertShard(rank=0, world_size=100, total_experts=2)
        # experts_per_node = 0 → returns 0
        assert shard.owner_of(0) == 0
        assert shard.owner_of(1) == 0

    def test_stats(self):
        shard = ExpertShard(rank=0, world_size=2, total_experts=60)
        stats = shard.stats()
        assert stats["rank"] == 0
        assert stats["world_size"] == 2
        assert stats["total_experts"] == 60
        assert stats["local_experts"] == 30
        assert stats["memory_fraction"] == 0.5
        assert "0-29" in stats["expert_range"]

    def test_stats_empty(self):
        shard = ExpertShard(rank=0, world_size=1, total_experts=0, owned_experts=[])
        stats = shard.stats()
        assert stats["local_experts"] == 0
        assert stats["expert_range"] == "none"

    def test_custom_owned_experts(self):
        shard = ExpertShard(rank=0, world_size=2, total_experts=10, owned_experts=[0, 5, 9])
        assert shard.owned_experts == [0, 5, 9]
        assert shard.is_local(5) is True
        assert shard.is_local(1) is False


class TestDistributedConfig:
    def test_default_single_node(self):
        cfg = DistributedConfig()
        assert cfg.world_size == 1
        assert cfg.rank == 0
        assert cfg.is_distributed is False
        assert cfg.is_main is True

    def test_multi_node(self):
        cfg = DistributedConfig(world_size=2, rank=1)
        assert cfg.is_distributed is True
        assert cfg.is_main is False

    def test_from_env(self):
        with _env_vars(MLX_WORLD_SIZE="4", MLX_RANK="2"):
            cfg = DistributedConfig.from_env()
            assert cfg.world_size == 4
            assert cfg.rank == 2

    def test_from_env_defaults(self):
        with _env_vars():
            cfg = DistributedConfig.from_env()
            assert cfg.world_size == 1
            assert cfg.rank == 0

    def test_backend_default(self):
        cfg = DistributedConfig()
        assert cfg.backend == "jaccl"

    def test_tb5_bandwidth(self):
        cfg = DistributedConfig()
        assert cfg.tb5_bandwidth_gbps == 80.0


class TestPlanExpertDistribution:
    def test_two_nodes(self):
        plan = plan_expert_distribution(num_experts=60, num_layers=24, world_size=2, expert_size_mb=50.0)
        assert plan["world_size"] == 2
        assert plan["num_experts"] == 60
        assert plan["experts_per_node"] == 30
        assert plan["remainder_experts"] == 0
        assert plan["memory_savings_pct"] == 50.0
        assert len(plan["nodes"]) == 2

    def test_single_node_no_savings(self):
        plan = plan_expert_distribution(num_experts=60, num_layers=24, world_size=1)
        assert plan["memory_savings_pct"] == 0.0
        assert plan["comm_latency_ms"] == 0.0

    def test_four_nodes(self):
        plan = plan_expert_distribution(num_experts=60, num_layers=24, world_size=4)
        assert plan["experts_per_node"] == 15
        assert plan["memory_savings_pct"] == 75.0
        assert len(plan["nodes"]) == 4

    def test_total_memory(self):
        plan = plan_expert_distribution(num_experts=10, num_layers=2, world_size=1, expert_size_mb=100.0)
        # 10 experts * 2 layers * 100MB / 1024 = 1.953125 GB
        expected = 10 * 2 * 100.0 / 1024
        assert abs(plan["total_memory_gb"] - round(expected, 1)) < 0.2

    def test_communication_latency_increases_with_nodes(self):
        plan2 = plan_expert_distribution(num_experts=60, num_layers=24, world_size=2)
        plan4 = plan_expert_distribution(num_experts=60, num_layers=24, world_size=4)
        assert plan4["comm_latency_ms"] > plan2["comm_latency_ms"]


class TestEstimateDistributedSpeedup:
    def test_single_node_speedup_one(self):
        result = estimate_distributed_speedup(num_experts=10, num_layers=2, world_size=1)
        assert abs(result["speedup"] - 1.0) < 0.05
        assert abs(result["efficiency"] - 1.0) < 0.05

    def test_two_nodes_speedup(self):
        result = estimate_distributed_speedup(
            num_experts=10,
            num_layers=2,
            world_size=2,
            expert_compute_ms=1.0,
            comm_overhead_ms=0.0,
        )
        # With zero comm overhead, speedup should be 2.0
        assert result["speedup"] == 2.0
        assert result["efficiency"] == 1.0

    def test_comm_overhead_reduces_speedup(self):
        result_low = estimate_distributed_speedup(
            num_experts=10,
            num_layers=2,
            world_size=2,
            expert_compute_ms=1.0,
            comm_overhead_ms=0.01,
        )
        result_high = estimate_distributed_speedup(
            num_experts=10,
            num_layers=2,
            world_size=2,
            expert_compute_ms=1.0,
            comm_overhead_ms=1.0,
        )
        assert result_high["speedup"] < result_low["speedup"]

    def test_efficiency_below_one_with_overhead(self):
        result = estimate_distributed_speedup(
            num_experts=10,
            num_layers=2,
            world_size=4,
            expert_compute_ms=1.0,
            comm_overhead_ms=0.5,
        )
        assert result["efficiency"] < 1.0

    def test_output_fields(self):
        result = estimate_distributed_speedup(num_experts=10, num_layers=2, world_size=2)
        assert "world_size" in result
        assert "single_node_ms" in result
        assert "distributed_ms" in result
        assert "speedup" in result
        assert "efficiency" in result
        assert "communication_overhead_pct" in result


# -- Helpers --


class _env_vars:
    """Context manager to temporarily set/clear env vars."""

    def __init__(self, **kwargs):
        self._vars = kwargs
        self._old = {}

    def __enter__(self):
        for key in ["MLX_WORLD_SIZE", "MLX_RANK"]:
            self._old[key] = os.environ.pop(key, None)
        for key, val in self._vars.items():
            os.environ[key] = val
        return self

    def __exit__(self, *args):
        for key in self._vars:
            os.environ.pop(key, None)
        for key, val in self._old.items():
            if val is not None:
                os.environ[key] = val
