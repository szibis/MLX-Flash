"""Tests for distributed experts, KV-cache sharing, and HF calculator."""
import pytest
from mlx_flash_compress.distributed_experts import (
    ExpertShard, DistributedConfig, plan_expert_distribution,
    estimate_distributed_speedup,
)
from mlx_flash_compress.kv_cache_sharing import (
    plan_kv_sharing, estimate_kv_memory, KVSharingPlan,
)
from mlx_flash_compress.hf_calculator import estimate_model, format_estimate


class TestExpertShard:
    def test_single_node(self):
        shard = ExpertShard(rank=0, world_size=1, total_experts=60)
        assert len(shard.owned_experts) == 60
        assert shard.is_local(0)
        assert shard.is_local(59)

    def test_two_nodes(self):
        s0 = ExpertShard(rank=0, world_size=2, total_experts=60)
        s1 = ExpertShard(rank=1, world_size=2, total_experts=60)
        assert len(s0.owned_experts) == 30
        assert len(s1.owned_experts) == 30
        assert s0.is_local(0)
        assert not s0.is_local(30)
        assert s1.is_local(30)
        assert not s1.is_local(0)

    def test_owner_of(self):
        shard = ExpertShard(rank=0, world_size=4, total_experts=60)
        assert shard.owner_of(0) == 0
        assert shard.owner_of(14) == 0
        assert shard.owner_of(15) == 1
        assert shard.owner_of(59) == 3

    def test_stats(self):
        shard = ExpertShard(rank=0, world_size=2, total_experts=60)
        s = shard.stats()
        assert s["local_experts"] == 30
        assert s["memory_fraction"] == 0.5


class TestDistributedConfig:
    def test_from_env_default(self):
        cfg = DistributedConfig.from_env()
        assert cfg.world_size >= 1
        assert cfg.rank >= 0

    def test_not_distributed(self):
        cfg = DistributedConfig(world_size=1)
        assert not cfg.is_distributed
        assert cfg.is_main


class TestPlanDistribution:
    def test_basic(self):
        plan = plan_expert_distribution(num_experts=60, num_layers=24, world_size=2)
        assert plan["world_size"] == 2
        assert plan["experts_per_node"] == 30
        assert plan["memory_savings_pct"] == 50.0
        assert len(plan["nodes"]) == 2

    def test_single_node(self):
        plan = plan_expert_distribution(num_experts=8, num_layers=32, world_size=1)
        assert plan["memory_savings_pct"] == 0.0

    def test_speedup(self):
        s = estimate_distributed_speedup(num_experts=60, num_layers=24, world_size=2)
        assert s["speedup"] > 1.0
        assert s["efficiency"] > 0 and s["efficiency"] <= 1.0


class TestKVSharing:
    def test_pair_sharing(self):
        plan = plan_kv_sharing(num_layers=8, strategy="pair")
        assert plan.num_donors == 4
        assert plan.num_receivers == 4
        assert plan.memory_savings_pct == 50.0

    def test_group_sharing(self):
        plan = plan_kv_sharing(num_layers=12, strategy="group", group_size=3)
        assert plan.num_donors == 4
        assert plan.num_receivers == 8
        assert plan.memory_savings_pct > 60

    def test_no_sharing(self):
        plan = plan_kv_sharing(num_layers=8, strategy="none")
        assert plan.num_donors == 8
        assert plan.num_receivers == 0
        assert plan.memory_savings_pct == 0.0

    def test_sharing_map(self):
        plan = plan_kv_sharing(num_layers=6, strategy="pair")
        assert plan.sharing_map[1] == 0
        assert plan.sharing_map[3] == 2
        assert plan.sharing_map[5] == 4

    def test_estimate_memory(self):
        est = estimate_kv_memory(num_layers=32, strategy="pair")
        assert est["savings_gb"] > 0
        assert est["savings_pct"] == 50.0
        assert est["total_with_sharing_gb"] < est["total_no_sharing_gb"]

    def test_estimate_with_8bit(self):
        fp16 = estimate_kv_memory(num_layers=32, kv_bits=16, strategy="none")
        int8 = estimate_kv_memory(num_layers=32, kv_bits=8, strategy="none")
        assert int8["total_no_sharing_gb"] < fp16["total_no_sharing_gb"]


class TestHFCalculator:
    def test_known_model(self):
        est = estimate_model(model_name="Qwen3-30B-A3B", ram_gb=36)
        assert est["type"] == "MoE"
        assert est["total_params_b"] == 30
        assert est["num_experts"] == 128
        assert est["total_size_gb"] > 0

    def test_dense_model(self):
        est = estimate_model(model_name="Qwen3-8B", ram_gb=8)
        assert est["type"] == "Dense"
        assert est["fits_full"]

    def test_huge_model(self):
        est = estimate_model(model_name="DeepSeek-V3-671B", ram_gb=36)
        assert not est["fits_full"]  # 671B at 4-bit = ~335GB, way too large
        assert est["total_size_gb"] > 100  # huge model

    def test_format_estimate(self):
        est = estimate_model(model_name="Mixtral-8x7B", ram_gb=48)
        text = format_estimate(est)
        assert "Mixtral-8x7B" in text
        assert "MoE" in text
        assert "GB" in text

    def test_custom_model(self):
        est = estimate_model(total_params_b=100, active_params_b=10,
                              num_experts=64, num_layers=40, ram_gb=48)
        assert est["type"] == "MoE"
        assert est["savings_vs_full_pct"] > 0
