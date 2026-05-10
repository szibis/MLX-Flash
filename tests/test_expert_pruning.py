"""Tests for dynamic expert pruning."""

import numpy as np
import pytest

try:
    import mlx.core as mx

    from mlx_flash_compress.expert_pruning import (
        ExpertPruner,
        ExpertPruningConfig,
        install_expert_pruning,
        prune_experts,
    )

    HAS_MODULE = True
except (ImportError, ModuleNotFoundError):
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="expert_pruning requires mlx")


# -- ExpertPruningConfig tests --


class TestExpertPruningConfig:
    def test_defaults(self):
        cfg = ExpertPruningConfig()
        assert cfg.gate_threshold == 0.05
        assert cfg.min_experts == 1
        assert cfg.adaptive is True
        assert cfg.warmup_tokens == 100

    def test_custom_values(self):
        cfg = ExpertPruningConfig(gate_threshold=0.1, min_experts=2, adaptive=False, warmup_tokens=50)
        assert cfg.gate_threshold == 0.1
        assert cfg.min_experts == 2
        assert cfg.adaptive is False
        assert cfg.warmup_tokens == 50

    def test_invalid_threshold_negative(self):
        with pytest.raises(ValueError, match="gate_threshold"):
            ExpertPruningConfig(gate_threshold=-0.1)

    def test_invalid_threshold_above_one(self):
        with pytest.raises(ValueError, match="gate_threshold"):
            ExpertPruningConfig(gate_threshold=1.5)

    def test_invalid_min_experts_zero(self):
        with pytest.raises(ValueError, match="min_experts"):
            ExpertPruningConfig(min_experts=0)

    def test_invalid_warmup_negative(self):
        with pytest.raises(ValueError, match="warmup_tokens"):
            ExpertPruningConfig(warmup_tokens=-1)

    def test_edge_threshold_zero(self):
        cfg = ExpertPruningConfig(gate_threshold=0.0)
        assert cfg.gate_threshold == 0.0

    def test_edge_threshold_one(self):
        cfg = ExpertPruningConfig(gate_threshold=1.0)
        assert cfg.gate_threshold == 1.0


# -- ExpertPruner tests --


class TestExpertPruner:
    def test_default_init(self):
        pruner = ExpertPruner()
        assert pruner.config.gate_threshold == 0.05
        assert pruner.in_warmup is True
        assert pruner._tokens_seen == 0

    def test_warmup_computes_all(self):
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=10))
        mask = pruner.should_compute([0.8, 0.15, 0.03, 0.02])
        assert mask == [True, True, True, True]

    def test_pruning_after_warmup(self):
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0, gate_threshold=0.05, adaptive=False))
        # 0.03 < 0.05 * 0.8 = 0.04, so expert 3 pruned
        # 0.02 < 0.04, so expert 4 pruned
        mask = pruner.should_compute([0.8, 0.15, 0.03, 0.02])
        assert mask[0] is True  # top-1 always kept
        assert mask[1] is True  # 0.15 >= 0.04
        assert mask[2] is False  # 0.03 < 0.04
        assert mask[3] is False  # 0.02 < 0.04

    def test_all_above_threshold(self):
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0, gate_threshold=0.01, adaptive=False))
        mask = pruner.should_compute([0.4, 0.3, 0.2, 0.1])
        assert mask == [True, True, True, True]

    def test_all_below_threshold_min_experts(self):
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0, gate_threshold=0.99, min_experts=1, adaptive=False))
        # All except top-1 are < 0.99 * 0.5 = 0.495
        mask = pruner.should_compute([0.5, 0.3, 0.15, 0.05])
        assert mask[0] is True  # min_experts=1 guarantees top-1
        # At least 1 expert must be True
        assert sum(mask) >= 1

    def test_min_experts_guarantee(self):
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0, gate_threshold=0.99, min_experts=3, adaptive=False))
        mask = pruner.should_compute([0.4, 0.3, 0.2, 0.1])
        assert sum(mask) >= 3

    def test_single_expert(self):
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0, gate_threshold=0.05, adaptive=False))
        mask = pruner.should_compute([1.0])
        assert mask == [True]

    def test_empty_weights(self):
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0))
        mask = pruner.should_compute([])
        assert mask == []

    def test_zero_weights(self):
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0, min_experts=2, adaptive=False))
        mask = pruner.should_compute([0.0, 0.0, 0.0])
        # min_experts=2 should keep at least 2
        assert sum(mask) >= 2

    def test_record_decision_updates_stats(self):
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0, adaptive=False))
        pruner.record_decision(pruned_count=2, total_count=4)
        pruner.record_decision(pruned_count=1, total_count=4)

        stats = pruner.get_stats()
        assert stats["decisions"] == 2
        assert stats["total_pruned"] == 3
        assert stats["total_experts"] == 8
        assert stats["tokens_seen"] == 2

    def test_prune_rate_calculation(self):
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0, adaptive=False))
        # Prune 3 out of 10 experts
        pruner.record_decision(pruned_count=3, total_count=10)
        stats = pruner.get_stats()
        assert stats["prune_rate"] == 0.3

    def test_warmup_transition(self):
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=2))
        assert pruner.in_warmup is True

        # During warmup, all experts computed
        mask1 = pruner.should_compute([0.9, 0.01])
        assert mask1 == [True, True]

        # Advance past warmup
        pruner.record_decision(0, 4)
        pruner.record_decision(0, 4)
        assert pruner.in_warmup is False

        # Now pruning should apply (0.01 < 0.05 * 0.9 = 0.045)
        pruner.config.adaptive = False
        mask2 = pruner.should_compute([0.9, 0.01])
        assert mask2[0] is True
        assert mask2[1] is False

    def test_adaptive_threshold_aggressive_prune(self):
        """When pruning too aggressively, adaptive mode raises threshold."""
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0, gate_threshold=0.05, adaptive=True))
        initial_threshold = pruner._current_threshold

        # Record many aggressive prunes (>80% pruned)
        for _ in range(50):
            pruner.record_decision(pruned_count=9, total_count=10)

        # Threshold should have increased (more conservative)
        assert pruner._current_threshold > initial_threshold

    def test_adaptive_threshold_conservative(self):
        """When barely pruning, adaptive mode lowers threshold."""
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0, gate_threshold=0.05, adaptive=True))
        initial_threshold = pruner._current_threshold

        # Record many conservative prunes (<10% pruned)
        for _ in range(50):
            pruner.record_decision(pruned_count=0, total_count=10)

        # Threshold should have decreased (more aggressive)
        assert pruner._current_threshold < initial_threshold

    def test_stats_output_format(self):
        pruner = ExpertPruner()
        stats = pruner.get_stats()
        expected_keys = {
            "decisions",
            "tokens_seen",
            "total_pruned",
            "total_experts",
            "avg_pruned_per_token",
            "prune_rate",
            "current_threshold",
            "base_threshold",
            "adaptive",
            "in_warmup",
            "ema_ratio",
        }
        assert set(stats.keys()) == expected_keys


# -- prune_experts functional API tests --


class TestPruneExperts:
    def test_basic_2d(self):
        weights = mx.array([[0.7, 0.2, 0.08, 0.02]])
        pruned, mask = prune_experts(weights, threshold=0.05, min_experts=1)

        pruned_np = np.array(pruned.tolist())
        mask_np = np.array(mask.tolist())

        # 0.02 < 0.05 * 0.7 = 0.035, should be zeroed
        assert pruned_np[0, 0] > 0  # top-1 kept
        assert pruned_np[0, 1] > 0  # 0.2 >= 0.035
        assert pruned_np[0, 2] > 0  # 0.08 >= 0.035
        assert pruned_np[0, 3] == 0  # 0.02 < 0.035

    def test_all_above_threshold(self):
        weights = mx.array([[0.4, 0.3, 0.2, 0.1]])
        pruned, mask = prune_experts(weights, threshold=0.01, min_experts=1)
        pruned_np = np.array(pruned.tolist())
        # All >= 0.01 * 0.4 = 0.004
        assert all(pruned_np[0] > 0)

    def test_all_below_threshold(self):
        # threshold=0.99: only keep experts >= 0.99 * top_weight
        weights = mx.array([[0.5, 0.3, 0.15, 0.05]])
        pruned, mask = prune_experts(weights, threshold=0.99, min_experts=1)

        mask_np = np.array(mask.tolist())
        # Only top-1 should pass (0.3 < 0.99 * 0.5 = 0.495)
        assert mask_np[0, 0] is True or mask_np[0, 0] == 1
        # min_experts=1 guarantees at least 1 kept
        assert mask_np.sum() >= 1

    def test_min_experts_2(self):
        weights = mx.array([[0.9, 0.05, 0.03, 0.02]])
        pruned, mask = prune_experts(weights, threshold=0.5, min_experts=2)

        mask_np = np.array(mask.tolist())
        assert mask_np.sum() >= 2

    def test_batch_dimension(self):
        weights = mx.array(
            [
                [0.7, 0.2, 0.08, 0.02],
                [0.4, 0.3, 0.2, 0.1],
            ]
        )
        pruned, mask = prune_experts(weights, threshold=0.05, min_experts=1)

        assert pruned.shape == weights.shape
        assert mask.shape == weights.shape

    def test_3d_input(self):
        weights = mx.array([[[0.7, 0.2, 0.08, 0.02], [0.4, 0.3, 0.2, 0.1]]])
        pruned, mask = prune_experts(weights, threshold=0.05, min_experts=1)

        assert pruned.shape == weights.shape  # (1, 2, 4)
        assert mask.shape == weights.shape

    def test_preserves_kept_values(self):
        weights = mx.array([[0.6, 0.3, 0.1]])
        pruned, mask = prune_experts(weights, threshold=0.05, min_experts=1)

        pruned_np = np.array(pruned.tolist())
        weights_np = np.array(weights.tolist())

        # All above threshold (0.1 >= 0.05 * 0.6 = 0.03), so all preserved
        np.testing.assert_allclose(pruned_np, weights_np, atol=1e-6)

    def test_zeros_pruned_values(self):
        weights = mx.array([[0.9, 0.01]])
        pruned, mask = prune_experts(weights, threshold=0.05, min_experts=1)

        pruned_np = np.array(pruned.tolist())
        # 0.01 < 0.05 * 0.9 = 0.045
        assert pruned_np[0, 1] == 0.0

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            prune_experts(mx.array([0.5, 0.3, 0.2]), threshold=0.05)

    def test_threshold_zero_keeps_all(self):
        weights = mx.array([[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]])
        pruned, mask = prune_experts(weights, threshold=0.0, min_experts=1)
        pruned_np = np.array(pruned.tolist())
        # threshold=0 means cutoff=0, so all w>0 kept
        assert all(pruned_np[0] > 0)

    def test_uniform_weights(self):
        weights = mx.array([[0.25, 0.25, 0.25, 0.25]])
        pruned, mask = prune_experts(weights, threshold=0.05, min_experts=1)
        pruned_np = np.array(pruned.tolist())
        # All equal to top-1, so all >= 0.05 * 0.25 = 0.0125
        np.testing.assert_allclose(pruned_np, [[0.25, 0.25, 0.25, 0.25]], atol=1e-6)


# -- install_expert_pruning tests --


class TestInstallExpertPruning:
    def test_returns_pruner(self):
        class FakeModel:
            pass

        pruner = install_expert_pruning(FakeModel())
        assert isinstance(pruner, ExpertPruner)

    def test_noop_when_no_layers(self):
        class FakeModel:
            pass

        pruner = install_expert_pruning(FakeModel())
        stats = pruner.get_stats()
        assert stats["decisions"] == 0

    def test_noop_when_no_gate(self):
        class FakeMLP:
            pass

        class FakeLayer:
            def __init__(self):
                self.mlp = FakeMLP()

        class FakeModel:
            class model:
                layers = [FakeLayer()]

        pruner = install_expert_pruning(FakeModel())
        stats = pruner.get_stats()
        assert stats["decisions"] == 0

    def test_custom_config(self):
        class FakeModel:
            pass

        cfg = ExpertPruningConfig(gate_threshold=0.1, warmup_tokens=50)
        pruner = install_expert_pruning(FakeModel(), config=cfg)
        assert pruner.config.gate_threshold == 0.1
        assert pruner.config.warmup_tokens == 50

    def test_finds_layers_via_model_model(self):
        """Verify it walks model.model.layers path."""

        class FakeLayer:
            pass

        class FakeModel:
            class model:
                layers = [FakeLayer(), FakeLayer()]

        pruner = install_expert_pruning(FakeModel())
        assert isinstance(pruner, ExpertPruner)

    def test_finds_layers_via_direct_layers(self):
        """Verify it walks model.layers path."""

        class FakeLayer:
            pass

        class FakeModel:
            layers = [FakeLayer()]

        pruner = install_expert_pruning(FakeModel())
        assert isinstance(pruner, ExpertPruner)

    def test_hooks_gate_not_mlp(self):
        """Verify gate is replaced with wrapper and mlp.__call__ is NOT wrapped."""

        class FakeGate:
            def __call__(self, x):
                return x

        class FakeMLP:
            def __init__(self):
                self.gate = FakeGate()

            def __call__(self, x):
                return x

        class FakeLayer:
            def __init__(self):
                self.mlp = FakeMLP()

        layer = FakeLayer()
        original_gate = layer.mlp.gate
        original_mlp_call = type(layer.mlp).__call__

        class FakeModel:
            class model:
                layers = [layer]

        pruner = install_expert_pruning(FakeModel())

        # gate should be replaced with a _GateWrapper
        assert layer.mlp.gate is not original_gate
        from mlx_flash_compress.expert_pruning import _GateWrapper

        assert isinstance(layer.mlp.gate, _GateWrapper)

        # mlp.__call__ should NOT be replaced
        assert type(layer.mlp).__call__ is original_mlp_call

        # Uninstall should restore original gate
        pruner.uninstall()
        assert layer.mlp.gate is original_gate

    def test_gate_hook_modifies_logits(self):
        """Verify the gate hook modifies logits to prune low-weight experts."""

        class FakeGate:
            def __call__(self, x):
                # Return logits where expert 0 dominates, experts 2-3 are weak
                return mx.array([[5.0, 2.0, -5.0, -6.0]])

        class FakeMLP:
            def __init__(self):
                self.gate = FakeGate()

        class FakeLayer:
            def __init__(self):
                self.mlp = FakeMLP()

        layer = FakeLayer()

        class FakeModel:
            class model:
                layers = [layer]

        cfg = ExpertPruningConfig(warmup_tokens=0, gate_threshold=0.05, adaptive=False)
        pruner = install_expert_pruning(FakeModel(), config=cfg)

        # Call the hooked gate
        x = mx.array([[1.0]])
        modified_logits = layer.mlp.gate(x)

        # Experts below threshold should have -inf logits
        logits_np = np.array(modified_logits.tolist())
        # Expert 0 (dominant) should be finite
        assert np.isfinite(logits_np[0, 0])
        # At least one expert should have been pruned (set to -inf)
        assert np.any(np.isinf(logits_np))

        pruner.uninstall()


# -- Integration/edge case tests --


class TestPruningEdgeCases:
    def test_single_batch_single_expert(self):
        weights = mx.array([[1.0]])
        pruned, mask = prune_experts(weights, threshold=0.05, min_experts=1)
        assert np.array(pruned.tolist())[0, 0] == 1.0

    def test_large_batch(self):
        rng = np.random.default_rng(42)
        data = rng.dirichlet(np.ones(8), size=64).astype(np.float32)
        weights = mx.array(data)
        pruned, mask = prune_experts(weights, threshold=0.05, min_experts=1)
        assert pruned.shape == (64, 8)

        # Every sample should have at least 1 expert
        mask_np = np.array(mask.tolist())
        assert all(mask_np[i].sum() >= 1 for i in range(64))

    def test_pruner_thread_safety(self):
        """Basic check that record_decision uses a lock."""
        pruner = ExpertPruner(ExpertPruningConfig(warmup_tokens=0, adaptive=False))

        import threading

        def record_many():
            for _ in range(100):
                pruner.record_decision(1, 4)

        threads = [threading.Thread(target=record_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = pruner.get_stats()
        assert stats["decisions"] == 400
        assert stats["total_pruned"] == 400
