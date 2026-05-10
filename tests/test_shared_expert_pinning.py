"""Tests for shared expert pinning."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

try:
    from mlx_flash_compress.shared_expert_pinning import (
        SharedExpertDetector,
        SharedExpertPinner,
        _extract_model_config,
        detect_and_pin_shared_experts,
    )

    HAS_MODULE = True
except (ImportError, ModuleNotFoundError):
    HAS_MODULE = False

pytestmark = pytest.mark.skipif(not HAS_MODULE, reason="shared_expert_pinning not available")


# -- Helper: fake RoutingEvent matching router_hook.py's dataclass --


@dataclass
class FakeRoutingEvent:
    token_idx: int
    layer_idx: int
    expert_ids: list
    expert_weights: list


# -- SharedExpertDetector config-based detection --


class TestDetectFromConfig:
    def test_deepseek_v2_config(self):
        """DeepSeek-V2 exposes num_shared_experts in config."""
        config = {
            "num_shared_experts": 2,
            "num_hidden_layers": 4,
            "first_k_dense_replace": 0,
            "moe_layer_freq": 1,
        }
        detector = SharedExpertDetector()
        result = detector.detect_from_config(config)

        assert len(result) == 4
        for layer_idx in range(4):
            assert result[layer_idx] == [0, 1]

    def test_deepseek_v3_with_dense_layers(self):
        """DeepSeek-V3 has first K layers as dense (no MoE)."""
        config = {
            "num_shared_experts": 1,
            "num_hidden_layers": 6,
            "first_k_dense_replace": 2,
            "moe_layer_freq": 1,
        }
        detector = SharedExpertDetector()
        result = detector.detect_from_config(config)

        # Layers 0, 1 are dense — no shared experts there
        assert 0 not in result
        assert 1 not in result
        # Layers 2-5 are MoE
        for layer_idx in [2, 3, 4, 5]:
            assert result[layer_idx] == [0]

    def test_moe_layer_frequency(self):
        """Some models have MoE only on every Nth layer."""
        config = {
            "num_shared_experts": 2,
            "num_hidden_layers": 8,
            "first_k_dense_replace": 0,
            "moe_layer_freq": 2,  # MoE on even layers only
        }
        detector = SharedExpertDetector()
        result = detector.detect_from_config(config)

        # MoE layers: 0, 2, 4, 6
        assert 0 in result
        assert 1 not in result
        assert 2 in result
        assert 3 not in result
        assert 4 in result

    def test_explicit_shared_expert_ids(self):
        """Config with explicit shared_expert_ids list."""
        config = {
            "shared_expert_ids": [5, 10, 15],
            "num_hidden_layers": 3,
        }
        detector = SharedExpertDetector()
        result = detector.detect_from_config(config)

        for layer_idx in range(3):
            assert result[layer_idx] == [5, 10, 15]

    def test_n_shared_experts_variant(self):
        """Alternative config key: n_shared_experts."""
        config = {
            "n_shared_experts": 3,
            "num_hidden_layers": 2,
        }
        detector = SharedExpertDetector()
        result = detector.detect_from_config(config)

        for layer_idx in range(2):
            assert result[layer_idx] == [0, 1, 2]

    def test_no_shared_experts_in_config(self):
        """Config without shared expert fields returns empty."""
        config = {
            "num_hidden_layers": 10,
            "hidden_size": 4096,
            "num_experts": 16,
        }
        detector = SharedExpertDetector()
        result = detector.detect_from_config(config)

        assert result == {}

    def test_zero_shared_experts(self):
        """num_shared_experts=0 means no shared experts."""
        config = {
            "num_shared_experts": 0,
            "num_hidden_layers": 4,
        }
        detector = SharedExpertDetector()
        result = detector.detect_from_config(config)

        assert result == {}

    def test_qwen3_moe_config(self):
        """Qwen3-MoE uses num_shared_experts."""
        config = {
            "num_shared_experts": 4,
            "num_hidden_layers": 24,
            "num_layers": 24,
        }
        detector = SharedExpertDetector()
        result = detector.detect_from_config(config)

        assert len(result) == 24
        for layer_idx in range(24):
            assert result[layer_idx] == [0, 1, 2, 3]

    def test_no_num_layers_uses_global(self):
        """If no layer count in config, store as global (layer -1)."""
        config = {
            "num_shared_experts": 2,
        }
        detector = SharedExpertDetector()
        result = detector.detect_from_config(config)

        assert -1 in result
        assert result[-1] == [0, 1]

    def test_get_shared_experts_after_detect(self):
        config = {
            "num_shared_experts": 2,
            "num_hidden_layers": 3,
        }
        detector = SharedExpertDetector()
        detector.detect_from_config(config)
        shared = detector.get_shared_experts()

        assert len(shared) == 3
        for layer_idx in range(3):
            assert shared[layer_idx] == [0, 1]

    def test_get_shared_experts_before_detect(self):
        detector = SharedExpertDetector()
        shared = detector.get_shared_experts()
        assert shared == {}


# -- SharedExpertDetector observation-based detection --


class TestDetectFromObservation:
    def _make_routing_log(self, num_tokens, num_layers, always_on_experts, other_experts):
        """Create synthetic routing log where always_on_experts appear every token."""
        log = []
        for token in range(num_tokens):
            for layer in range(num_layers):
                # Always-on experts
                expert_ids = list(always_on_experts)
                # Add some other experts sometimes
                if token % 3 == 0 and other_experts:
                    expert_ids.extend(other_experts[:2])
                log.append(
                    FakeRoutingEvent(
                        token_idx=token,
                        layer_idx=layer,
                        expert_ids=expert_ids,
                        expert_weights=[1.0 / len(expert_ids)] * len(expert_ids),
                    )
                )
        return log

    def test_detects_always_active_experts(self):
        log = self._make_routing_log(num_tokens=100, num_layers=2, always_on_experts=[0, 1], other_experts=[5, 6, 7])
        detector = SharedExpertDetector()
        result = detector.detect_from_observation(log, threshold=0.95)

        # Experts 0 and 1 appear in every token — should be detected
        for layer_idx in range(2):
            assert 0 in result[layer_idx]
            assert 1 in result[layer_idx]
            # Experts 5, 6, 7 appear only ~33% of the time — not shared
            assert 5 not in result[layer_idx]

    def test_high_threshold_fewer_detections(self):
        log = self._make_routing_log(num_tokens=100, num_layers=1, always_on_experts=[0], other_experts=[5, 6])
        detector = SharedExpertDetector()

        result_low = detector.detect_from_observation(log, threshold=0.5)
        result_high = detector.detect_from_observation(log, threshold=0.99)

        # Expert 0 has 100% activation rate, should pass both thresholds
        assert 0 in result_low.get(0, [])
        assert 0 in result_high.get(0, [])

        # Expert 5 has ~33% rate, should only pass low threshold
        if 5 in result_low.get(0, []):
            assert 5 not in result_high.get(0, [])

    def test_empty_routing_log(self):
        detector = SharedExpertDetector()
        result = detector.detect_from_observation([], threshold=0.95)
        assert result == {}

    def test_no_shared_experts_found(self):
        """When no expert is active >95%, return empty."""
        log = []
        for token in range(100):
            # Each token activates a different expert
            log.append(
                FakeRoutingEvent(
                    token_idx=token,
                    layer_idx=0,
                    expert_ids=[token % 10],
                    expert_weights=[1.0],
                )
            )
        detector = SharedExpertDetector()
        result = detector.detect_from_observation(log, threshold=0.95)
        # Each expert appears only ~10% of the time
        assert 0 not in result

    def test_single_token(self):
        log = [FakeRoutingEvent(token_idx=0, layer_idx=0, expert_ids=[3, 7], expert_weights=[0.6, 0.4])]
        detector = SharedExpertDetector()
        result = detector.detect_from_observation(log, threshold=0.95)
        # With 1 token, activation rate is 1.0 for both
        assert 3 in result.get(0, [])
        assert 7 in result.get(0, [])


# -- SharedExpertPinner tests --


class TestSharedExpertPinner:
    def test_is_pinned_basic(self):
        pinner = SharedExpertPinner({0: [0, 1], 1: [0]})
        assert pinner.is_pinned(0, 0) is True
        assert pinner.is_pinned(0, 1) is True
        assert pinner.is_pinned(0, 2) is False
        assert pinner.is_pinned(1, 0) is True
        assert pinner.is_pinned(1, 1) is False

    def test_is_pinned_empty(self):
        pinner = SharedExpertPinner({})
        assert pinner.is_pinned(0, 0) is False
        assert pinner.is_pinned(5, 10) is False

    def test_should_evict(self):
        pinner = SharedExpertPinner({0: [0, 1]})
        # Pinned: cannot evict
        assert pinner.should_evict(0, 0) is False
        assert pinner.should_evict(0, 1) is False
        # Not pinned: can evict
        assert pinner.should_evict(0, 2) is True
        assert pinner.should_evict(1, 0) is True

    def test_global_pinning(self):
        """Layer -1 means all layers."""
        pinner = SharedExpertPinner({-1: [0, 1]})
        assert pinner.is_pinned(0, 0) is True
        assert pinner.is_pinned(5, 0) is True
        assert pinner.is_pinned(99, 1) is True
        assert pinner.is_pinned(0, 2) is False

    def test_get_pinned_count(self):
        pinner = SharedExpertPinner({0: [0, 1], 1: [0, 2, 3]})
        assert pinner.get_pinned_count() == 5

    def test_get_pinned_count_empty(self):
        pinner = SharedExpertPinner({})
        assert pinner.get_pinned_count() == 0

    def test_filter_eviction_candidates(self):
        pinner = SharedExpertPinner({0: [0, 1]})
        candidates = [(0, 0), (0, 1), (0, 2), (0, 3)]
        filtered = pinner.filter_eviction_candidates(candidates)
        assert filtered == [(0, 2), (0, 3)]

    def test_filter_eviction_candidates_empty(self):
        pinner = SharedExpertPinner({0: [0, 1, 2, 3]})
        candidates = [(0, 0), (0, 1), (0, 2), (0, 3)]
        filtered = pinner.filter_eviction_candidates(candidates)
        assert filtered == []

    def test_stats_output(self):
        pinner = SharedExpertPinner({0: [0, 1]})
        # Trigger some checks
        pinner.should_evict(0, 0)  # blocked
        pinner.should_evict(0, 1)  # blocked
        pinner.should_evict(0, 5)  # allowed

        stats = pinner.get_stats()
        assert stats["pinned_count"] == 2
        assert stats["pinned_layers"] == 1
        assert stats["eviction_checks"] == 3
        assert stats["eviction_blocks"] == 2
        assert stats["block_rate"] == round(2 / 3, 4)

    def test_stats_keys(self):
        pinner = SharedExpertPinner({})
        stats = pinner.get_stats()
        expected_keys = {
            "pinned_count",
            "pinned_layers",
            "eviction_checks",
            "eviction_blocks",
            "block_rate",
            "pinned_experts",
        }
        assert set(stats.keys()) == expected_keys

    def test_get_pinned_experts(self):
        shared = {0: [0, 1], 2: [5]}
        pinner = SharedExpertPinner(shared)
        assert pinner.get_pinned_experts() == shared


# -- detect_and_pin_shared_experts convenience function --


class TestDetectAndPin:
    def test_model_with_config_dict(self):
        model = MagicMock()
        model.config = {
            "num_shared_experts": 2,
            "num_hidden_layers": 4,
        }
        # Ensure model doesn't have 'args' to avoid confusion
        del model.args

        pinner = detect_and_pin_shared_experts(model)
        assert pinner.get_pinned_count() > 0
        assert pinner.is_pinned(0, 0) is True
        assert pinner.is_pinned(0, 1) is True

    def test_model_with_config_object(self):
        class ModelConfig:
            num_shared_experts = 3
            num_hidden_layers = 2

        model = MagicMock()
        model.config = ModelConfig()
        del model.args

        pinner = detect_and_pin_shared_experts(model)
        assert pinner.get_pinned_count() > 0

    def test_model_with_args(self):
        """mlx_lm pattern: model.args instead of model.config."""
        model = MagicMock()
        del model.config
        model.args = {
            "num_shared_experts": 1,
            "num_hidden_layers": 3,
        }

        pinner = detect_and_pin_shared_experts(model)
        assert pinner.get_pinned_count() > 0

    def test_model_without_shared_experts(self):
        model = MagicMock()
        model.config = {"num_hidden_layers": 10, "hidden_size": 4096}
        del model.args

        pinner = detect_and_pin_shared_experts(model)
        assert pinner.get_pinned_count() == 0

    def test_model_with_no_config(self):
        class BareModel:
            pass

        pinner = detect_and_pin_shared_experts(BareModel())
        assert pinner.get_pinned_count() == 0

    def test_with_cache_manager(self):
        """cache_manager argument should not crash (reserved for future)."""
        model = MagicMock()
        model.config = {"num_shared_experts": 2, "num_hidden_layers": 2}
        del model.args
        cache_mgr = MagicMock()

        pinner = detect_and_pin_shared_experts(model, cache_manager=cache_mgr)
        assert pinner.get_pinned_count() > 0


# -- _extract_model_config helper tests --


class TestExtractModelConfig:
    def test_dict_config(self):
        model = MagicMock()
        model.config = {"num_shared_experts": 2}
        del model.args
        cfg = _extract_model_config(model)
        assert cfg is not None
        assert cfg["num_shared_experts"] == 2

    def test_object_config(self):
        class Cfg:
            num_shared_experts = 3
            hidden_size = 4096

        model = MagicMock()
        model.config = Cfg()
        del model.args
        cfg = _extract_model_config(model)
        assert cfg is not None
        assert cfg["num_shared_experts"] == 3

    def test_nested_model_model_config(self):
        inner = MagicMock()
        inner.config = {"num_shared_experts": 1}
        del inner.args
        model = MagicMock()
        model.model = inner
        model.config = {}
        del model.args
        cfg = _extract_model_config(model)
        # Should find it via model.model.config
        assert cfg is not None

    def test_no_config_returns_none(self):
        class BareModel:
            pass

        cfg = _extract_model_config(BareModel())
        assert cfg is None


# -- Integration: pinner with eviction policy --


class TestPinnerEvictionIntegration:
    def test_eviction_policy_respects_pinning(self):
        """Simulate eviction policy checking pinner before evicting."""
        pinner = SharedExpertPinner({0: [0, 1]})

        # Simulated cache with experts [0, 1, 2, 3, 4]
        all_experts = [(0, eid) for eid in range(5)]

        # Eviction candidates: all experts
        evictable = pinner.filter_eviction_candidates(all_experts)

        # Experts 0 and 1 should NOT be evictable
        assert (0, 0) not in evictable
        assert (0, 1) not in evictable
        # Experts 2, 3, 4 should be evictable
        assert (0, 2) in evictable
        assert (0, 3) in evictable
        assert (0, 4) in evictable

    def test_multi_layer_eviction(self):
        """Eviction across multiple layers."""
        pinner = SharedExpertPinner(
            {
                0: [0],
                1: [0, 1],
                2: [0, 1, 2],
            }
        )

        # Layer 0: only expert 0 pinned
        candidates_l0 = [(0, eid) for eid in range(5)]
        evictable_l0 = pinner.filter_eviction_candidates(candidates_l0)
        assert len(evictable_l0) == 4

        # Layer 2: experts 0, 1, 2 pinned
        candidates_l2 = [(2, eid) for eid in range(5)]
        evictable_l2 = pinner.filter_eviction_candidates(candidates_l2)
        assert len(evictable_l2) == 2

    def test_all_pinned_no_eviction_possible(self):
        """When all experts are pinned, no eviction is possible."""
        pinner = SharedExpertPinner({0: [0, 1, 2]})
        candidates = [(0, 0), (0, 1), (0, 2)]
        evictable = pinner.filter_eviction_candidates(candidates)
        assert evictable == []
