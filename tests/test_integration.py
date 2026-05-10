"""Integration tests verifying that mlx-flash-compress modules work together.

These tests verify cross-module interactions without requiring a real MLX
model or Metal GPU. MLX imports are deferred to inside test methods.
"""

import pytest

# ===========================================================================
# Helper: check if mlx is available
# ===========================================================================


def _has_mlx():
    try:
        import mlx.core

        return True
    except ImportError:
        return False


# ===========================================================================
# 1. Expert pruning + shared expert pinning
# ===========================================================================


class TestExpertPruningWithSharedPinning:
    """Integration: ExpertPruner skips low-weight experts, but
    SharedExpertPinner protects shared experts from eviction.
    These two systems should compose correctly."""

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_pruner_respects_pinned_experts(self):
        """Pruner decides to skip experts, but pinner overrides for shared ones."""
        from mlx_flash_compress.expert_pruning import ExpertPruner, ExpertPruningConfig
        from mlx_flash_compress.shared_expert_pinning import SharedExpertPinner

        # Set up pruner past warmup
        config = ExpertPruningConfig(gate_threshold=0.1, warmup_tokens=0)
        pruner = ExpertPruner(config)

        # Shared expert at layer 0, expert_id 0
        pinner = SharedExpertPinner({0: [0]})

        # Gate weights: expert 0 is top-1, expert 1 borderline, expert 2-3 prunable
        gate_weights = [0.6, 0.05, 0.02, 0.01]
        mask = pruner.should_compute(gate_weights)

        # Pruner says skip experts 2, 3 (below 10% of top-1=0.6, threshold=0.06)
        # Expert 1: 0.05 < 0.06 -> pruned
        assert mask[0] is True, "Top expert must always be kept"

        # Now verify pinner protects expert 0 from eviction
        assert pinner.is_pinned(0, 0) is True
        assert pinner.should_evict(0, 0) is False

        # Non-shared experts can be evicted
        assert pinner.should_evict(0, 1) is True
        assert pinner.should_evict(0, 2) is True

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_pinner_filters_pruned_candidates(self):
        """Combining pruner mask with pinner filter for eviction candidates."""
        from mlx_flash_compress.expert_pruning import ExpertPruner, ExpertPruningConfig
        from mlx_flash_compress.shared_expert_pinning import SharedExpertPinner

        # Shared experts: layer 0 has experts [0, 1], layer 1 has expert [0]
        pinner = SharedExpertPinner({0: [0, 1], 1: [0]})

        # Eviction candidates from a cache manager
        candidates = [
            (0, 0),  # pinned
            (0, 1),  # pinned
            (0, 2),  # not pinned
            (0, 3),  # not pinned
            (1, 0),  # pinned
            (1, 1),  # not pinned
        ]

        filtered = pinner.filter_eviction_candidates(candidates)
        assert (0, 0) not in filtered
        assert (0, 1) not in filtered
        assert (1, 0) not in filtered
        assert (0, 2) in filtered
        assert (0, 3) in filtered
        assert (1, 1) in filtered
        assert len(filtered) == 3

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_detector_config_feeds_pinner(self):
        """SharedExpertDetector.detect_from_config output feeds into SharedExpertPinner."""
        from mlx_flash_compress.shared_expert_pinning import (
            SharedExpertDetector,
            SharedExpertPinner,
        )

        detector = SharedExpertDetector()

        # Simulate DeepSeek-V2 config
        model_config = {
            "num_shared_experts": 2,
            "num_hidden_layers": 8,
            "first_k_dense_replace": 2,  # First 2 layers are dense
            "moe_layer_freq": 1,
        }

        shared = detector.detect_from_config(model_config)

        # Layers 0-1 are dense (skipped), layers 2-7 should have shared experts
        assert 0 not in shared
        assert 1 not in shared
        for layer_idx in range(2, 8):
            assert layer_idx in shared
            assert shared[layer_idx] == [0, 1]

        # Feed into pinner
        pinner = SharedExpertPinner(shared)
        assert pinner.get_pinned_count() == 12  # 6 layers * 2 experts

        # Verify pinning works
        assert pinner.is_pinned(2, 0) is True
        assert pinner.is_pinned(2, 1) is True
        assert pinner.is_pinned(2, 2) is False
        assert pinner.is_pinned(0, 0) is False  # Dense layer, not pinned

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_pruner_stats_track_decisions(self):
        """Pruner stats work after multiple decisions alongside pinner."""
        from mlx_flash_compress.expert_pruning import ExpertPruner, ExpertPruningConfig
        from mlx_flash_compress.shared_expert_pinning import SharedExpertPinner

        config = ExpertPruningConfig(gate_threshold=0.1, warmup_tokens=0, adaptive=False)
        pruner = ExpertPruner(config)
        pinner = SharedExpertPinner({0: [0]})

        # Simulate 5 tokens through the system
        for _ in range(5):
            weights = [0.7, 0.2, 0.08, 0.02]
            mask = pruner.should_compute(weights)
            pruned = mask.count(False)
            pruner.record_decision(pruned, len(weights))

        stats = pruner.get_stats()
        assert stats["decisions"] == 5
        assert stats["tokens_seen"] == 5

        # Pinner stats
        for _ in range(10):
            pinner.should_evict(0, 0)
            pinner.should_evict(0, 1)

        pinner_stats = pinner.get_stats()
        assert pinner_stats["eviction_checks"] == 20
        assert pinner_stats["eviction_blocks"] == 10  # 10 checks on pinned expert 0
        assert pinner_stats["block_rate"] == 0.5


# ===========================================================================
# 2. StreamingLLM + Quantized KV cache
# ===========================================================================


class TestStreamingLLMWithQuantizedKV:
    """Integration: StreamingLLM manages which tokens to keep in cache,
    QuantizedKVConfig controls how those tokens are stored. Both are
    KV cache strategies that should be independently configurable."""

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_configs_are_independent(self):
        """Both configs can be instantiated without conflict."""
        from mlx_flash_compress.quantized_kv_cache import QuantizedKVConfig
        from mlx_flash_compress.streaming_llm import StreamingLLMConfig

        streaming_cfg = StreamingLLMConfig(
            num_sink_tokens=4,
            window_size=512,
            eviction_batch=128,
        )
        quant_cfg = QuantizedKVConfig(
            key_bits=4,
            value_bits=4,
            group_size=64,
            calibration_tokens=16,
        )

        # Both should have their own independent settings
        assert streaming_cfg.num_sink_tokens == 4
        assert streaming_cfg.window_size == 512
        assert quant_cfg.key_bits == 4
        assert quant_cfg.value_bits == 4

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_streaming_config_defaults(self):
        """StreamingLLMConfig has sensible defaults."""
        from mlx_flash_compress.streaming_llm import StreamingLLMConfig

        cfg = StreamingLLMConfig()
        assert cfg.num_sink_tokens == 4
        assert cfg.window_size == 1024
        assert cfg.eviction_batch == 256

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_quantized_kv_config_defaults(self):
        """QuantizedKVConfig has sensible defaults."""
        from mlx_flash_compress.quantized_kv_cache import QuantizedKVConfig

        cfg = QuantizedKVConfig()
        assert cfg.key_bits == 4
        assert cfg.value_bits == 4
        assert cfg.group_size == 64
        assert cfg.calibration_tokens == 32

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_streaming_window_fits_quantized_calibration(self):
        """Window size should be larger than calibration tokens for the two
        strategies to make sense together."""
        from mlx_flash_compress.quantized_kv_cache import QuantizedKVConfig
        from mlx_flash_compress.streaming_llm import StreamingLLMConfig

        streaming_cfg = StreamingLLMConfig(window_size=1024)
        quant_cfg = QuantizedKVConfig(calibration_tokens=32)

        # Sanity: window should be much larger than calibration period
        assert streaming_cfg.window_size > quant_cfg.calibration_tokens, (
            "StreamingLLM window should be larger than quantization calibration period"
        )

        # Total cache size = sink + window
        total_capacity = streaming_cfg.num_sink_tokens + streaming_cfg.window_size
        assert total_capacity > quant_cfg.calibration_tokens


# ===========================================================================
# 3. Layer quantization config + estimate functions
# ===========================================================================


class TestLayerQuantizationConfig:
    """Integration: LayerQuantConfig and LayerSensitivityProfile work
    together to produce precision maps."""

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_default_precision_map_heuristic(self):
        """Default precision map assigns Q8 to first/last layers, Q4 to middle."""
        from mlx_flash_compress.layer_quantization import (
            LayerQuantConfig,
            LayerSensitivityProfile,
        )

        config = LayerQuantConfig(
            default_bits=4,
            sensitive_bits=8,
            num_sensitive_start=2,
            num_sensitive_end=2,
        )
        num_layers = 32

        precision_map = LayerSensitivityProfile.default_precision_map(num_layers, config)

        assert len(precision_map) == 32

        # First 2 layers should be Q8
        assert precision_map[0] == 8
        assert precision_map[1] == 8

        # Last 2 layers should be Q8
        assert precision_map[30] == 8
        assert precision_map[31] == 8

        # Middle layers should be Q4
        for i in range(2, 30):
            assert precision_map[i] == 4, f"Layer {i} should be Q4"

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_custom_sensitive_layers_with_scores(self):
        """When sensitivity_scores exist and sensitive_layers is set,
        get_precision_map uses the explicit sensitive_layers list."""
        from mlx_flash_compress.layer_quantization import (
            LayerQuantConfig,
            LayerSensitivityProfile,
        )

        config = LayerQuantConfig(
            default_bits=4,
            sensitive_bits=8,
            sensitive_layers=[0, 5, 10, 15],
        )

        profile = LayerSensitivityProfile(num_layers=16)
        # Provide fake sensitivity scores so get_precision_map doesn't
        # fall back to default_precision_map (which uses num_sensitive_start/end)
        profile.sensitivity_scores = [0.1] * 16

        precision_map = profile.get_precision_map(config)

        # With sensitive_layers explicitly set, those layers get Q8
        for i in range(16):
            if i in config.sensitive_layers:
                assert precision_map[i] == 8, f"Layer {i} should be Q8 (sensitive)"
            else:
                assert precision_map[i] == 4, f"Layer {i} should be Q4 (default)"

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_default_precision_map_uses_start_end(self):
        """default_precision_map uses num_sensitive_start/end, not sensitive_layers."""
        from mlx_flash_compress.layer_quantization import (
            LayerQuantConfig,
            LayerSensitivityProfile,
        )

        config = LayerQuantConfig(
            default_bits=4,
            sensitive_bits=8,
            num_sensitive_start=1,
            num_sensitive_end=1,
        )
        precision_map = LayerSensitivityProfile.default_precision_map(10, config)

        # First layer: Q8, last layer: Q8, middle: Q4
        assert precision_map[0] == 8
        for i in range(1, 9):
            assert precision_map[i] == 4, f"Layer {i} should be Q4"
        assert precision_map[9] == 8

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_config_validation(self):
        """LayerQuantConfig accepts valid configurations."""
        from mlx_flash_compress.layer_quantization import LayerQuantConfig

        cfg = LayerQuantConfig(
            default_bits=3,
            sensitive_bits=8,
            num_sensitive_start=3,
            num_sensitive_end=3,
            group_size=128,
        )
        assert cfg.default_bits == 3
        assert cfg.group_size == 128

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_precision_map_small_model(self):
        """Edge case: model with fewer layers than sensitive count."""
        from mlx_flash_compress.layer_quantization import (
            LayerQuantConfig,
            LayerSensitivityProfile,
        )

        config = LayerQuantConfig(
            default_bits=4,
            sensitive_bits=8,
            num_sensitive_start=3,
            num_sensitive_end=3,
        )

        # Only 4 layers total, but 3+3=6 sensitive requested
        precision_map = LayerSensitivityProfile.default_precision_map(4, config)
        assert len(precision_map) == 4
        # All layers should be Q8 since they overlap
        for i in range(4):
            assert precision_map[i] == 8


# ===========================================================================
# 4. KV compression config validation
# ===========================================================================


class TestKVCompressionConfig:
    """Integration: KVCompressionConfig works with different scoring strategies."""

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_h2o_config(self):
        """H2O scoring strategy config."""
        from mlx_flash_compress.kv_compression import KVCompressionConfig

        cfg = KVCompressionConfig(
            budget_ratio=0.2,
            sink_tokens=4,
            recent_window=128,
            scoring="h2o",
        )
        assert cfg.scoring == "h2o"
        assert cfg.budget_ratio == 0.2

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_scissorhands_config(self):
        """ScissorHands scoring strategy config."""
        from mlx_flash_compress.kv_compression import KVCompressionConfig

        cfg = KVCompressionConfig(
            budget_ratio=0.3,
            scoring="scissorhands",
        )
        assert cfg.scoring == "scissorhands"
        assert cfg.budget_ratio == 0.3

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_kv_compression_defaults(self):
        """Default config values are reasonable."""
        from mlx_flash_compress.kv_compression import KVCompressionConfig

        cfg = KVCompressionConfig()
        assert 0.0 < cfg.budget_ratio <= 1.0
        assert cfg.sink_tokens >= 0
        assert cfg.recent_window > 0
        assert cfg.scoring in ("h2o", "scissorhands")

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_kv_compression_with_quantize_evicted(self):
        """Config with quantize_evicted enabled."""
        from mlx_flash_compress.kv_compression import KVCompressionConfig

        cfg = KVCompressionConfig(quantize_evicted=True)
        assert cfg.quantize_evicted is True

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_attention_tracker_init(self):
        """AttentionScoreTracker initializes for both scoring modes."""
        from mlx_flash_compress.kv_compression import AttentionScoreTracker

        for scoring in ("h2o", "scissorhands"):
            tracker = AttentionScoreTracker(num_layers=32, max_seq_len=4096, scoring=scoring)
            assert tracker.scoring == scoring
            assert tracker.num_layers == 32


# ===========================================================================
# 5. Multiple module configs can coexist
# ===========================================================================


class TestConfigCoexistence:
    """Verify that multiple module configurations can be instantiated
    simultaneously without interference."""

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_all_configs_coexist(self):
        """Create one config from each module and verify independence."""
        from mlx_flash_compress.expert_pruning import ExpertPruningConfig
        from mlx_flash_compress.kv_compression import KVCompressionConfig
        from mlx_flash_compress.layer_quantization import LayerQuantConfig
        from mlx_flash_compress.quantized_kv_cache import QuantizedKVConfig
        from mlx_flash_compress.streaming_llm import StreamingLLMConfig

        configs = {
            "pruning": ExpertPruningConfig(gate_threshold=0.1),
            "streaming": StreamingLLMConfig(window_size=2048),
            "quant_kv": QuantizedKVConfig(key_bits=8),
            "kv_compress": KVCompressionConfig(budget_ratio=0.3),
            "layer_quant": LayerQuantConfig(default_bits=3),
        }

        # All should be distinct types
        types_seen = set()
        for name, cfg in configs.items():
            cfg_type = type(cfg)
            assert cfg_type not in types_seen, f"Duplicate config type for {name}"
            types_seen.add(cfg_type)

        # Values should be independent
        assert configs["pruning"].gate_threshold == 0.1
        assert configs["streaming"].window_size == 2048
        assert configs["quant_kv"].key_bits == 8
        assert configs["kv_compress"].budget_ratio == 0.3
        assert configs["layer_quant"].default_bits == 3

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_pruner_and_pinner_with_profile(self):
        """ExpertPruner + SharedExpertPinner + DFlash profile selection
        can all coexist in a single inference pipeline configuration."""
        from mlx_flash_compress.dflash_profile import (
            PROFILES,
            ModelProfile,
            select_profile,
        )
        from mlx_flash_compress.expert_pruning import ExpertPruner, ExpertPruningConfig
        from mlx_flash_compress.shared_expert_pinning import SharedExpertPinner

        # Configure expert pruning
        pruner = ExpertPruner(ExpertPruningConfig(gate_threshold=0.05))

        # Configure shared expert pinning
        pinner = SharedExpertPinner({0: [0], 1: [0]})

        # Select DFlash profile based on model characteristics
        model_profile = ModelProfile(
            category="large_moe",
            total_params_b=47.0,
            active_params_b=14.0,
            num_layers=32,
            num_ssm_layers=0,
            num_attn_layers=32,
            is_moe=True,
            is_quantized=True,
            quant_bits=4,
            ar_tok_s=18.0,
            has_ssm=False,
        )
        dflash_profile = select_profile(model_profile, priority="auto")

        # All three should work independently
        assert pruner.config.gate_threshold == 0.05
        assert pinner.is_pinned(0, 0) is True
        assert dflash_profile.name in PROFILES
        assert isinstance(dflash_profile.description, str)

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_streaming_and_kv_compression_configs_differ(self):
        """StreamingLLM and KV compression are different eviction strategies
        that should not share state."""
        from mlx_flash_compress.kv_compression import KVCompressionConfig
        from mlx_flash_compress.streaming_llm import StreamingLLMConfig

        streaming = StreamingLLMConfig(num_sink_tokens=4, window_size=512)
        kv_compress = KVCompressionConfig(sink_tokens=4, recent_window=128)

        # Both have "sink" concept but are separate configs
        assert streaming.num_sink_tokens == kv_compress.sink_tokens == 4
        # But window sizes differ
        assert streaming.window_size != kv_compress.recent_window

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_multiple_pruners_independent(self):
        """Multiple ExpertPruner instances don't share state."""
        from mlx_flash_compress.expert_pruning import ExpertPruner, ExpertPruningConfig

        pruner_a = ExpertPruner(ExpertPruningConfig(gate_threshold=0.05, warmup_tokens=0))
        pruner_b = ExpertPruner(ExpertPruningConfig(gate_threshold=0.2, warmup_tokens=0))

        weights = [0.7, 0.1, 0.05, 0.01]

        mask_a = pruner_a.should_compute(weights)
        mask_b = pruner_b.should_compute(weights)

        # Pruner A (threshold=0.05): 0.05*0.7=0.035, so 0.1>0.035, 0.05>0.035, 0.01<0.035
        # mask_a = [True, True, True, False]
        assert mask_a[0] is True  # top-1
        assert mask_a[1] is True  # 0.1 > 0.035
        assert mask_a[2] is True  # 0.05 > 0.035

        # Pruner B (threshold=0.2): 0.2*0.7=0.14, so 0.1<0.14, 0.05<0.14, 0.01<0.14
        # mask_b = [True, False, False, False]
        assert mask_b[0] is True  # top-1
        assert mask_b[1] is False  # 0.1 < 0.14

        # Record decision on A doesn't affect B
        pruner_a.record_decision(1, 4)
        assert pruner_a.get_stats()["decisions"] == 1
        assert pruner_b.get_stats()["decisions"] == 0


# ===========================================================================
# 6. Cross-module import consistency
# ===========================================================================


class TestCrossModuleImportConsistency:
    """Verify that objects imported from different paths are the same."""

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_expert_pruning_config_same_object(self):
        """ExpertPruningConfig from __init__ and direct import are identical."""
        import mlx_flash_compress
        from mlx_flash_compress.expert_pruning import ExpertPruningConfig as Direct

        lazy = mlx_flash_compress.ExpertPruningConfig
        assert lazy is Direct

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_shared_expert_classes_same_object(self):
        """SharedExpertDetector/Pinner from __init__ and direct import match."""
        import mlx_flash_compress
        from mlx_flash_compress.shared_expert_pinning import SharedExpertDetector as DirectD
        from mlx_flash_compress.shared_expert_pinning import SharedExpertPinner as DirectP

        assert mlx_flash_compress.SharedExpertDetector is DirectD
        assert mlx_flash_compress.SharedExpertPinner is DirectP

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_streaming_llm_classes_same_object(self):
        import mlx_flash_compress
        from mlx_flash_compress.streaming_llm import StreamingLLMConfig as Direct

        assert mlx_flash_compress.StreamingLLMConfig is Direct

    @pytest.mark.skipif(not _has_mlx(), reason="mlx not available")
    def test_kv_compression_classes_same_object(self):
        import mlx_flash_compress
        from mlx_flash_compress.kv_compression import KVCompressionConfig as Direct

        assert mlx_flash_compress.KVCompressionConfig is Direct
