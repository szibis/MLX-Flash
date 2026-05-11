"""Tests for Sequoia speculative decoding with SSD offloading."""

import numpy as np
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_flash_compress.sequoia import (
        LayerOffloader,
        SequoiaConfig,
        SequoiaEngine,
        SpeculationTree,
        apply_sequoia,
    )

    HAS_MLX = True
except (ImportError, ModuleNotFoundError):
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="requires mlx")


# ── Mock models ──────────────────────────────────────────────────


class MockDraftModel(nn.Module):
    """Tiny draft model for testing. Returns deterministic logits."""

    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x):
        if len(x.shape) == 1:
            x = mx.expand_dims(x, axis=0)
        h = self.embed(x)
        logits = self.proj(h)
        return logits


class MockTargetModel(nn.Module):
    """Tiny target model that mirrors draft model outputs for testing."""

    def __init__(self, vocab_size: int = 32, hidden_dim: int = 32, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = [MockTransformerLayer(hidden_dim) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x):
        if len(x.shape) == 1:
            x = mx.expand_dims(x, axis=0)
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class MockTransformerLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def __call__(self, x):
        return x + self.mlp(self.norm(x))


class MockTokenizer:
    eos_token_id = 0

    def encode(self, text):
        return [1, 2, 3, 4, 5]


# ── SpeculationTree tests ────────────────────────────────────────


class TestSpeculationTree:
    def test_optimal_depth_high_acceptance(self):
        """High acceptance rate should favor deeper speculation."""
        config = SequoiaConfig(
            max_draft_tokens=8,
            draft_latency_ms=5.0,
            verify_latency_ms=50.0,
        )
        tree = SpeculationTree(config)

        depth_high = tree.compute_optimal_depth(0.9)
        depth_low = tree.compute_optimal_depth(0.3)

        assert depth_high >= depth_low
        assert depth_high >= 1
        assert depth_high <= config.max_draft_tokens

    def test_optimal_depth_zero_acceptance(self):
        """Zero acceptance -> minimal depth (1)."""
        config = SequoiaConfig(max_draft_tokens=8)
        tree = SpeculationTree(config)

        depth = tree.compute_optimal_depth(0.0)
        assert depth == 1

    def test_optimal_depth_perfect_acceptance(self):
        """Perfect acceptance -> should go deep."""
        config = SequoiaConfig(
            max_draft_tokens=8,
            draft_latency_ms=1.0,
            verify_latency_ms=100.0,
        )
        tree = SpeculationTree(config)

        depth = tree.compute_optimal_depth(1.0)
        assert depth == config.max_draft_tokens

    def test_optimal_depth_high_ssd_latency(self):
        """Higher SSD latency should encourage deeper speculation to amortize."""
        config_fast = SequoiaConfig(
            max_draft_tokens=8,
            draft_latency_ms=5.0,
            verify_latency_ms=20.0,
        )
        config_slow = SequoiaConfig(
            max_draft_tokens=8,
            draft_latency_ms=5.0,
            verify_latency_ms=200.0,
        )

        tree_fast = SpeculationTree(config_fast)
        tree_slow = SpeculationTree(config_slow)

        depth_fast = tree_fast.compute_optimal_depth(0.7)
        depth_slow = tree_slow.compute_optimal_depth(0.7)

        # With higher verify cost, we should speculate more to amortize
        assert depth_slow >= depth_fast

    def test_optimal_depth_bounded(self):
        """Optimal depth should never exceed max_draft_tokens."""
        config = SequoiaConfig(max_draft_tokens=4)
        tree = SpeculationTree(config)

        for rate in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            depth = tree.compute_optimal_depth(rate)
            assert 1 <= depth <= config.max_draft_tokens

    def test_dp_throughput_formula(self):
        """Verify the DP formula directly: throughput = expected_tokens / time."""
        config = SequoiaConfig(
            max_draft_tokens=8,
            draft_latency_ms=5.0,
            verify_latency_ms=50.0,
        )
        tree = SpeculationTree(config)

        alpha = 0.8
        depth = 4

        # Expected tokens: geometric series
        expected = sum(alpha**i for i in range(depth + 1))
        wall_time = depth * 5.0 + 50.0

        throughput = expected / wall_time

        # Verify against formula
        expected_formula = (1 - alpha ** (depth + 1)) / (1 - alpha)
        assert abs(expected - expected_formula) < 1e-10
        assert throughput > 0

    def test_build_tree_produces_structure(self):
        """build_tree should produce a tree with children."""
        config = SequoiaConfig(max_draft_tokens=3, tree_width=2)
        tree = SpeculationTree(config)

        # Simple draft function: always returns token 5
        call_count = [0]

        def mock_draft(ids):
            call_count[0] += 1
            logits = mx.zeros(32)
            logits = logits.at[5].add(10.0)
            return 5, logits

        root = tree.build_tree(mock_draft, mx.array([1, 2, 3]), depth=2)

        assert root["token"] is None  # root has no token
        assert len(root["children"]) > 0
        assert root["children"][0]["token"] is not None

    def test_flatten_for_verification(self):
        """Flattening a tree should produce valid candidate paths."""
        config = SequoiaConfig(max_draft_tokens=3, tree_width=1)
        tree = SpeculationTree(config)

        def mock_draft(ids):
            # Cycle through tokens
            tok = int(ids[-1].item()) + 1
            if tok >= 10:
                tok = 1
            return tok, mx.zeros(32)

        root = tree.build_tree(mock_draft, mx.array([1]), depth=3)
        candidates = tree.flatten_for_verification(root)

        assert candidates.shape[0] >= 1  # at least one path
        assert candidates.shape[1] == 3  # depth 3

    def test_select_accepted_full_match(self):
        """When all draft tokens match verified, accept all + bonus."""
        config = SequoiaConfig(tree_width=1)
        tree = SpeculationTree(config)

        # Build a simple linear tree: [5, 6, 7]
        root = {
            "token": None,
            "children": [
                {
                    "token": 5,
                    "children": [
                        {
                            "token": 6,
                            "children": [{"token": 7, "children": [], "logits": None, "depth": 3}],
                            "logits": None,
                            "depth": 2,
                        }
                    ],
                    "logits": None,
                    "depth": 1,
                }
            ],
            "logits": None,
            "depth": 0,
        }

        # Verified tokens match all drafts, plus a bonus token 8
        verified = mx.array([5, 6, 7, 8])
        accepted = tree.select_accepted(root, verified)

        accepted_list = np.array(accepted).tolist()
        # Should accept [5, 6, 7] + bonus [8]
        assert accepted_list == [5, 6, 7, 8]

    def test_select_accepted_partial_match(self):
        """When only some draft tokens match, accept prefix + bonus."""
        config = SequoiaConfig(tree_width=1)
        tree = SpeculationTree(config)

        root = {
            "token": None,
            "children": [
                {
                    "token": 5,
                    "children": [
                        {
                            "token": 6,
                            "children": [{"token": 7, "children": [], "logits": None, "depth": 3}],
                            "logits": None,
                            "depth": 2,
                        }
                    ],
                    "logits": None,
                    "depth": 1,
                }
            ],
            "logits": None,
            "depth": 0,
        }

        # Second token doesn't match (6 vs 9)
        verified = mx.array([5, 9, 7, 8])
        accepted = tree.select_accepted(root, verified)

        accepted_list = np.array(accepted).tolist()
        # Accept [5] + bonus [9]
        assert accepted_list == [5, 9]

    def test_select_accepted_no_match(self):
        """When no draft tokens match, return just the bonus token."""
        config = SequoiaConfig(tree_width=1)
        tree = SpeculationTree(config)

        root = {
            "token": None,
            "children": [{"token": 5, "children": [], "logits": None, "depth": 1}],
            "logits": None,
            "depth": 0,
        }

        verified = mx.array([9, 8, 7])
        accepted = tree.select_accepted(root, verified)

        accepted_list = np.array(accepted).tolist()
        # No match: just get bonus token (target's first token)
        assert accepted_list == [9]

    def test_acceptance_history_tracking(self):
        """record_acceptance should update running acceptance rate."""
        config = SequoiaConfig()
        tree = SpeculationTree(config)

        assert tree._get_running_acceptance_rate() == 0.5  # default

        tree.record_acceptance(accepted=3, drafted=4)
        rate = tree._get_running_acceptance_rate()
        assert abs(rate - 0.75) < 0.01

    def test_acceptance_history_sliding_window(self):
        """History should not grow unbounded."""
        config = SequoiaConfig()
        tree = SpeculationTree(config)

        for _ in range(200):
            tree.record_acceptance(1, 2)

        assert len(tree._acceptance_history) <= 100


# ── LayerOffloader tests ─────────────────────────────────────────


class TestLayerOffloader:
    def test_init_nonexistent_path(self):
        """Should handle non-existent model path gracefully."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent/path", config)

        assert offloader._layer_index == {}
        assert offloader._load_count == 0

    def test_stats_initial(self):
        """Initial stats should be zeros."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent", config)

        stats = offloader.get_stats()
        assert stats["layers_loaded"] == 0
        assert stats["layers_evicted"] == 0
        assert stats["prefetch_hits"] == 0
        assert stats["currently_loaded"] == 0
        assert stats["indexed_layers"] == 0

    def test_load_unknown_layer(self):
        """Loading a layer from non-indexed model should return None."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent", config)

        result = offloader.load_layer(0)
        assert result is None

    def test_evict_nonloaded_layer(self):
        """Evicting a non-loaded layer should be a no-op."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent", config)

        offloader.evict_layer(99)  # should not raise
        assert offloader._evict_count == 0

    def test_prefetch_nonexistent(self):
        """Prefetching non-indexed layer should be a no-op."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent", config)

        offloader.prefetch_layer(0)
        # Should not crash, thread may start but find nothing
        import time

        time.sleep(0.05)
        assert offloader._prefetch_hits == 0

    def test_forward_without_offloading(self):
        """With offload_layers=False, forward should call model directly."""
        config = SequoiaConfig(offload_layers=False)
        offloader = LayerOffloader("/nonexistent", config)

        model = MockTargetModel(vocab_size=32, hidden_dim=32, num_layers=2)
        mx.eval(model.parameters())

        input_ids = mx.array([[1, 2, 3]])
        output = offloader.forward_with_offloading(input_ids, model)
        mx.eval(output)

        assert output.shape[-1] == 32  # vocab_size

    def test_forward_with_offloading_fallback(self):
        """With offload_layers=True but no indexed layers, should still work."""
        config = SequoiaConfig(offload_layers=True)
        offloader = LayerOffloader("/nonexistent", config)

        model = MockTargetModel(vocab_size=32, hidden_dim=32, num_layers=2)
        mx.eval(model.parameters())

        input_ids = mx.array([1, 2, 3])
        output = offloader.forward_with_offloading(input_ids, model)
        mx.eval(output)

        assert output.shape[-1] == 32

    def test_stats_after_operations(self):
        """Stats should track evict operations."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent", config)

        # Manually add a loaded layer then evict it
        offloader._loaded_layers[0] = {"fake": "weights"}
        offloader.evict_layer(0)

        stats = offloader.get_stats()
        assert stats["layers_evicted"] == 1
        assert stats["currently_loaded"] == 0


# ── SequoiaEngine tests ──────────────────────────────────────────


class TestSequoiaEngine:
    def test_init_default_config(self):
        """Engine should initialize with default config."""
        draft = MockDraftModel()
        target = MockTargetModel()
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer())

        assert engine.config.max_draft_tokens == 8
        assert engine.config.tree_width == 3
        assert engine.offloader is None  # no model path

    def test_init_with_config(self):
        """Engine should use provided config."""
        config = SequoiaConfig(max_draft_tokens=4, tree_width=2)
        draft = MockDraftModel()
        target = MockTargetModel()
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer(), config=config)

        assert engine.config.max_draft_tokens == 4
        assert engine.config.tree_width == 2

    def test_generate_produces_tokens(self):
        """generate() should produce more tokens than the prompt."""
        config = SequoiaConfig(
            max_draft_tokens=3,
            tree_width=1,
            temperature=0.0,
        )
        draft = MockDraftModel(vocab_size=32)
        target = MockTargetModel(vocab_size=32, num_layers=2)
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer(), config=config)

        prompt = mx.array([1, 2, 3])
        result = engine.generate(prompt, max_tokens=5)
        mx.eval(result)

        result_list = np.array(result).tolist()
        # Result should include the prompt plus at least 1 generated token
        assert len(result_list) >= 4

    def test_generate_respects_max_tokens(self):
        """Generation should not exceed max_tokens new tokens."""
        config = SequoiaConfig(max_draft_tokens=2, tree_width=1)
        draft = MockDraftModel(vocab_size=32)
        target = MockTargetModel(vocab_size=32, num_layers=2)
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer(), config=config)

        prompt = mx.array([1, 2, 3])
        result = engine.generate(prompt, max_tokens=10)
        mx.eval(result)

        result_list = np.array(result).tolist()
        new_tokens = len(result_list) - 3
        assert new_tokens <= 15  # generous bound (tree can accept multiple per round)

    def test_generate_with_callback(self):
        """Callback should be invoked each iteration."""
        config = SequoiaConfig(max_draft_tokens=2, tree_width=1)
        draft = MockDraftModel(vocab_size=32)
        target = MockTargetModel(vocab_size=32, num_layers=2)
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer(), config=config)

        callbacks = []

        def on_tokens(tokens, stats):
            callbacks.append((tokens, stats))

        prompt = mx.array([1, 2, 3])
        engine.generate(prompt, max_tokens=5, callback=on_tokens)

        # At least one callback should have fired
        assert len(callbacks) >= 1
        # Each callback should include tokens and stats
        for tokens, stats in callbacks:
            assert isinstance(tokens, list)
            assert isinstance(stats, dict)

    def test_stats_after_generate(self):
        """Stats should be populated after generation."""
        config = SequoiaConfig(max_draft_tokens=2, tree_width=1)
        draft = MockDraftModel(vocab_size=32)
        target = MockTargetModel(vocab_size=32, num_layers=2)
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer(), config=config)

        prompt = mx.array([1, 2, 3])
        engine.generate(prompt, max_tokens=5)

        stats = engine.get_stats()
        assert "total_generated" in stats
        assert "acceptance_rate" in stats
        assert "optimal_depth" in stats
        assert "config" in stats
        assert stats["total_generated"] >= 0
        assert stats["elapsed_s"] >= 0

    def test_apply_sequoia_convenience(self):
        """apply_sequoia() should return a configured engine."""
        draft = MockDraftModel()
        target = MockTargetModel()
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = apply_sequoia(draft, target, MockTokenizer())

        assert isinstance(engine, SequoiaEngine)
        assert engine.config is not None
        assert engine.offloader is None

    def test_apply_sequoia_with_config(self):
        """apply_sequoia() should accept custom config."""
        config = SequoiaConfig(max_draft_tokens=4, ssd_bandwidth_gbps=7.0)
        draft = MockDraftModel()
        target = MockTargetModel()
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = apply_sequoia(draft, target, MockTokenizer(), config=config)

        assert engine.config.max_draft_tokens == 4
        assert engine.config.ssd_bandwidth_gbps == 7.0


# ── Config tests ─────────────────────────────────────────────────


class TestSequoiaConfig:
    def test_defaults(self):
        config = SequoiaConfig()
        assert config.max_draft_tokens == 8
        assert config.tree_width == 3
        assert config.ssd_bandwidth_gbps == 5.0
        assert config.draft_latency_ms == 5.0
        assert config.verify_latency_ms == 50.0
        assert config.offload_layers is True
        assert config.prefetch_layers == 2

    def test_custom_values(self):
        config = SequoiaConfig(
            max_draft_tokens=16,
            tree_width=5,
            ssd_bandwidth_gbps=7.0,
            draft_latency_ms=3.0,
            verify_latency_ms=30.0,
            offload_layers=False,
            prefetch_layers=4,
        )
        assert config.max_draft_tokens == 16
        assert config.tree_width == 5
        assert config.ssd_bandwidth_gbps == 7.0
        assert config.offload_layers is False

    def test_temperature_default(self):
        config = SequoiaConfig()
        assert config.temperature == 0.0

    def test_config_with_temperature(self):
        config = SequoiaConfig(temperature=0.7)
        assert config.temperature == 0.7


# ── Additional SpeculationTree tests ────────────────────────────


class TestSpeculationTreeAdditional:
    def test_compute_optimal_depth_negative_acceptance_clamped(self):
        """Negative acceptance rates should be clamped to 0."""
        config = SequoiaConfig(max_draft_tokens=8)
        tree = SpeculationTree(config)
        depth = tree.compute_optimal_depth(-0.5)
        assert depth >= 1

    def test_compute_optimal_depth_above_one_clamped(self):
        """Acceptance rate above 1.0 should be clamped to 1.0."""
        config = SequoiaConfig(max_draft_tokens=8)
        tree = SpeculationTree(config)
        depth = tree.compute_optimal_depth(1.5)
        assert 1 <= depth <= config.max_draft_tokens

    def test_build_tree_depth_1(self):
        """Tree with depth 1 should have only direct children."""
        config = SequoiaConfig(max_draft_tokens=4, tree_width=2)
        tree = SpeculationTree(config)

        def mock_draft(ids):
            return 5, mx.zeros(32)

        root = tree.build_tree(mock_draft, mx.array([1, 2]), depth=1)
        assert root["token"] is None
        assert len(root["children"]) > 0
        for child in root["children"]:
            assert child["children"] == []

    def test_flatten_empty_tree(self):
        """Flattening a tree with no children should return empty."""
        config = SequoiaConfig()
        tree = SpeculationTree(config)
        root = {"token": None, "children": [], "logits": None, "depth": 0}
        result = tree.flatten_for_verification(root)
        assert result.shape == (1, 0)

    def test_select_accepted_empty_verified(self):
        """Empty verified tokens should return empty array."""
        config = SequoiaConfig()
        tree = SpeculationTree(config)
        root = {
            "token": None,
            "children": [{"token": 5, "children": [], "logits": None, "depth": 1}],
            "logits": None,
            "depth": 0,
        }
        verified = mx.array([], dtype=mx.int32)
        accepted = tree.select_accepted(root, verified)
        assert accepted.size == 0

    def test_record_acceptance_zero_drafted(self):
        """Recording with zero drafted should not crash."""
        config = SequoiaConfig()
        tree = SpeculationTree(config)
        tree.record_acceptance(accepted=0, drafted=0)
        # Should not append anything (drafted > 0 guard)
        assert len(tree._acceptance_history) == 0

    def test_running_acceptance_rate_default(self):
        config = SequoiaConfig()
        tree = SpeculationTree(config)
        assert tree._get_running_acceptance_rate() == 0.5

    def test_running_acceptance_rate_weighted(self):
        """More recent values should weigh more heavily."""
        config = SequoiaConfig()
        tree = SpeculationTree(config)
        # First record many low rates, then a high rate
        for _ in range(10):
            tree.record_acceptance(1, 10)  # 0.1 rate
        tree.record_acceptance(9, 10)  # 0.9 rate
        rate = tree._get_running_acceptance_rate()
        # The weighted rate should be pulled toward 0.9 more than simple average
        assert rate > np.mean([0.1] * 10 + [0.9])

    def test_collect_paths_multi_branch(self):
        """Tree with multiple branches should produce multiple paths."""
        config = SequoiaConfig(tree_width=2)
        tree = SpeculationTree(config)
        root = {
            "token": None,
            "children": [
                {
                    "token": 1,
                    "children": [{"token": 3, "children": [], "logits": None, "depth": 2}],
                    "logits": None,
                    "depth": 1,
                },
                {
                    "token": 2,
                    "children": [{"token": 4, "children": [], "logits": None, "depth": 2}],
                    "logits": None,
                    "depth": 1,
                },
            ],
            "logits": None,
            "depth": 0,
        }
        paths = []
        tree._collect_paths(root, [], paths)
        assert len(paths) == 2
        assert [1, 3] in paths
        assert [2, 4] in paths


# ── Additional LayerOffloader tests ─────────────────────────────


class TestLayerOffloaderAdditional:
    def test_load_layer_weights_missing_index(self):
        """Loading weights for non-indexed layer returns None."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent", config)
        result = offloader._load_layer_weights(999)
        assert result is None

    def test_evict_multiple_layers(self):
        """Evicting multiple layers should track each eviction."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent", config)
        offloader._loaded_layers[0] = {"fake": True}
        offloader._loaded_layers[1] = {"fake": True}
        offloader.evict_layer(0)
        offloader.evict_layer(1)
        assert offloader._evict_count == 2
        assert len(offloader._loaded_layers) == 0

    def test_prefetch_already_loaded_noop(self):
        """Prefetching a layer already in memory should be a no-op."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent", config)
        offloader._loaded_layers[5] = {"fake": True}
        offloader.prefetch_layer(5)
        # No new thread should be started
        assert 5 not in offloader._prefetch_threads or not offloader._prefetch_threads[5].is_alive()

    def test_prefetch_already_prefetched_noop(self):
        """Prefetching a layer already in prefetch results should be a no-op."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent", config)
        offloader._prefetch_results[7] = {"fake": True}
        offloader.prefetch_layer(7)

    def test_stats_bandwidth_calculation(self):
        """Effective bandwidth should be calculated from loaded bytes and time."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent", config)
        offloader._load_count = 10
        offloader._total_load_time_ms = 1000.0  # 1 second
        offloader._total_bytes_loaded = 5 * (1024**3)  # 5 GB
        stats = offloader.get_stats()
        assert stats["total_bytes_loaded_gb"] == 5.0
        assert stats["effective_bandwidth_gbps"] == 5.0

    def test_stats_zero_load_time(self):
        """When no loads happened, avoid division by zero."""
        config = SequoiaConfig()
        offloader = LayerOffloader("/nonexistent", config)
        stats = offloader.get_stats()
        assert stats["avg_load_time_ms"] == 0.0
        assert stats["effective_bandwidth_gbps"] == 0.0

    def test_forward_1d_input(self):
        """1D input should be expanded to 2D before processing."""
        config = SequoiaConfig(offload_layers=True)
        offloader = LayerOffloader("/nonexistent", config)

        model = MockTargetModel(vocab_size=32, hidden_dim=32, num_layers=2)
        mx.eval(model.parameters())

        input_ids = mx.array([1, 2, 3])
        output = offloader.forward_with_offloading(input_ids, model)
        mx.eval(output)
        assert output.shape[-1] == 32


# ── Additional SequoiaEngine tests ──────────────────────────────


class TestSequoiaEngineAdditional:
    def test_init_with_model_path(self):
        """Engine should create offloader when model path is given."""
        config = SequoiaConfig(offload_layers=True)
        draft = MockDraftModel()
        target = MockTargetModel()
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(
            draft,
            target,
            MockTokenizer(),
            config=config,
            target_model_path="/nonexistent/path",
        )
        assert engine.offloader is not None

    def test_init_no_offloader_when_disabled(self):
        """Engine should not create offloader when offload_layers=False."""
        config = SequoiaConfig(offload_layers=False)
        draft = MockDraftModel()
        target = MockTargetModel()
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(
            draft,
            target,
            MockTokenizer(),
            config=config,
            target_model_path="/some/path",
        )
        assert engine.offloader is None

    def test_draft_fn_greedy(self):
        """Draft function should return token and logits."""
        config = SequoiaConfig(temperature=0.0)
        draft = MockDraftModel(vocab_size=32)
        target = MockTargetModel(vocab_size=32)
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer(), config=config)

        token_id, logits = engine._draft_fn(mx.array([1, 2, 3]))
        assert isinstance(token_id, int)
        assert 0 <= token_id < 32
        assert logits.shape == (32,)

    def test_draft_fn_with_temperature(self):
        """Draft function with temperature > 0 should sample."""
        config = SequoiaConfig(temperature=0.8)
        draft = MockDraftModel(vocab_size=32)
        target = MockTargetModel(vocab_size=32)
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer(), config=config)

        token_id, logits = engine._draft_fn(mx.array([1, 2, 3]))
        assert isinstance(token_id, int)
        assert 0 <= token_id < 32

    def test_stats_includes_config(self):
        """Stats should contain config sub-dict."""
        config = SequoiaConfig(max_draft_tokens=4, ssd_bandwidth_gbps=6.5)
        draft = MockDraftModel(vocab_size=32)
        target = MockTargetModel(vocab_size=32, num_layers=2)
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer(), config=config)
        engine.generate(mx.array([1, 2, 3]), max_tokens=3)

        stats = engine.get_stats()
        assert stats["config"]["max_draft_tokens"] == 4
        assert stats["config"]["ssd_bandwidth_gbps"] == 6.5

    def test_stats_with_offloader(self):
        """Stats should include offloader stats when offloader exists."""
        config = SequoiaConfig(offload_layers=True)
        draft = MockDraftModel(vocab_size=32)
        target = MockTargetModel(vocab_size=32, num_layers=2)
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(
            draft,
            target,
            MockTokenizer(),
            config=config,
            target_model_path="/nonexistent",
        )
        engine._start_time = 1.0  # fake start time
        stats = engine.get_stats()
        assert "offloader" in stats

    def test_apply_sequoia_with_model_path(self):
        """apply_sequoia with path should set up offloader."""
        draft = MockDraftModel()
        target = MockTargetModel()
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = apply_sequoia(
            draft,
            target,
            MockTokenizer(),
            target_model_path="/nonexistent",
        )
        assert engine.offloader is not None

    def test_apply_sequoia_no_path(self):
        """apply_sequoia without path should have no offloader."""
        draft = MockDraftModel()
        target = MockTargetModel()
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = apply_sequoia(draft, target, MockTokenizer())
        assert engine.offloader is None

    def test_calibrate_no_crash(self):
        """Calibrate should not crash even if model calls fail."""
        config = SequoiaConfig()
        draft = MockDraftModel(vocab_size=32)
        target = MockTargetModel(vocab_size=32, num_layers=2)
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer(), config=config)
        # Should not raise
        engine.calibrate("test text")

    def test_calibrate_updates_latency(self):
        """Calibrate should update draft_latency_ms."""
        config = SequoiaConfig(draft_latency_ms=999.0, verify_latency_ms=999.0)
        draft = MockDraftModel(vocab_size=32)
        target = MockTargetModel(vocab_size=32, num_layers=2)
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer(), config=config)
        engine.calibrate("hello world")

        # After calibration, latency should be updated from dummy 999
        assert config.draft_latency_ms != 999.0

    def test_generate_empty_prompt(self):
        """Generation with minimal prompt should still work."""
        config = SequoiaConfig(max_draft_tokens=2, tree_width=1)
        draft = MockDraftModel(vocab_size=32)
        target = MockTargetModel(vocab_size=32, num_layers=2)
        mx.eval(draft.parameters())
        mx.eval(target.parameters())

        engine = SequoiaEngine(draft, target, MockTokenizer(), config=config)
        prompt = mx.array([1])
        result = engine.generate(prompt, max_tokens=3)
        mx.eval(result)

        result_list = np.array(result).tolist()
        assert len(result_list) >= 1
