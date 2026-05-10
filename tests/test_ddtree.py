"""Tests for DDTree: Dynamic Draft Trees."""

import numpy as np
import pytest

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from mlx_flash_compress.ddtree import DDTreeBuilder, DDTreeConfig, sequoia_optimal_tree


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestDDTreeBuilder:
    def test_build_tree_basic(self):
        config = DDTreeConfig(tree_width=3, max_tree_size=20)
        builder = DDTreeBuilder(config)

        logits = mx.random.normal((1, 8, 100))
        tree = builder.build_tree(logits)

        assert tree.size > 0
        assert tree.size <= 20
        assert tree.token_ids.shape == (tree.size,)
        assert tree.attention_mask.shape == (tree.size, tree.size)
        assert tree.depth_ids.shape == (tree.size,)

    def test_attention_mask_ancestor_only(self):
        config = DDTreeConfig(tree_width=2, max_tree_size=10, min_confidence=0.0)
        builder = DDTreeBuilder(config)

        logits = mx.zeros((4, 10))
        logits = logits.at[:, 0].add(10.0)
        logits = logits.at[:, 1].add(8.0)
        logits = mx.expand_dims(logits, 0)

        tree = builder.build_tree(logits)
        mask = np.array(tree.attention_mask)

        for i in range(tree.size):
            assert mask[i, i], "Node must attend to itself"

            node = tree.nodes[i]
            j = node.parent_idx
            while j >= 0:
                assert mask[i, j], f"Node {i} must attend to ancestor {j}"
                j = tree.nodes[j].parent_idx

    def test_tree_respects_budget(self):
        config = DDTreeConfig(tree_width=5, max_tree_size=30)
        builder = DDTreeBuilder(config)

        logits = mx.random.normal((1, 15, 1000))
        tree = builder.build_tree(logits)
        assert tree.size <= 30

    def test_verify_tree_accepts_matching(self):
        config = DDTreeConfig(tree_width=2, max_tree_size=10)
        builder = DDTreeBuilder(config)

        logits = mx.zeros((1, 4, 10))
        logits = logits.at[:, :, 3].add(100.0)
        tree = builder.build_tree(logits)

        ctx_len = 5
        total_len = ctx_len + tree.size
        verify_logits = mx.zeros((1, total_len, 10))
        verify_logits = verify_logits.at[:, ctx_len - 1:, 3].add(100.0)

        accepted, n = builder.verify_tree(tree, verify_logits, ctx_len)
        assert n > 0
        assert all(t == 3 for t in accepted)

    def test_verify_tree_bonus_token(self):
        config = DDTreeConfig(tree_width=1, max_tree_size=5, min_confidence=0.0)
        builder = DDTreeBuilder(config)

        logits = mx.zeros((1, 3, 10))
        logits = logits.at[:, :, 5].add(100.0)
        tree = builder.build_tree(logits)

        ctx_len = 3
        total_len = ctx_len + tree.size
        verify_logits = mx.zeros((1, total_len, 10))
        verify_logits = verify_logits.at[:, ctx_len - 1, 5].add(100.0)
        verify_logits = verify_logits.at[:, ctx_len, 7].add(100.0)

        accepted, n = builder.verify_tree(tree, verify_logits, ctx_len)
        assert n >= 1

    def test_empty_tree(self):
        config = DDTreeConfig(tree_width=1, max_tree_size=0)
        builder = DDTreeBuilder(config)
        logits = mx.random.normal((1, 4, 10))
        tree = builder.build_tree(logits)
        assert tree.size == 0

    def test_stats_tracking(self):
        config = DDTreeConfig(tree_width=2, max_tree_size=10)
        builder = DDTreeBuilder(config)

        logits = mx.random.normal((1, 4, 50))
        tree = builder.build_tree(logits)

        ctx_len = 3
        verify_logits = mx.random.normal((1, ctx_len + tree.size, 50))
        builder.verify_tree(tree, verify_logits, ctx_len)

        stats = builder.get_stats()
        assert stats["trees_built"] == 1
        assert stats["avg_tree_size"] > 0


class TestSequoiaOptimalTree:
    def test_basic_dp(self):
        probs = [0.8, 0.5, 0.3]
        result = sequoia_optimal_tree(budget=10, acceptance_probs=probs)
        assert result["expected_tokens"] > 0
        assert result["budget"] == 10
        assert result["optimal_root_branches"] >= 1

    def test_larger_budget_more_tokens(self):
        probs = [0.7, 0.5, 0.3, 0.2]
        small = sequoia_optimal_tree(budget=5, acceptance_probs=probs)
        large = sequoia_optimal_tree(budget=20, acceptance_probs=probs)
        assert large["expected_tokens"] >= small["expected_tokens"]

    def test_single_node(self):
        result = sequoia_optimal_tree(budget=1, acceptance_probs=[0.5])
        assert result["expected_tokens"] == 1.0
