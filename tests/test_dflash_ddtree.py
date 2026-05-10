"""Tests for DFlash block diffusion and DDTree draft trees."""

import numpy as np
import pytest

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from mlx_flash_compress.dflash import (
    DFlashConfig, DFlashEngine, DFlashStats, BlockDiffusionDrafter,
    DrafterBlock, NGramDrafter,
)
from mlx_flash_compress.ddtree import (
    DDTreeConfig, DDTreeBuilder, DraftTree, TreeNode,
)


# -- DFlash Config Tests --

class TestDFlashConfig:
    def test_default_config(self):
        config = DFlashConfig()
        assert config.num_spec_tokens == 15
        assert config.num_denoise_steps == 2
        assert config.temperature == 0.0
        assert config.tree_width == 1

    def test_custom_config(self):
        config = DFlashConfig(num_spec_tokens=10, num_denoise_steps=3, tree_width=5)
        assert config.num_spec_tokens == 10
        assert config.num_denoise_steps == 3
        assert config.tree_width == 5


# -- DFlash Stats Tests --

class TestDFlashStats:
    def test_empty_stats(self):
        stats = DFlashStats()
        assert stats.acceptance_rate == 0.0
        assert stats.tokens_per_draft == 0.0
        assert stats.speedup_factor == 1.0

    def test_stats_with_data(self):
        stats = DFlashStats(
            total_drafts=100,
            total_accepted=75,
            total_target_calls=10,
        )
        assert stats.acceptance_rate == 0.75
        assert stats.tokens_per_draft == 7.5
        assert stats.speedup_factor == 7.5


# -- NGram Drafter Tests --

class TestNGramDrafter:
    def test_observe_and_draft(self):
        drafter = NGramDrafter(n=3, num_draft=5)
        tokens = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5]
        drafter.observe(tokens)

        drafts = drafter.draft([1, 2, 3])
        assert len(drafts) > 0
        assert drafts[0] == 4  # most common continuation of [1,2,3]

    def test_empty_context(self):
        drafter = NGramDrafter(n=3, num_draft=5)
        drafts = drafter.draft([1, 2, 3])
        assert drafts == []

    def test_long_draft_chain(self):
        drafter = NGramDrafter(n=3, num_draft=10)
        # Repeating pattern: 1,2,3,4,5,1,2,3,4,5,...
        tokens = [1, 2, 3, 4, 5] * 10
        drafter.observe(tokens)

        drafts = drafter.draft([3, 4, 5])
        assert len(drafts) > 0
        assert drafts[0] == 1  # after [3,4,5] comes 1


# -- DDTree Config Tests --

class TestDDTreeConfig:
    def test_default_config(self):
        config = DDTreeConfig()
        assert config.tree_width == 3
        assert config.max_depth == 15
        assert config.max_tree_size == 64
        assert config.adaptive_width is True

    def test_custom_config(self):
        config = DDTreeConfig(tree_width=5, max_depth=10, max_tree_size=100)
        assert config.tree_width == 5
        assert config.max_depth == 10
        assert config.max_tree_size == 100


# -- DDTree Builder Tests --

@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestDDTreeBuilder:
    def test_build_tree_basic(self):
        config = DDTreeConfig(tree_width=2, max_depth=3, max_tree_size=20)
        builder = DDTreeBuilder(config)

        # Simulate DFlash logits: [1, 3, vocab_size=100]
        logits = mx.random.normal((1, 3, 100))
        tree = builder.build_tree(logits)

        assert isinstance(tree, DraftTree)
        assert tree.size > 0
        assert tree.size <= config.max_tree_size
        assert tree.max_depth <= config.max_depth

    def test_tree_attention_mask(self):
        config = DDTreeConfig(tree_width=2, max_depth=2, max_tree_size=10)
        builder = DDTreeBuilder(config)

        logits = mx.random.normal((1, 2, 50))
        tree = builder.build_tree(logits)

        # Attention mask should be square
        mask = tree.attention_mask
        assert mask.shape[0] == mask.shape[1]
        assert mask.shape[0] == tree.size

        # Diagonal should be True (self-attention)
        mask_np = np.array(mask)
        for i in range(tree.size):
            assert mask_np[i, i] == True

    def test_path_to_root(self):
        config = DDTreeConfig(tree_width=2, max_depth=3, max_tree_size=20)
        builder = DDTreeBuilder(config)

        logits = mx.random.normal((1, 3, 50))
        tree = builder.build_tree(logits)

        # Every node should have a path to root
        for i in range(tree.size):
            path = tree.node_to_path[i]
            assert len(path) > 0
            assert path[-1] == i
            # Root nodes have parent_idx == -1
            first_node = tree.nodes[path[0]]
            assert first_node.parent_idx == -1

    def test_max_tree_size_respected(self):
        config = DDTreeConfig(tree_width=5, max_depth=10, max_tree_size=30)
        builder = DDTreeBuilder(config)

        logits = mx.random.normal((1, 10, 200))
        tree = builder.build_tree(logits)

        assert tree.size <= config.max_tree_size

    def test_tree_stats(self):
        config = DDTreeConfig(tree_width=2, max_depth=3, max_tree_size=20)
        builder = DDTreeBuilder(config)

        logits = mx.random.normal((1, 3, 50))
        builder.build_tree(logits)
        builder.build_tree(logits)

        stats = builder.get_stats()
        assert stats["trees_built"] == 2
        assert stats["total_nodes_explored"] > 0


# -- DFlash Drafter Model Tests --

@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestBlockDiffusionDrafter:
    def test_drafter_forward(self):
        vocab_size = 1000
        hidden_dim = 128
        drafter = BlockDiffusionDrafter(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            num_draft_positions=8,
            num_checkpoint_layers=3,
        )

        # Simulated hidden states from 3 checkpoint layers
        hidden_states = [mx.random.normal((1, 10, hidden_dim)) for _ in range(3)]
        noisy_tokens = mx.random.normal((1, 8, hidden_dim))

        logits = drafter(hidden_states, noisy_tokens)
        mx.eval(logits)

        assert logits.shape == (1, 8, vocab_size)

    def test_drafter_block(self):
        block = DrafterBlock(hidden_dim=64, num_heads=4)
        x = mx.random.normal((1, 10, 64))
        out = block(x)
        mx.eval(out)
        assert out.shape == x.shape


# -- DFlash Engine Tests --

@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestDFlashEngine:
    def test_engine_init_no_drafter(self):
        """Engine should work without a drafter (fallback mode)."""

        class FakeModel:
            class model:
                layers = [None] * 32
                @staticmethod
                def embed_tokens(x):
                    return mx.zeros((1, x.shape[1], 128))

        config = DFlashConfig(num_spec_tokens=10)
        engine = DFlashEngine(FakeModel(), drafter=None, config=config)

        assert len(config.checkpoint_layers) == 5
        assert config.checkpoint_layers[0] == 1

    def test_stats_summary(self):
        config = DFlashConfig()
        engine = DFlashEngine(None, drafter=None, config=config)
        engine.stats.total_accepted = 100
        engine.stats.total_target_calls = 15
        engine.stats.total_drafter_calls = 15
        engine.stats.total_drafts = 225

        summary = engine.get_stats_summary()
        assert "acceptance_rate" in summary
        assert "tokens_per_draft_step" in summary
        assert "effective_speedup" in summary

    def test_auto_detect_checkpoints(self):
        """Should auto-detect 5 evenly-spaced checkpoint layers."""

        class FakeModel:
            class model:
                layers = [None] * 64

        config = DFlashConfig()
        engine = DFlashEngine(FakeModel(), drafter=None, config=config)

        assert len(config.checkpoint_layers) == 5
        assert config.checkpoint_layers[0] == 1
        assert config.checkpoint_layers[-1] == 63


# -- Integration: DFlash + DDTree --

@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestDFlashDDTreeIntegration:
    def test_drafter_to_tree_pipeline(self):
        """Verify the full pipeline: drafter produces logits → DDTree builds tree."""
        vocab_size = 500
        hidden_dim = 64
        num_draft = 8

        drafter = BlockDiffusionDrafter(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            num_draft_positions=num_draft,
            num_checkpoint_layers=3,
        )

        hidden_states = [mx.random.normal((1, 5, hidden_dim)) for _ in range(3)]
        noisy_tokens = mx.random.normal((1, num_draft, hidden_dim))

        logits = drafter(hidden_states, noisy_tokens)
        mx.eval(logits)

        # Feed logits to DDTree
        tree_config = DDTreeConfig(tree_width=3, max_depth=num_draft, max_tree_size=30)
        builder = DDTreeBuilder(tree_config)
        tree = builder.build_tree(logits)

        assert tree.size > 0
        assert tree.size <= 30
        assert tree.token_ids.shape[0] == tree.size
        assert tree.attention_mask.shape == (tree.size, tree.size)
