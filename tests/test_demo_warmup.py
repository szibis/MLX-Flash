"""Tests for the warm-up demo simulation."""
import shutil
import tempfile
import numpy as np
import pytest
from pathlib import Path
from mlx_flash_compress.demo_warmup import (
    create_expert_files, make_topic_routing,
    WarmupSession, run_warmup_session,
)
from mlx_flash_compress.lcp_cache import LCPCache


class TestExpertFiles:
    def test_create_expert_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expert_dir = create_expert_files(tmpdir, num_layers=2, num_experts=4,
                                             expert_size_bytes=1024)
            files = list(Path(expert_dir).rglob("*.bin"))
            assert len(files) == 8  # 2 layers * 4 experts

    def test_expert_file_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expert_dir = create_expert_files(tmpdir, num_layers=1, num_experts=2,
                                             expert_size_bytes=512)
            for f in Path(expert_dir).rglob("*.bin"):
                assert f.stat().st_size == 512


class TestTopicRouting:
    def test_different_topics_different_distributions(self):
        rng = np.random.default_rng(42)
        p1 = make_topic_routing("coding", 60, rng)
        p2 = make_topic_routing("writing", 60, rng)
        assert p1.shape == (60,)
        assert abs(p1.sum() - 1.0) < 1e-6
        top_coding = set(np.argsort(p1)[-10:])
        top_writing = set(np.argsort(p2)[-10:])
        assert top_coding != top_writing

    def test_same_topic_same_distribution(self):
        rng = np.random.default_rng(42)
        p1 = make_topic_routing("coding", 60, rng)
        p2 = make_topic_routing("coding", 60, rng)
        np.testing.assert_array_equal(p1, p2)


class TestWarmupSession:
    def test_session_records_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            expert_dir = create_expert_files(tmpdir, num_layers=4, num_experts=8,
                                             expert_size_bytes=256)
            cache = LCPCache(str(expert_dir), capacity_bytes=4096,
                             enable_dendritic=False, enable_skip_fallback=False)
            rng = np.random.default_rng(42)
            session = run_warmup_session(cache, "test", num_tokens=10,
                                         num_layers=4, num_experts=8, k=2,
                                         ssd_latency_ms=0, rng=rng,
                                         show_every=100)
            assert len(session.token_metrics) == 10
            assert session.topic == "test"
            cache.shutdown()
