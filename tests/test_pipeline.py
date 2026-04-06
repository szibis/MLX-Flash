"""Tests for phase-level pipelined execution."""

import mmap
import os
import sys
import tempfile
import time

import pytest

from mlx_flash_compress.pipeline import (
    PipelinedExecutor,
    PipelineStats,
    PrefetchWorker,
    LayerPhase,
)
from mlx_flash_compress.page_cache import PageCacheAdvisor, EvictionStrategy


@pytest.fixture
def temp_mmap():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(os.urandom(65536))
        f.flush()
        path = f.name
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        yield mm
        mm.close()
    os.unlink(path)


@pytest.fixture
def executor():
    return PipelinedExecutor(eviction_strategy=EvictionStrategy.NONE)


class TestPipelineStats:
    def test_defaults(self):
        stats = PipelineStats()
        assert stats.layers_executed == 0
        assert stats.total_compute_ms == 0.0

    def test_io_hidden_pct_no_io(self):
        stats = PipelineStats(total_io_ms=0.0)
        assert stats.io_hidden_pct == 100.0

    def test_io_hidden_pct_with_overlap(self):
        stats = PipelineStats(total_io_ms=10.0, overlap_ratio=0.8)
        assert stats.io_hidden_pct == 80.0


class TestPrefetchWorker:
    def test_init(self):
        advisor = PageCacheAdvisor(strategy=EvictionStrategy.NONE)
        worker = PrefetchWorker(advisor)
        assert worker.prefetch_depth >= 1

    def test_prefetch_depth_fast_io(self):
        advisor = PageCacheAdvisor(strategy=EvictionStrategy.NONE)
        worker = PrefetchWorker(advisor)
        worker._ema_io_ms = 1.0
        worker._ema_compute_ms = 10.0
        assert worker.prefetch_depth == 1

    def test_prefetch_depth_balanced(self):
        advisor = PageCacheAdvisor(strategy=EvictionStrategy.NONE)
        worker = PrefetchWorker(advisor)
        worker._ema_io_ms = 10.0
        worker._ema_compute_ms = 10.0
        assert worker.prefetch_depth == 2

    def test_prefetch_depth_io_bound(self):
        advisor = PageCacheAdvisor(strategy=EvictionStrategy.NONE)
        worker = PrefetchWorker(advisor)
        worker._ema_io_ms = 30.0
        worker._ema_compute_ms = 10.0
        assert worker.prefetch_depth == 3

    def test_submit_and_wait(self, temp_mmap):
        advisor = PageCacheAdvisor(strategy=EvictionStrategy.NONE)
        worker = PrefetchWorker(advisor)
        worker.submit_prefetch(0, temp_mmap, [(0, 4096)])
        result = worker.wait_for(0)
        # With NONE strategy, the prefetch is a no-op but should complete
        assert result is True
        worker.shutdown()

    def test_wait_for_nonexistent(self):
        advisor = PageCacheAdvisor(strategy=EvictionStrategy.NONE)
        worker = PrefetchWorker(advisor)
        result = worker.wait_for(999)
        assert result is False
        worker.shutdown()

    def test_duplicate_submit_ignored(self, temp_mmap):
        advisor = PageCacheAdvisor(strategy=EvictionStrategy.NONE)
        worker = PrefetchWorker(advisor)
        worker.submit_prefetch(0, temp_mmap, [(0, 4096)])
        worker.submit_prefetch(0, temp_mmap, [(0, 4096)])  # should be ignored
        worker.wait_for(0)
        worker.shutdown()

    def test_update_timing(self):
        advisor = PageCacheAdvisor(strategy=EvictionStrategy.NONE)
        worker = PrefetchWorker(advisor)
        initial_io = worker._ema_io_ms
        worker.update_timing(100.0, 50.0)
        assert worker._ema_io_ms > initial_io


class TestPipelinedExecutor:
    def test_init(self, executor):
        assert executor.stats.layers_executed == 0

    def test_execute_layer_phases(self, executor, temp_mmap):
        """Test full phase-level pipeline execution."""
        call_order = []

        def norm_fn():
            call_order.append("norm")
            return "norm_out"

        def attn_fn(x):
            call_order.append("attn")
            assert x == "norm_out"
            return "attn_out"

        def mlp_fn(x):
            call_order.append("mlp")
            assert x == "attn_out"
            return "mlp_out"

        result = executor.execute_layer_phases(
            layer_idx=0,
            total_layers=10,
            mmap_obj=temp_mmap,
            attn_byte_ranges=[(0, 4096)],
            mlp_byte_ranges=[(4096, 4096)],
            compute_norm_fn=norm_fn,
            compute_attn_fn=attn_fn,
            compute_mlp_fn=mlp_fn,
            next_attn_byte_ranges=[(0, 4096)],
        )

        assert result == "mlp_out"
        assert call_order == ["norm", "attn", "mlp"]
        assert executor.stats.layers_executed == 1

    def test_execute_multiple_layers(self, executor, temp_mmap):
        for i in range(5):
            executor.execute_layer_phases(
                layer_idx=i,
                total_layers=10,
                mmap_obj=temp_mmap,
                attn_byte_ranges=[(0, 4096)],
                mlp_byte_ranges=[(4096, 4096)],
                compute_norm_fn=lambda: "n",
                compute_attn_fn=lambda x: "a",
                compute_mlp_fn=lambda x: "m",
            )

        assert executor.stats.layers_executed == 5
        assert executor.stats.total_compute_ms > 0

    def test_shutdown(self, executor):
        executor.shutdown()
        # Should not raise


class TestLayerPhase:
    def test_defaults(self):
        phase = LayerPhase(name="test")
        assert phase.name == "test"
        assert phase.weight_keys == []
        assert phase.compute_fn is None
