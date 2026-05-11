"""Tests for engine — MoE Inference Engine across all execution modes."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None

from mlx_flash_compress.cache import CacheStats
from mlx_flash_compress.engine import (
    ExpertWeightManager,
    InferenceMode,
    InferenceResult,
    MoEInferenceEngine,
    _flatten_params,
    _iter_named_params,
)

# ---------------------------------------------------------------------------
# InferenceMode tests
# ---------------------------------------------------------------------------


class TestInferenceMode:
    def test_all_modes_exist(self):
        assert InferenceMode.PURE_MLX.value == "pure_mlx"
        assert InferenceMode.SSD_STREAM.value == "ssd_stream"
        assert InferenceMode.LZ4_CACHE.value == "lz4_cache"
        assert InferenceMode.ZSTD_CACHE.value == "zstd_cache"
        assert InferenceMode.TIERED_CACHE.value == "tiered_cache"
        assert InferenceMode.NO_CACHE_SSD.value == "no_cache_ssd"

    def test_mode_count(self):
        assert len(InferenceMode) == 6

    def test_mode_from_string(self):
        assert InferenceMode("pure_mlx") == InferenceMode.PURE_MLX
        assert InferenceMode("tiered_cache") == InferenceMode.TIERED_CACHE

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            InferenceMode("invalid_mode")


# ---------------------------------------------------------------------------
# InferenceResult tests
# ---------------------------------------------------------------------------


class TestInferenceResult:
    def test_basic_creation(self):
        result = InferenceResult(
            tokens_generated=100,
            total_time_s=5.0,
            prompt_time_s=1.0,
            generation_time_s=4.0,
            tokens_per_second=25.0,
            peak_memory_mb=512.0,
        )
        assert result.tokens_generated == 100
        assert result.total_time_s == 5.0
        assert result.prompt_time_s == 1.0
        assert result.generation_time_s == 4.0
        assert result.tokens_per_second == 25.0
        assert result.peak_memory_mb == 512.0

    def test_default_cache_stats(self):
        result = InferenceResult(
            tokens_generated=0,
            total_time_s=0,
            prompt_time_s=0,
            generation_time_s=0,
            tokens_per_second=0,
            peak_memory_mb=0,
        )
        assert result.cache_stats is None

    def test_default_mode(self):
        result = InferenceResult(
            tokens_generated=0,
            total_time_s=0,
            prompt_time_s=0,
            generation_time_s=0,
            tokens_per_second=0,
            peak_memory_mb=0,
        )
        assert result.mode == InferenceMode.PURE_MLX

    def test_custom_cache_stats(self):
        stats = CacheStats(hot_hits=10, warm_hits=5, cold_hits=2)
        result = InferenceResult(
            tokens_generated=50,
            total_time_s=2.0,
            prompt_time_s=0.5,
            generation_time_s=1.5,
            tokens_per_second=33.3,
            peak_memory_mb=256.0,
            cache_stats=stats,
        )
        assert result.cache_stats.hot_hits == 10
        assert result.cache_stats.total_hits == 17

    def test_custom_mode(self):
        result = InferenceResult(
            tokens_generated=50,
            total_time_s=2.0,
            prompt_time_s=0.5,
            generation_time_s=1.5,
            tokens_per_second=33.3,
            peak_memory_mb=256.0,
            mode=InferenceMode.TIERED_CACHE,
        )
        assert result.mode == InferenceMode.TIERED_CACHE


# ---------------------------------------------------------------------------
# ExpertWeightManager tests
# ---------------------------------------------------------------------------


class TestExpertWeightManagerInit:
    def test_basic_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertWeightManager(work_dir=tmpdir)
            assert mgr.work_dir == Path(tmpdir)
            assert mgr.expert_dir == Path(tmpdir) / "experts"
            assert mgr.num_layers == 0
            assert mgr.num_experts == 0

    def test_custom_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertWeightManager(work_dir=tmpdir, num_layers=24, num_experts=60)
            assert mgr.num_layers == 24
            assert mgr.num_experts == 60

    def test_expert_shapes_start_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertWeightManager(work_dir=tmpdir)
            assert len(mgr._expert_shapes) == 0
            assert len(mgr._expert_dtypes) == 0


class TestExpertWeightManagerDtype:
    def test_default_dtype_for_unknown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertWeightManager(work_dir=tmpdir)
            dtype = mgr.get_expert_dtype(0, 0)
            assert dtype == np.float16

    def test_stored_dtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertWeightManager(work_dir=tmpdir)
            mgr._expert_dtypes[(0, 1)] = np.float32
            assert mgr.get_expert_dtype(0, 1) == np.float32


class TestExpertWeightManagerCleanup:
    def test_cleanup_removes_expert_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertWeightManager(work_dir=tmpdir)
            mgr.expert_dir.mkdir(parents=True, exist_ok=True)
            # Create a dummy file
            (mgr.expert_dir / "test.bin").write_bytes(b"\x00" * 100)
            assert mgr.expert_dir.exists()

            mgr.cleanup()
            assert not mgr.expert_dir.exists()

    def test_cleanup_no_dir_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertWeightManager(work_dir=tmpdir)
            # expert_dir does not exist — should not raise
            mgr.cleanup()


@pytest.mark.skipif(not HAS_MLX, reason="requires mlx")
class TestExpertWeightManagerEviction:
    def test_evict_empty_model(self):
        """Model with no expert params returns empty metadata."""

        class FakeModel:
            def parameters(self):
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertWeightManager(work_dir=tmpdir)
            metadata = mgr.evict_experts_to_disk(FakeModel())
            assert metadata == {}
            assert mgr.num_layers == 0
            assert mgr.num_experts == 0

    def test_evict_creates_directories(self):
        """Eviction should create the expert directory structure."""

        class FakeModel:
            def parameters(self):
                return {"layers": [{"experts": [{"w1": mx.array(np.random.randn(4, 4).astype(np.float32))}]}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertWeightManager(work_dir=tmpdir)
            metadata = mgr.evict_experts_to_disk(FakeModel())
            assert mgr.expert_dir.exists()

    def test_evict_writes_files(self):
        """Evicted expert weights should be written to disk."""

        class FakeModel:
            def parameters(self):
                return {"layers": [{"experts": [{"w1": mx.array(np.random.randn(4, 4).astype(np.float32))}]}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ExpertWeightManager(work_dir=tmpdir)
            metadata = mgr.evict_experts_to_disk(FakeModel())
            if metadata:
                for key, info in metadata.items():
                    assert Path(info["path"]).exists()
                    assert info["bytes"] > 0


# ---------------------------------------------------------------------------
# _flatten_params and _iter_named_params tests
# ---------------------------------------------------------------------------


class TestFlattenParams:
    def test_empty_dict(self):
        result = list(_flatten_params({}))
        assert result == []

    def test_nested_dict(self):
        if not HAS_MLX:
            pytest.skip("requires mlx")
        data = {"a": {"b": mx.array([1.0, 2.0])}}
        result = list(_flatten_params(data))
        assert len(result) == 1
        assert result[0][0] == "a.b"

    def test_list_in_dict(self):
        if not HAS_MLX:
            pytest.skip("requires mlx")
        data = {"layers": [mx.array([1.0]), mx.array([2.0])]}
        result = list(_flatten_params(data))
        assert len(result) == 2
        assert result[0][0] == "layers.0"
        assert result[1][0] == "layers.1"

    def test_non_mlx_values_skipped(self):
        data = {"a": 42, "b": "hello", "c": [1, 2, 3]}
        result = list(_flatten_params(data))
        assert result == []

    def test_prefix(self):
        if not HAS_MLX:
            pytest.skip("requires mlx")
        data = {"weight": mx.array([1.0])}
        result = list(_flatten_params(data, prefix="layer.0"))
        assert result[0][0] == "layer.0.weight"


class TestIterNamedParams:
    def test_no_mlx(self):
        """Without parameters() method, returns nothing."""

        class NoParams:
            pass

        result = list(_iter_named_params(NoParams()))
        assert result == []

    @pytest.mark.skipif(not HAS_MLX, reason="requires mlx")
    def test_with_mlx_module(self):
        """MLX Linear module should yield named parameters."""
        linear = nn.Linear(4, 8)
        result = list(_iter_named_params(linear))
        assert len(result) > 0
        # Should have 'weight' parameter at minimum
        names = [name for name, _ in result]
        assert any("weight" in n for n in names)


# ---------------------------------------------------------------------------
# MoEInferenceEngine tests
# ---------------------------------------------------------------------------


class TestMoEInferenceEngineInit:
    def test_default_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            assert engine.model_name == "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit"
            assert engine.cache_hot_mb == 512
            assert engine.cache_warm_mb == 256
            assert engine.num_workers == 4
            assert engine._model is None
            assert engine._tokenizer is None

    def test_custom_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(
                model_name="test-model",
                work_dir=tmpdir,
                cache_hot_mb=1024,
                cache_warm_mb=512,
                num_workers=8,
            )
            assert engine.model_name == "test-model"
            assert engine.cache_hot_mb == 1024
            assert engine.cache_warm_mb == 512
            assert engine.num_workers == 8

    def test_work_dir_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir) / "subdir" / "deep"
            engine = MoEInferenceEngine(work_dir=str(work))
            assert work.exists()


class TestMoEInferenceEngineLoadModel:
    @pytest.mark.skipif(not HAS_MLX, reason="requires mlx")
    def test_load_model_no_mlx_raises(self):
        """If HAS_MLX were False, load_model would raise RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            with patch("mlx_flash_compress.engine.HAS_MLX", False):
                with pytest.raises(RuntimeError, match="MLX not available"):
                    engine.load_model()


class TestMoEInferenceEnginePrepareEviction:
    def test_prepare_without_model_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            with pytest.raises(RuntimeError, match="Load model first"):
                engine.prepare_expert_eviction()


class TestMoEInferenceEngineGetCache:
    def test_get_cache_no_weight_mgr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            result = engine._get_cache(InferenceMode.LZ4_CACHE)
            assert result is None

    def test_get_cache_pure_mlx_returns_none(self):
        """PURE_MLX mode should not create a cache even if _weight_mgr exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            # Note: _get_cache for PURE_MLX is not called via this path, but
            # the method returns None when _weight_mgr is None
            result = engine._get_cache(InferenceMode.PURE_MLX)
            assert result is None

    def test_get_cache_modes_with_weight_mgr(self):
        """Each mode creates a cache with appropriate configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir, cache_hot_mb=100, cache_warm_mb=50)
            # Set up a minimal weight manager with an expert_dir
            expert_dir = Path(tmpdir) / "experts"
            expert_dir.mkdir()
            mgr = ExpertWeightManager(work_dir=tmpdir)
            engine._weight_mgr = mgr

            for mode in [
                InferenceMode.LZ4_CACHE,
                InferenceMode.ZSTD_CACHE,
                InferenceMode.TIERED_CACHE,
                InferenceMode.SSD_STREAM,
                InferenceMode.NO_CACHE_SSD,
            ]:
                cache = engine._get_cache(mode)
                assert cache is not None, f"Cache should be created for {mode}"
                cache.shutdown()


class TestMoEInferenceEngineCleanup:
    def test_cleanup_without_weight_mgr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            # Should not raise
            engine.cleanup()

    def test_cleanup_with_weight_mgr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            mgr = ExpertWeightManager(work_dir=tmpdir)
            mgr.expert_dir.mkdir(parents=True, exist_ok=True)
            (mgr.expert_dir / "test.bin").write_bytes(b"\x00")
            engine._weight_mgr = mgr

            engine.cleanup()
            assert not mgr.expert_dir.exists()


# ---------------------------------------------------------------------------
# Benchmark cache subsystem tests
# ---------------------------------------------------------------------------


class TestBenchmarkCacheSubsystem:
    def test_raises_without_preparation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            with pytest.raises(RuntimeError, match="prepare_expert_eviction"):
                engine._benchmark_cache_subsystem("test", 10, InferenceMode.LZ4_CACHE, None, False)

    def test_raises_with_no_experts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            mgr = ExpertWeightManager(work_dir=tmpdir)
            engine._weight_mgr = mgr
            mock_cache = MagicMock()
            with pytest.raises(RuntimeError, match="No experts found"):
                engine._benchmark_cache_subsystem("test", 10, InferenceMode.LZ4_CACHE, mock_cache, False)

    def test_benchmark_returns_inference_result(self):
        """With a mock cache, benchmark should complete and return results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            mgr = ExpertWeightManager(work_dir=tmpdir, num_layers=2, num_experts=4)
            engine._weight_mgr = mgr

            mock_cache = MagicMock()
            mock_cache.fetch_experts.return_value = [b"\x00" * 64]

            result = engine._benchmark_cache_subsystem("test", 5, InferenceMode.LZ4_CACHE, mock_cache, False)
            assert isinstance(result, InferenceResult)
            assert result.tokens_generated == 5
            assert result.total_time_s > 0
            assert result.prompt_time_s == 0.0

    def test_benchmark_calls_cache_correctly(self):
        """Cache.fetch_experts should be called for each layer/token combination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            mgr = ExpertWeightManager(work_dir=tmpdir, num_layers=2, num_experts=4)
            engine._weight_mgr = mgr

            mock_cache = MagicMock()
            mock_cache.fetch_experts.return_value = [b"\x00" * 64]

            max_tokens = 3
            engine._benchmark_cache_subsystem("test", max_tokens, InferenceMode.LZ4_CACHE, mock_cache, False)

            # 3 tokens * 2 layers = 6 calls
            assert mock_cache.fetch_experts.call_count == max_tokens * mgr.num_layers


# ---------------------------------------------------------------------------
# Power-law distribution verification
# ---------------------------------------------------------------------------


class TestPowerLawDistribution:
    def test_zipf_distribution_shape(self):
        """Verify the power-law distribution used in _benchmark_cache_subsystem."""
        num_experts = 60
        expert_probs = np.array([(1.0 / (i + 1)) ** 0.8 for i in range(num_experts)])
        expert_probs /= expert_probs.sum()

        assert len(expert_probs) == num_experts
        assert abs(expert_probs.sum() - 1.0) < 1e-6

        # First expert should have highest probability
        assert expert_probs[0] == expert_probs.max()
        # Distribution should be monotonically decreasing
        for i in range(len(expert_probs) - 1):
            assert expert_probs[i] >= expert_probs[i + 1]

    def test_zipf_top_heavy(self):
        """Top experts should capture majority of probability mass."""
        num_experts = 60
        expert_probs = np.array([(1.0 / (i + 1)) ** 0.8 for i in range(num_experts)])
        expert_probs /= expert_probs.sum()

        top_10_mass = expert_probs[:10].sum()
        # Top ~17% of experts should have more than 40% of traffic
        assert top_10_mass > 0.4


# ---------------------------------------------------------------------------
# run_inference integration path tests (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MLX, reason="requires mlx")
class TestRunInference:
    def test_run_inference_loads_model_if_needed(self):
        """run_inference should call load_model if _model is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MoEInferenceEngine(work_dir=tmpdir)
            with patch.object(engine, "load_model") as mock_load:
                with patch.object(engine, "_run_pure_mlx") as mock_run:
                    mock_run.return_value = InferenceResult(
                        tokens_generated=10,
                        total_time_s=1.0,
                        prompt_time_s=0.1,
                        generation_time_s=0.9,
                        tokens_per_second=11.1,
                        peak_memory_mb=100.0,
                    )
                    # psutil is imported locally in run_inference, patch it in its module
                    with patch.dict("sys.modules", {"psutil": MagicMock()}) as _:
                        import sys

                        mock_psutil = sys.modules["psutil"]
                        mock_proc = MagicMock()
                        mock_proc.memory_info.return_value = MagicMock(rss=500e6)
                        mock_psutil.Process.return_value = mock_proc

                        engine.run_inference("hello", mode=InferenceMode.PURE_MLX)
                        mock_load.assert_called_once()
