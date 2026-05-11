"""Tests for model browser: model info, scoring, search, formatting."""

from unittest.mock import MagicMock, patch

import pytest

from mlx_flash_compress.model_browser import (
    KNOWN_MODELS,
    MoEModelInfo,
    score_model,
)


class TestMoEModelInfo:
    def test_basic_fields(self):
        m = MoEModelInfo(
            name="test/model",
            display_name="Test Model",
            size_gb=10.0,
            num_layers=24,
            num_experts=60,
            top_k=4,
            expert_size_mb=5.0,
            gpu_layer_ms=1.0,
            category="small",
            description="Test model",
        )
        assert m.name == "test/model"
        assert m.size_gb == 10.0
        assert m.available is True  # default

    def test_unavailable_model(self):
        m = MoEModelInfo(
            name="N/A",
            display_name="Future Model",
            size_gb=200.0,
            num_layers=60,
            num_experts=512,
            top_k=4,
            expert_size_mb=6.0,
            gpu_layer_ms=2.0,
            category="huge",
            description="Not yet",
            available=False,
        )
        assert m.available is False


class TestKnownModels:
    def test_known_models_not_empty(self):
        assert len(KNOWN_MODELS) > 0

    def test_all_models_have_required_fields(self):
        for m in KNOWN_MODELS:
            assert m.name
            assert m.display_name
            assert m.size_gb > 0
            assert m.num_layers > 0
            assert m.num_experts > 0
            assert m.top_k > 0
            assert m.category in {"small", "medium", "large", "huge"}

    def test_models_sorted_by_size(self):
        """Verify models are roughly sorted by size (small to huge)."""
        sizes = [m.size_gb for m in KNOWN_MODELS]
        # Not strictly sorted, but first should be < last
        assert sizes[0] < sizes[-1]

    def test_has_small_model(self):
        small = [m for m in KNOWN_MODELS if m.category == "small"]
        assert len(small) > 0

    def test_has_available_models(self):
        available = [m for m in KNOWN_MODELS if m.available]
        assert len(available) > 0

    def test_has_unavailable_models(self):
        unavailable = [m for m in KNOWN_MODELS if not m.available]
        assert len(unavailable) > 0

    def test_expert_size_positive(self):
        for m in KNOWN_MODELS:
            assert m.expert_size_mb > 0


class TestScoreModel:
    @patch("mlx_flash_compress.model_browser.estimate_performance")
    def test_model_fits_ram(self, mock_est):
        hw = MagicMock()
        hw.available_ram_gb = 128.0

        mock_result = MagicMock()
        mock_result.estimated_tok_per_s = 15.0
        mock_result.estimated_hit_rate = 1.0
        mock_result.bottleneck = "compute"
        mock_est.return_value = mock_result

        model = MoEModelInfo(
            name="test/small",
            display_name="Small",
            size_gb=5.0,
            num_layers=24,
            num_experts=60,
            top_k=4,
            expert_size_mb=5.0,
            gpu_layer_ms=0.5,
            category="small",
            description="Fits in RAM",
        )
        score = score_model(model, hw)
        assert score["fits_ram"] is True
        assert score["needs_ssd"] is False

    @patch("mlx_flash_compress.model_browser.estimate_performance")
    def test_model_needs_ssd(self, mock_est):
        hw = MagicMock()
        hw.available_ram_gb = 16.0

        mock_result = MagicMock()
        mock_result.estimated_tok_per_s = 3.0
        mock_result.estimated_hit_rate = 0.7
        mock_result.bottleneck = "ssd_bandwidth"
        mock_est.return_value = mock_result

        model = MoEModelInfo(
            name="test/big",
            display_name="Big",
            size_gb=200.0,
            num_layers=60,
            num_experts=512,
            top_k=4,
            expert_size_mb=6.0,
            gpu_layer_ms=2.0,
            category="huge",
            description="Needs SSD",
        )
        score = score_model(model, hw)
        assert score["fits_ram"] is False
        assert score["needs_ssd"] is True

    @patch("mlx_flash_compress.model_browser.estimate_performance")
    def test_score_has_required_keys(self, mock_est):
        hw = MagicMock()
        hw.available_ram_gb = 64.0

        mock_result = MagicMock()
        mock_result.estimated_tok_per_s = 10.0
        mock_result.estimated_hit_rate = 0.9
        mock_result.bottleneck = "compute"
        mock_est.return_value = mock_result

        model = KNOWN_MODELS[0]
        score = score_model(model, hw)
        expected_keys = {
            "fits_ram", "needs_ssd", "base_tps", "optimized_tps",
            "hit_rate", "optimized_hit_rate", "speedup", "bottleneck",
        }
        assert expected_keys <= set(score.keys())

    @patch("mlx_flash_compress.model_browser.estimate_performance")
    def test_speedup_calculation(self, mock_est):
        hw = MagicMock()
        hw.available_ram_gb = 64.0

        base = MagicMock()
        base.estimated_tok_per_s = 5.0
        base.estimated_hit_rate = 0.7
        base.bottleneck = "ssd_bandwidth"

        optimized = MagicMock()
        optimized.estimated_tok_per_s = 10.0
        optimized.estimated_hit_rate = 0.9
        optimized.bottleneck = "ssd_bandwidth"

        mock_est.side_effect = [base, optimized]

        model = KNOWN_MODELS[0]
        score = score_model(model, hw)
        assert score["speedup"] == 2.0


class TestRunModel:
    def test_import(self):
        from mlx_flash_compress.model_browser import run_model
        assert callable(run_model)

    @patch("mlx_flash_compress.model_browser.subprocess.run")
    def test_run_unavailable_model_does_not_call_subprocess(self, mock_run, capsys):
        from mlx_flash_compress.model_browser import run_model

        model = MoEModelInfo(
            name="N/A",
            display_name="Unavailable",
            size_gb=200.0,
            num_layers=60,
            num_experts=512,
            top_k=4,
            expert_size_mb=6.0,
            gpu_layer_ms=2.0,
            category="huge",
            description="Not available",
            available=False,
        )
        run_model(model)
        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "not yet available" in captured.out
