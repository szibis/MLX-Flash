"""Tests for layer-wise quantization of dense transformer models."""

import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="mlx not available")

if HAS_MLX:
    from mlx_flash_compress.layer_quantization import (
        LayerQuantConfig,
        LayerSensitivityProfile,
        LayerQuantizer,
        apply_layer_quantization,
        estimate_model_size,
        _find_linear_layers,
        _count_linear_params,
    )


# -- Mock model classes mimicking standard transformer structure --


class MockSelfAttn(nn.Module):
    """Mock self-attention with q/k/v/o projections."""

    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def __call__(self, x):
        # Simplified: just sum projections
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


class MockMLP(nn.Module):
    """Mock MLP with gate/up/down projections."""

    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def __call__(self, x):
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MockTransformerLayer(nn.Module):
    """Mock transformer layer with self_attn and mlp."""

    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = MockSelfAttn(dim)
        self.mlp = MockMLP(dim)

    def __call__(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class MockModelInner(nn.Module):
    """Inner model containing the layers list."""

    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = [MockTransformerLayer(dim) for _ in range(num_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MockModel(nn.Module):
    """Top-level model with model.model.layers structure."""

    def __init__(self, dim: int = 64, num_layers: int = 8):
        super().__init__()
        self.model = MockModelInner(dim, num_layers)

    def __call__(self, x):
        return self.model(x)


# All dims must be divisible by group_size (64) for MLX quantization.
MODEL_DIM = 64


@pytest.fixture
def small_model():
    """Create a small model for testing (dim=64, 8 layers)."""
    model = MockModel(dim=MODEL_DIM, num_layers=8)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def tiny_model():
    """Create a tiny model for fast tests (dim=64, 4 layers)."""
    model = MockModel(dim=MODEL_DIM, num_layers=4)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def single_layer_model():
    """Model with just one layer."""
    model = MockModel(dim=MODEL_DIM, num_layers=1)
    mx.eval(model.parameters())
    return model


# -- Tests for LayerQuantConfig --


class TestLayerQuantConfig:
    def test_defaults(self):
        config = LayerQuantConfig()
        assert config.default_bits == 4
        assert config.sensitive_bits == 8
        assert config.sensitive_layers == []
        assert config.num_sensitive_start == 2
        assert config.num_sensitive_end == 2
        assert config.group_size == 64
        assert config.calibration_samples == 32

    def test_custom_config(self):
        config = LayerQuantConfig(
            default_bits=3,
            sensitive_bits=8,
            num_sensitive_start=3,
            num_sensitive_end=1,
        )
        assert config.default_bits == 3
        assert config.num_sensitive_start == 3
        assert config.num_sensitive_end == 1


# -- Tests for default_precision_map --


class TestDefaultPrecisionMap:
    def test_standard_32_layers(self):
        """32-layer model: first 2 and last 2 at Q8, rest at Q4."""
        pmap = LayerSensitivityProfile.default_precision_map(32)
        assert pmap[0] == 8  # first layer
        assert pmap[1] == 8  # second layer
        assert pmap[2] == 4  # third layer (middle)
        assert pmap[15] == 4  # middle layer
        assert pmap[29] == 4  # layer 29
        assert pmap[30] == 8  # second-to-last
        assert pmap[31] == 8  # last layer

    def test_all_layers_covered(self):
        """Every layer index should be present."""
        for n in [1, 4, 8, 32, 64]:
            pmap = LayerSensitivityProfile.default_precision_map(n)
            assert len(pmap) == n
            assert set(pmap.keys()) == set(range(n))

    def test_custom_sensitive_count(self):
        """Custom number of sensitive layers at start/end."""
        config = LayerQuantConfig(
            num_sensitive_start=3,
            num_sensitive_end=1,
            sensitive_bits=8,
            default_bits=4,
        )
        pmap = LayerSensitivityProfile.default_precision_map(10, config)
        # First 3 should be Q8
        assert pmap[0] == 8
        assert pmap[1] == 8
        assert pmap[2] == 8
        # Middle should be Q4
        assert pmap[3] == 4
        assert pmap[5] == 4
        assert pmap[8] == 4
        # Last 1 should be Q8
        assert pmap[9] == 8

    def test_single_layer(self):
        """Single layer model: should be Q8 (it's both first and last)."""
        pmap = LayerSensitivityProfile.default_precision_map(1)
        assert pmap[0] == 8

    def test_two_layers(self):
        """Two layers: both are first/last, both Q8."""
        pmap = LayerSensitivityProfile.default_precision_map(2)
        assert pmap[0] == 8
        assert pmap[1] == 8

    def test_four_layers_all_sensitive(self):
        """4 layers with 2 start + 2 end = all sensitive."""
        pmap = LayerSensitivityProfile.default_precision_map(4)
        for i in range(4):
            assert pmap[i] == 8

    def test_five_layers_one_middle(self):
        """5 layers: 2 start Q8, 1 middle Q4, 2 end Q8."""
        pmap = LayerSensitivityProfile.default_precision_map(5)
        assert pmap[0] == 8
        assert pmap[1] == 8
        assert pmap[2] == 4  # only middle layer
        assert pmap[3] == 8
        assert pmap[4] == 8

    def test_all_same_precision(self):
        """Config where default == sensitive → all same bits."""
        config = LayerQuantConfig(default_bits=4, sensitive_bits=4)
        pmap = LayerSensitivityProfile.default_precision_map(10, config)
        assert all(bits == 4 for bits in pmap.values())

    def test_q3_middle_layers(self):
        """Config with Q3 for middle layers."""
        config = LayerQuantConfig(default_bits=3, sensitive_bits=8)
        pmap = LayerSensitivityProfile.default_precision_map(10, config)
        assert pmap[0] == 8
        assert pmap[5] == 3  # middle layer at Q3
        assert pmap[9] == 8


# -- Tests for LayerSensitivityProfile --


class TestLayerSensitivityProfile:
    def test_init(self):
        profile = LayerSensitivityProfile(32)
        assert profile.num_layers == 32
        assert profile.sensitivity_scores == []

    def test_measure_sensitivity_returns_scores(self, tiny_model):
        """Sensitivity profiling should return one score per layer."""
        profile = LayerSensitivityProfile(4)
        calibration = mx.random.normal((2, 8, MODEL_DIM))  # batch=2, seq=8
        mx.eval(calibration)

        scores = profile.measure_sensitivity(tiny_model, calibration)

        assert len(scores) == 4
        assert all(isinstance(s, float) for s in scores)
        assert all(s >= 0.0 for s in scores)

    def test_profiled_precision_map(self, tiny_model):
        """Profiled precision map should cover all layers."""
        profile = LayerSensitivityProfile(4)
        calibration = mx.random.normal((2, 8, MODEL_DIM))
        mx.eval(calibration)

        profile.measure_sensitivity(tiny_model, calibration)
        pmap = profile.get_precision_map()

        assert len(pmap) == 4
        assert set(pmap.keys()) == {0, 1, 2, 3}
        assert all(bits in (4, 8) for bits in pmap.values())

    def test_precision_map_with_explicit_sensitive_layers(self):
        """Explicit sensitive_layers config should override profiling."""
        profile = LayerSensitivityProfile(8)
        profile.sensitivity_scores = [0.1] * 8  # dummy scores

        config = LayerQuantConfig(sensitive_layers=[0, 3, 7])
        pmap = profile.get_precision_map(config)

        assert pmap[0] == 8  # explicitly sensitive
        assert pmap[1] == 4
        assert pmap[3] == 8  # explicitly sensitive
        assert pmap[7] == 8  # explicitly sensitive

    def test_precision_map_without_profiling_falls_back(self):
        """Without profiling data, should return default heuristic map."""
        profile = LayerSensitivityProfile(8)
        pmap = profile.get_precision_map()

        # Should match default_precision_map behavior
        expected = LayerSensitivityProfile.default_precision_map(8)
        assert pmap == expected


# -- Tests for finding linear layers in model --


class TestFindLinearLayers:
    def test_finds_all_projections(self):
        layer = MockTransformerLayer(dim=MODEL_DIM)
        mx.eval(layer.parameters())
        linears = _find_linear_layers(layer)

        names = [name for name, _ in linears]
        assert "self_attn.q_proj" in names
        assert "self_attn.k_proj" in names
        assert "self_attn.v_proj" in names
        assert "self_attn.o_proj" in names
        assert "mlp.gate_proj" in names
        assert "mlp.up_proj" in names
        assert "mlp.down_proj" in names
        assert len(linears) == 7

    def test_count_params(self):
        d = MODEL_DIM
        h = d * 4  # hidden_dim
        layer = MockTransformerLayer(dim=d)
        mx.eval(layer.parameters())
        params = _count_linear_params(layer)

        # self_attn: 4 * (d*d)
        # mlp: gate (d*h) + up (d*h) + down (h*d)
        expected = 4 * (d * d) + 2 * (d * h) + 1 * (h * d)
        assert params == expected


# -- Tests for LayerQuantizer --


class TestLayerQuantizer:
    def test_init_defaults(self):
        q = LayerQuantizer()
        assert q.config.default_bits == 4
        stats = q.get_stats()
        assert stats["layers_quantized"] == 0
        assert stats["linears_quantized"] == 0

    def test_quantize_model_shapes_preserved(self, tiny_model):
        """After quantization, model should still accept same input shape."""
        quantizer = LayerQuantizer()
        pmap = LayerSensitivityProfile.default_precision_map(4)

        result = quantizer.quantize_model(tiny_model, pmap)

        assert result["num_layers"] == 4
        assert result["precision_map"] == pmap

        # Verify model still works
        x = mx.random.normal((1, 4, MODEL_DIM))
        mx.eval(x)
        out = tiny_model(x)
        mx.eval(out)
        assert out.shape == (1, 4, MODEL_DIM)

    def test_quantize_model_stats(self, tiny_model):
        """Quantizer should track statistics."""
        quantizer = LayerQuantizer()
        pmap = LayerSensitivityProfile.default_precision_map(4)

        quantizer.quantize_model(tiny_model, pmap)
        stats = quantizer.get_stats()

        assert stats["layers_quantized"] == 4
        # 7 linear layers per transformer layer * 4 layers = 28
        assert stats["linears_quantized"] == 28
        assert stats["elapsed_ms"] > 0

    def test_quantize_single_layer(self, tiny_model):
        """Quantize a single layer and verify it still works."""
        quantizer = LayerQuantizer()
        layer = tiny_model.model.layers[0]

        x = mx.random.normal((1, 4, MODEL_DIM))
        mx.eval(x)

        # Quantize at Q4
        quantizer.quantize_layer(layer, bits=4, group_size=64)

        out = layer(x)
        mx.eval(out)
        assert out.shape == (1, 4, MODEL_DIM)

    def test_quantize_uses_default_map_when_none(self, tiny_model):
        """When no precision_map given, uses default heuristic."""
        quantizer = LayerQuantizer()
        result = quantizer.quantize_model(tiny_model)

        # Default: first 2 and last 2 at Q8, but model has 4 layers
        # so all should be Q8
        assert result["precision_map"][0] == 8
        assert result["precision_map"][3] == 8

    def test_quantize_with_custom_bits(self, small_model):
        """Custom precision map: mix of Q3, Q4, Q8."""
        pmap = {
            0: 8,
            1: 4,
            2: 4,
            3: 3,
            4: 3,
            5: 4,
            6: 4,
            7: 8,
        }
        quantizer = LayerQuantizer()
        result = quantizer.quantize_model(small_model, pmap)

        assert result["precision_map"] == pmap
        assert result["stats"]["layers_quantized"] == 8

        # Model should still work
        x = mx.random.normal((1, 4, MODEL_DIM))
        mx.eval(x)
        out = small_model(x)
        mx.eval(out)
        assert out.shape == (1, 4, MODEL_DIM)


# -- Tests for memory estimation --


class TestMemoryEstimation:
    def test_uniform_q4_no_savings(self, small_model):
        """Uniform Q4 should have zero savings vs Q4 baseline."""
        pmap = {i: 4 for i in range(8)}
        quantizer = LayerQuantizer()
        est = quantizer.estimate_memory_savings(small_model, pmap)

        assert est["savings_ratio"] == pytest.approx(0.0)
        assert est["uniform_q4_bytes"] == est["mixed_bytes"]
        assert est["num_layers"] == 8

    def test_q8_layers_increase_size(self, small_model):
        """Q8 sensitive layers should make mixed size larger than Q4 uniform."""
        pmap = {i: 8 if i in (0, 7) else 4 for i in range(8)}
        quantizer = LayerQuantizer()
        est = quantizer.estimate_memory_savings(small_model, pmap)

        # Q8 layers are larger than Q4
        assert est["mixed_bytes"] > est["uniform_q4_bytes"]
        assert est["savings_ratio"] < 0

    def test_q3_middle_saves_memory(self, small_model):
        """Q3 middle layers with Q8 ends: net savings depends on ratio."""
        pmap = {i: 8 if i in (0, 7) else 3 for i in range(8)}
        quantizer = LayerQuantizer()
        est = quantizer.estimate_memory_savings(small_model, pmap)

        # 6 layers at Q3 (0.375 bytes/param) vs Q4 (0.5 bytes/param) saves
        # 2 layers at Q8 (1.0 bytes/param) costs more
        # Net depends on balance
        assert est["total_params"] > 0
        assert est["effective_bits"] > 0

    def test_all_q2_maximum_savings(self, small_model):
        """All Q2 should give maximum savings vs Q4."""
        pmap = {i: 2 for i in range(8)}
        quantizer = LayerQuantizer()
        est = quantizer.estimate_memory_savings(small_model, pmap)

        # Q2 = 0.25 bytes/param vs Q4 = 0.5 bytes/param → 50% savings
        assert est["savings_ratio"] == pytest.approx(0.5)
        assert est["effective_bits"] == pytest.approx(2.0)

    def test_estimate_model_size_convenience(self, small_model):
        """estimate_model_size should work as standalone function."""
        pmap = {i: 4 for i in range(8)}
        est = estimate_model_size(small_model, pmap)

        assert "total_params" in est
        assert "uniform_q4_bytes" in est
        assert "mixed_bytes" in est
        assert "savings_ratio" in est
        assert "effective_bits" in est

    def test_effective_bits_calculation(self, small_model):
        """Effective bits should reflect the weighted average."""
        # All layers same size, so effective bits = average bits
        pmap = {i: 4 for i in range(8)}
        est = estimate_model_size(small_model, pmap)
        assert est["effective_bits"] == pytest.approx(4.0)


# -- Tests for apply_layer_quantization (one-line API) --


class TestApplyLayerQuantization:
    def test_default_no_profiling(self, tiny_model):
        """Default call: heuristic map, no profiling."""
        result = apply_layer_quantization(tiny_model)

        assert result["profiling"]["profiled"] is False
        assert result["num_layers"] == 4
        assert "memory" in result
        assert "precision_map" in result

        # Model should still work
        x = mx.random.normal((1, 4, MODEL_DIM))
        mx.eval(x)
        out = tiny_model(x)
        mx.eval(out)
        assert out.shape == (1, 4, MODEL_DIM)

    def test_with_custom_config(self, small_model):
        """Custom config: Q3 for middle, Q8 for edges."""
        config = LayerQuantConfig(
            default_bits=3,
            sensitive_bits=8,
            num_sensitive_start=1,
            num_sensitive_end=1,
        )
        result = apply_layer_quantization(small_model, config=config)

        pmap = result["precision_map"]
        assert pmap[0] == 8  # first layer sensitive
        assert pmap[7] == 8  # last layer sensitive
        assert pmap[3] == 3  # middle layer at Q3
        assert pmap[4] == 3

    def test_profile_requires_calibration_data(self, tiny_model):
        """Profiling without calibration data should raise."""
        with pytest.raises(ValueError, match="calibration_data is required"):
            apply_layer_quantization(tiny_model, profile=True)

    def test_with_profiling(self, tiny_model):
        """Profiling mode should run and produce sensitivity scores."""
        calibration = mx.random.normal((2, 8, MODEL_DIM))
        mx.eval(calibration)

        result = apply_layer_quantization(
            tiny_model, profile=True, calibration_data=calibration
        )

        assert result["profiling"]["profiled"] is True
        assert len(result["profiling"]["sensitivity_scores"]) == 4
        assert result["num_layers"] == 4


# -- Edge cases --


class TestEdgeCases:
    def test_single_layer_model(self, single_layer_model):
        """Single layer model should work with default config."""
        result = apply_layer_quantization(single_layer_model)

        assert result["num_layers"] == 1
        # Single layer is both first and last → Q8
        assert result["precision_map"][0] == 8

        x = mx.random.normal((1, 4, MODEL_DIM))
        mx.eval(x)
        out = single_layer_model(x)
        mx.eval(out)
        assert out.shape == (1, 4, MODEL_DIM)

    def test_all_layers_same_precision(self, small_model):
        """All layers at same precision should work."""
        config = LayerQuantConfig(default_bits=4, sensitive_bits=4)
        result = apply_layer_quantization(small_model, config=config)

        assert all(bits == 4 for bits in result["precision_map"].values())

    def test_empty_precision_map_uses_default(self, tiny_model):
        """Empty precision map should fall back to default."""
        quantizer = LayerQuantizer()
        result = quantizer.quantize_model(tiny_model)

        assert result["num_layers"] == 4
        assert result["stats"]["layers_quantized"] == 4

    def test_zero_sensitive_start_end(self):
        """Zero sensitive layers at start/end → all default bits."""
        config = LayerQuantConfig(
            num_sensitive_start=0,
            num_sensitive_end=0,
            default_bits=4,
        )
        pmap = LayerSensitivityProfile.default_precision_map(10, config)
        assert all(bits == 4 for bits in pmap.values())

    def test_sensitive_count_exceeds_layers(self):
        """When sensitive start+end > num_layers, all layers are sensitive."""
        config = LayerQuantConfig(
            num_sensitive_start=5,
            num_sensitive_end=5,
            sensitive_bits=8,
        )
        pmap = LayerSensitivityProfile.default_precision_map(6, config)
        # All 6 layers covered by start(5) or end(5) → all Q8
        assert all(bits == 8 for bits in pmap.values())

    def test_precision_map_keys_are_ints(self):
        """Precision map keys should be integers."""
        pmap = LayerSensitivityProfile.default_precision_map(10)
        assert all(isinstance(k, int) for k in pmap.keys())
        assert all(isinstance(v, int) for v in pmap.values())
