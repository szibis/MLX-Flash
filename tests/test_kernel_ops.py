"""Tests for wired Metal kernel operations."""

import pytest
import numpy as np

from mlx_flash_compress.kernels.ops import (
    moe_dispatch_numpy,
    is_metal_available,
    get_kernel_status,
)

try:
    import mlx.core as mx
    from mlx_flash_compress.kernels.ops import swiglu, moe_dispatch, _swiglu_mlx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class TestSwiGLU:
    @pytest.mark.skipif(not HAS_MLX, reason="MLX required")
    def test_basic_swiglu(self):
        gate = mx.array([1.0, 2.0, 3.0])
        up = mx.array([1.0, 1.0, 1.0])
        result = swiglu(gate, up)
        assert result.shape == (3,)
        # silu(x) = x * sigmoid(x), so silu(1) * 1 ≈ 0.731
        assert float(result[0]) > 0.5

    @pytest.mark.skipif(not HAS_MLX, reason="MLX required")
    def test_swiglu_zero_gate(self):
        gate = mx.array([0.0, 0.0])
        up = mx.array([5.0, 10.0])
        result = swiglu(gate, up)
        # silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert abs(float(result[0])) < 0.01

    @pytest.mark.skipif(not HAS_MLX, reason="MLX required")
    def test_swiglu_matches_reference(self):
        """Verify our swiglu matches the standard silu(gate) * up."""
        gate = mx.array([1.0, -1.0, 2.0, -2.0, 0.5])
        up = mx.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = swiglu(gate, up)
        # Reference: silu(x) = x * sigmoid(x)
        ref = mx.sigmoid(gate) * gate * up
        mx.eval(result, ref)
        diff = float(mx.max(mx.abs(result - ref)))
        assert diff < 1e-5

    @pytest.mark.skipif(not HAS_MLX, reason="MLX required")
    def test_swiglu_batch(self):
        gate = mx.random.normal((4, 128))
        up = mx.random.normal((4, 128))
        result = swiglu(gate, up)
        assert result.shape == (4, 128)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX required")
    def test_swiglu_mlx_fallback(self):
        gate = mx.array([1.0, 2.0])
        up = mx.array([3.0, 4.0])
        result = _swiglu_mlx(gate, up)
        assert result.shape == (2,)


class TestMoEDispatch:
    @pytest.mark.skipif(not HAS_MLX, reason="MLX required")
    def test_basic_dispatch(self):
        expert_outputs = mx.array([
            [1.0, 0.0, 0.0],  # expert 0
            [0.0, 1.0, 0.0],  # expert 1
            [0.0, 0.0, 1.0],  # expert 2
        ])
        selected = mx.array([0, 2])
        weights = mx.array([0.7, 0.3])
        result = moe_dispatch(expert_outputs, selected, weights)
        # 0.7 * [1,0,0] + 0.3 * [0,0,1] = [0.7, 0.0, 0.3]
        mx.eval(result)
        assert abs(float(result[0]) - 0.7) < 1e-5
        assert abs(float(result[2]) - 0.3) < 1e-5

    @pytest.mark.skipif(not HAS_MLX, reason="MLX required")
    def test_single_expert(self):
        expert_outputs = mx.array([[1.0, 2.0, 3.0]])
        selected = mx.array([0])
        weights = mx.array([1.0])
        result = moe_dispatch(expert_outputs, selected, weights)
        mx.eval(result)
        assert float(result[0]) == 1.0
        assert float(result[1]) == 2.0

    @pytest.mark.skipif(not HAS_MLX, reason="MLX required")
    def test_equal_weights(self):
        expert_outputs = mx.array([
            [2.0, 0.0],
            [0.0, 4.0],
        ])
        selected = mx.array([0, 1])
        weights = mx.array([0.5, 0.5])
        result = moe_dispatch(expert_outputs, selected, weights)
        mx.eval(result)
        assert abs(float(result[0]) - 1.0) < 1e-5
        assert abs(float(result[1]) - 2.0) < 1e-5


class TestMoEDispatchNumpy:
    def test_basic(self):
        expert_outputs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        selected = np.array([0, 1])
        weights = np.array([0.6, 0.4])
        result = moe_dispatch_numpy(expert_outputs, selected, weights)
        assert abs(result[0] - 0.6) < 1e-5
        assert abs(result[1] - 0.4) < 1e-5

    def test_single_expert(self):
        expert_outputs = np.array([[3.0, 5.0]])
        selected = np.array([0])
        weights = np.array([1.0])
        result = moe_dispatch_numpy(expert_outputs, selected, weights)
        assert result[0] == 3.0
        assert result[1] == 5.0

    def test_large(self):
        expert_outputs = np.random.randn(128, 512)
        selected = np.array([0, 5, 10, 50])
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        result = moe_dispatch_numpy(expert_outputs, selected, weights)
        assert result.shape == (512,)


class TestKernelStatus:
    def test_status_fields(self):
        status = get_kernel_status()
        assert "metal_compiler" in status
        assert "mlx_fast_metal_kernel" in status
        assert "swiglu_metal" in status
        assert "moe_dispatch" in status
        assert "compiled_shaders" in status

    def test_metal_available_is_bool(self):
        assert isinstance(is_metal_available(), bool)
