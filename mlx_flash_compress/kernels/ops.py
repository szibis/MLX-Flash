"""Wired Metal kernel operations with Python fallbacks.

Provides optimized operations that use custom Metal kernels when available,
falling back to standard MLX operations when Metal compiler is not present.

Wired operations:
  swiglu()          — Fused SiLU(gate) * up in single kernel (saves 1 memory write)
  moe_dispatch()    — Parallel expert gather + weighted sum (saves Python loop)

Usage:
  from mlx_flash_compress.kernels.ops import swiglu, moe_dispatch

  # Automatically uses Metal kernel if available, Python fallback otherwise
  output = swiglu(gate_result, up_result)
  output = moe_dispatch(expert_outputs, selected_ids, routing_weights)
"""

import sys
from typing import Optional

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from mlx_flash_compress.kernels.loader import get_kernel_loader


_metal_swiglu_available: Optional[bool] = None
_metal_moe_dispatch_available: Optional[bool] = None


def _check_metal_kernel_support() -> bool:
    """Check if MLX supports custom Metal kernels via mx.fast.metal_kernel."""
    if not HAS_MLX:
        return False
    return hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")


def swiglu(gate: "mx.array", up: "mx.array") -> "mx.array":
    """Fused SwiGLU activation: silu(gate) * up.

    Standard implementation requires 3 operations and 3 memory writes:
      1. silu_gate = gate * sigmoid(gate)   → write intermediate
      2. output = silu_gate * up            → write output

    Metal kernel fuses into 1 operation and 1 write:
      output[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]

    Falls back to standard MLX ops if Metal kernel unavailable.
    """
    global _metal_swiglu_available

    if not HAS_MLX:
        raise RuntimeError("MLX required for swiglu")

    # Try Metal kernel path
    if _metal_swiglu_available is None:
        _metal_swiglu_available = _check_metal_kernel_support()
        if _metal_swiglu_available:
            loader = get_kernel_loader()
            _metal_swiglu_available = loader.available and loader.compile_shader("flash_dequant") is not None

    if _metal_swiglu_available:
        try:
            return _swiglu_metal(gate, up)
        except Exception:
            _metal_swiglu_available = False

    # Fallback: standard MLX ops
    return _swiglu_mlx(gate, up)


def _swiglu_mlx(gate: "mx.array", up: "mx.array") -> "mx.array":
    """SwiGLU via standard MLX operations."""
    return mx.sigmoid(gate) * gate * up


def _swiglu_metal(gate: "mx.array", up: "mx.array") -> "mx.array":
    """SwiGLU via custom Metal kernel (if mx.fast.metal_kernel available)."""
    source = """
        uint idx = thread_position_in_grid.x;
        float g = gate[idx];
        float u = up[idx];
        float silu_g = g / (1.0f + exp(-g));
        output[idx] = silu_g * u;
    """
    kernel = mx.fast.metal_kernel(
        name="swiglu_fused",
        input_names=["gate", "up"],
        output_names=["output"],
        source=source,
    )
    return kernel(
        inputs=[gate.astype(mx.float32), up.astype(mx.float32)],
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
        grid=(gate.size, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]


def moe_dispatch(
    expert_outputs: "mx.array",
    selected_ids: "mx.array",
    routing_weights: "mx.array",
) -> "mx.array":
    """Weighted sum of selected expert outputs.

    Args:
        expert_outputs: (num_experts, hidden_dim) — all expert outputs
        selected_ids: (top_k,) — indices of selected experts
        routing_weights: (top_k,) — routing scores for selected experts

    Returns:
        (hidden_dim,) — weighted sum of selected expert outputs

    Standard implementation loops over top_k experts in Python.
    This uses vectorized gather + broadcast multiply.
    """
    if not HAS_MLX:
        raise RuntimeError("MLX required for moe_dispatch")

    # Vectorized: gather selected experts, multiply by weights, sum
    selected = expert_outputs[selected_ids]  # (top_k, hidden_dim)
    weights = routing_weights[:, None]        # (top_k, 1)
    return mx.sum(selected * weights, axis=0) # (hidden_dim,)


def moe_dispatch_numpy(
    expert_outputs: "np.ndarray",
    selected_ids: "np.ndarray",
    routing_weights: "np.ndarray",
) -> "np.ndarray":
    """NumPy fallback for moe_dispatch (used in tests without MLX)."""
    if not HAS_NUMPY:
        raise RuntimeError("NumPy required")
    selected = expert_outputs[selected_ids]
    weights = routing_weights[:, None]
    return np.sum(selected * weights, axis=0)


def is_metal_available() -> bool:
    """Check if Metal kernel acceleration is available."""
    return _check_metal_kernel_support() and get_kernel_loader().available


def get_kernel_status() -> dict:
    """Get status of all wired kernels."""
    loader = get_kernel_loader()
    return {
        "metal_compiler": loader.available,
        "mlx_fast_metal_kernel": _check_metal_kernel_support(),
        "swiglu_metal": _metal_swiglu_available or False,
        "moe_dispatch": "vectorized_mlx",  # always uses vectorized path
        "compiled_shaders": list(loader.compiled.keys()),
    }
