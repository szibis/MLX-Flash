"""Metal kernel acceleration for MLX-Flash.

Provides custom Metal shaders for:
- Q4 dequantization + fused GEMV (avoids intermediate FP16 materialization)
- MoE expert dispatch (parallel expert gather + scatter)
- SwiGLU fusion (single kernel for gate * silu(up) instead of 3 ops)

These kernels are optional — MLX-Flash works without them, but they
reduce memory bandwidth and kernel launch overhead by 15-30%.
"""

from mlx_flash_compress.kernels.loader import MetalKernelLoader, get_kernel_loader

__all__ = ["MetalKernelLoader", "get_kernel_loader"]
