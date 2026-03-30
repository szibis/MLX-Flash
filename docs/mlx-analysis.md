# MLX Technical Analysis

Source: https://github.com/ml-explore/mlx

## Overview

Apple's NumPy-style ML framework for Apple Silicon. 24,873 stars, C++ core with Python bindings. Key design pillars: lazy computation graph, unified memory (zero CPU-GPU copy).

## Architecture Stack

```
Python API  (mlx.core / mlx.nn)
    |
C++ Array + Primitives  (mlx/array.h, mlx/primitives.h)
    |
Scheduler + Graph Executor  (BFS topological sort)
    |
Backend dispatch
    +-- Metal (GPU)  mlx/backend/metal/
    +-- CPU          mlx/backend/cpu/
```

## Lazy Computation

Every `mx.add()`, `mx.matmul()` etc. creates an `array` node with an attached `Primitive` and records input edges. Nothing executes until `mx.eval()` or `mx.async_eval()`.

### `mx.compile()` Kernel Fusion

1. Traces function to build compute graph
2. Identifies fusable primitive runs (unary/binary/ternary ops)
3. Collapses into single `Compiled` primitive
4. JIT-compiles Metal Shading Language source at runtime
5. Caches in `library_kernels_[mtl_lib][hash_name]`

## Metal Backend

### Command Buffer Batching

Multiple dispatches batched into one command buffer. Auto-committed at thresholds:
- Phone: 20 ops or 40 MB
- Pro/Base GPU: 40 ops or 40 MB
- Max: 50 ops or 50 MB

### Synchronization

- **Intra-buffer**: `DispatchTypeConcurrent` + `memoryBarrier(BarrierScopeBuffers)` only when needed
- **Inter-encoder**: `MTL::Fence` per encoder, `waitForFence`/`updateFence` pattern
- All buffers: `MTLResourceStorageModeShared` + `HazardTrackingModeUntracked` (MLX handles all hazards)

## Quantized Operations

### Supported Formats

- **Affine**: int2, int3, int4, int5, int6, int8 with per-group scale+bias
- **FP**: FP4 E2M1, FP8 E4M3, FP8 E8M0 scales
- **Group sizes**: 32, 64, 128

### Kernel Selection (decode phase)

1. `qmv_quad`: Metal SIMD quadgroups, fastest M=1 decode path
2. `qmv_fast`: When `N % 8 == 0` and `K % 512 == 0`
3. `qmv`: General fallback
4. `gather_qmm_rhs`: **MoE-specific** -- gathers expert weights by index then quantized GEMM

### Steel GEMM

8x8 SIMD-group matrix fragments via `metal::simdgroup_matrix`. Tile sizes selected at dispatch time based on arch, dtype, problem shape. Used for dense matmul and prefill.

## MoE Support (mlx-lm)

### `SwitchLinear` / `QuantizedSwitchLinear`

Expert weights stored as 3D tensor `(num_experts, output_dims, input_dims)`. Routing done with `mx.gather_mm` (dense) or `mx.gather_qmm` (quantized).

### Token Sorting Optimization

When batch x seq >= 64 tokens, inputs sorted by expert index before GEMM (sequential access), then un-sorted. Dramatically improves memory efficiency.

## Memory Management

### `MetalAllocator`

- **Small allocations** (<4KB): Suballocated from `MTL::Heap`
- **Large allocations**: Direct `device_->newBuffer()` with `HazardTrackingModeUntracked`
- **GC pressure**: Evicts cached buffers at 95% of `recommendedMaxWorkingSetSize`
- **Residency sets**: `MTL::ResidencySet` to wire model weights into GPU memory

### KV Cache Quantization

`QuantizedKVCache` quantizes keys/values to 8-bit per step. Compresses cache 4-8x. Optional `quantized_kv_start` to keep first N tokens dense.

## No Disk Streaming

MLX has no per-layer streaming from disk. `mx.load()` with `lazy=True` mmap's the file, but actual usage triggers full Metal buffer allocation and copy. The entire weight must fit in DRAM.

This is the gap that the compressed cache addresses.

## Python to Metal Overhead

The measured overhead is <5% vs hand-tuned Metal for non-trivial kernels. The path:
1. Python to C++ pybind11 call (~100ns)
2. Graph node creation (~50ns)
3. Metal API calls (~1-5us each)
4. Command buffer batching amortizes to sub-microsecond per kernel
