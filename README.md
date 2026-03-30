# MLX-Flash-Compress

Tiered compressed expert cache for Mixture-of-Experts inference on Apple Silicon. Combines MLX's ML framework with Flash-MoE's SSD streaming philosophy, adding CPU-parallel compressed caching as the missing middle layer.

## The Problem

Large MoE models (Qwen3-397B, Mixtral-8x22B) have hundreds of GB of expert weights that don't fit in RAM. [Flash-MoE](https://github.com/danveloper/flash-moe) demonstrated that streaming expert weights from NVMe SSD at 4.36 tok/s on a 48GB MacBook is viable — but **56% of per-layer time is spent waiting for SSD reads** (2.41ms per layer). Meanwhile, the CPU's 12+ performance cores sit idle during inference.

## The Solution

Add a compressed expert cache between the MLX interface and SSD storage:

```
MLX Interface (Python)     ← Clean API, quantized KV cache, graph compilation
       |
Compressed Expert Cache    ← LZ4/ZSTD in RAM, frequency-aware eviction
       |
SSD Engine (Cold path)     ← Direct NVMe reads for cache misses
```

Cache hits decompress from RAM (~0.02ms per expert with LZ4) instead of reading from SSD (~0.6ms per expert). With 3-4x compression ratio on quantized weights, the same RAM holds 3-4x more experts.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  HOT TIER (LZ4 compressed in RAM)                           │
│  - Fastest decompress: ~25 GB/s on Apple Silicon            │
│  - Ratio on quantized weights: ~1.6x (real) to ~3.7x (Q4)  │
│  - Frequency-aware LFU eviction                             │
├─────────────────────────────────────────────────────────────┤
│  WARM TIER (ZSTD compressed in RAM)                         │
│  - Better ratio: ~2.0-2.4x on quantized weights             │
│  - Slower decompress: ~1.4 GB/s                             │
│  - For infrequently-accessed experts                        │
├─────────────────────────────────────────────────────────────┤
│  COLD TIER (SSD)                                            │
│  - Direct NVMe pread, same as Flash-MoE                     │
│  - Async cache population: compress in background thread    │
│  - F_NOCACHE bypass for benchmark fairness                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Fast-path cache hits**: Cache lookups are resolved synchronously in the calling thread — no thread pool overhead. Only cold SSD reads use the thread pool for parallel I/O.

2. **Async cache population**: When a cold miss occurs, the raw data is returned immediately. Compression + cache insertion happens asynchronously in a background thread pool, so the caller is never blocked on compression.

3. **Pre-warming**: For production use, experts are pre-compressed at model download time. The cache starts warm with zero cold-start penalty.

4. **Apple native compression**: Supports `libcompression` via ctypes for LZFSE (Apple-proprietary, excellent ratio on Apple Silicon) and native LZ4 in addition to Python C extensions.

## Benchmark Results

**Hardware**: Apple Silicon Mac (results scale with SSD speed and core count)

### Compression Ratios (synthetic 4-bit quantized data)

| Algorithm | Ratio | Compress MB/s | Decompress MB/s |
|-----------|-------|---------------|-----------------|
| LZ4 (Python C ext) | 3.72x | 3,583 | 25,783 |
| ZSTD-1 | 7.56x | 1,351 | 1,406 |
| ZSTD-3 | 7.97x | 1,259 | 1,307 |
| LZFSE (Apple native) | 6.71x | 251 | 908 |
| LZ4_RAW (Apple native) | 3.63x | 2,025 | 19,852 |

LZ4 wins decisively on decompression speed (25 GB/s vs ~1.4 GB/s for ZSTD). ZSTD and LZFSE win on ratio. For hot-tier cache where speed matters most, LZ4 is the clear choice.

> **Note**: Real quantized weights (GGUF Q4_K_M) typically compress ~1.5-1.7x with LZ4. Our synthetic data achieves higher ratios due to the archetype-based generation. The architecture's value scales with compression ratio.

### Cache Performance (pre-warmed, simulated SSD latency)

| Scenario | SSD tok/s | LZ4 Cache tok/s | Speedup | Cache Hit Rate |
|----------|-----------|-----------------|---------|----------------|
| OS page cache (warm) | 194.0 | 160.8 | 0.83x | 40% |
| NVMe cold read (0.6ms/2MB) | 182.1 | 111.7 | 0.61x | 57% |
| NVMe + unified mem contention (1.5ms) | 90.5 | 91.5 | **1.01x** | 71% |
| Flash-MoE calibrated (2.4ms) | 59.0 | 80.6 | **1.37x** | 78% |

### When the Cache Wins

The compressed cache breaks even at ~1.5ms SSD latency per 2MB and wins decisively at Flash-MoE's measured latency (2.4ms per 4 experts). This corresponds to the **real-world scenario**: models that significantly exceed RAM, where the OS page cache can't absorb the working set and every expert access hits the NVMe.

```
SSD latency (ms/2MB)    0     0.5     1.0     1.5     2.0     2.5
                        |-------|-------|-------|-------|-------|
Cache overhead wins:    <-- SSD faster --| breakeven |-- Cache faster -->
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/szibis/MLX-Flash-compress.git
cd MLX-Flash-compress
uv venv && source .venv/bin/activate
uv pip install lz4 zstandard numpy psutil tabulate pytest

# Run synthetic benchmark (no model download needed)
python -m mlx_flash_compress.bench --synthetic

# Run with larger experts and more tokens
python -m mlx_flash_compress.bench --synthetic --layers 16 --experts 64 --expert-kb 2048 --tokens 50

# Run with a real MLX MoE model (downloads model)
python -m mlx_flash_compress.bench --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit

# Run tests
python -m pytest tests/ -v
```

### Benchmark Options

```
--synthetic          Run synthetic benchmarks (no model needed)
--model MODEL        MLX MoE model name for real inference
--layers N           Number of MoE layers (synthetic, default: 8)
--experts N          Experts per layer (synthetic, default: 64)
--expert-kb N        Expert size in KB (synthetic, default: 256)
--tokens N           Tokens to generate/simulate (default: 50)
--hot-mb N           Hot tier cache size in MB (default: 256)
--warm-mb N          Warm tier cache size in MB (default: 128)
--workers N          Parallel decompression workers (default: 4)
```

## API Usage

```python
from mlx_flash_compress.cache import ExpertCacheManager
import numpy as np

# Create cache with 2GB hot + 1GB warm tiers
cache = ExpertCacheManager(
    expert_dir="path/to/expert_weights/",
    hot_limit_bytes=2 * 1024**3,
    warm_limit_bytes=1 * 1024**3,
    num_workers=4,
    hot_algo="lz4",  # or "lzfse" for Apple native
)

# Optional: pre-warm cache from disk
cache.prewarm(num_layers=60, num_experts=512)

# Fetch experts (fast-path for cache hits, parallel SSD for misses)
results = cache.fetch_experts(
    layer_idx=5,
    expert_ids=[12, 45, 200, 387],
    expert_dtype=np.float16,
)

for weights, tier in results:
    print(f"Tier: {tier.name}, Shape: {weights.shape}")

# Check cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Hot: {stats.hot_hits}, Warm: {stats.warm_hits}, Cold: {stats.cold_hits}")
```

## Project Structure

```
mlx_flash_compress/
    __init__.py              # Package exports
    compression.py           # LZ4/ZSTD backends (Python C extensions)
    compression_native.py    # Apple libcompression (LZFSE, native LZ4)
    cache.py                 # Tiered ExpertCacheManager (core)
    engine.py                # MLX model wrapper and inference modes
    bench.py                 # Benchmark harness
tests/
    test_compression.py      # Compression roundtrip and ratio tests
    test_cache.py            # Cache hit/miss/eviction tests
```

## Findings and Insights

### What We Learned

1. **Python ThreadPoolExecutor overhead dominates small tasks.** Each future submit/collect costs ~0.05ms, while LZ4 decompress of 550KB takes ~0.02ms. The synchronous fast-path for cache hits was essential — bypassing the thread pool for cached data gave a 2-3x improvement.

2. **LZ4 cannot compress packed nibble data.** 4-bit quantized weights packed as nibble pairs look random at the byte level. LZ4's hash-based match finder needs 4-byte repeating sequences. Real compressibility comes from block-level structure (repeated scales, dead neurons, row similarity).

3. **Compression-on-insert kills cold path performance.** Naively compressing expert data when inserting into the cache makes every cold miss pay a 0.5ms+ compress penalty. Async background population is mandatory.

4. **The architecture works when SSD is the real bottleneck.** The cache breaks even at ~1.5ms/2MB SSD latency and wins at Flash-MoE's measured 2.4ms. For models that fit in the OS page cache, the cache adds overhead.

5. **LZFSE is not faster than LZ4.** Despite being Apple's native format, LZFSE compresses at only 251 MB/s vs LZ4's 3,583 MB/s. It's designed for better ratio, not speed. For hot-tier caching, LZ4 is the clear winner.

### What Would Move the Needle

For a **production C implementation** (like extending Flash-MoE):

- **GCD dispatch** instead of Python ThreadPoolExecutor: <1us overhead vs ~50us
- **Pre-compressed expert files**: Ship models with LZ4-compressed experts, eliminating compress-on-insert entirely
- **Metal buffer integration**: Decompress directly into 2MB-aligned `MTLBuffer` via `newBufferWithBytesNoCopy`
- **Pipeline overlap**: Schedule CPU decompression during GPU attention compute (1.77ms GPU window)

Projected improvement with C implementation: **1.5-2x** over Python prototype at Flash-MoE latency levels.

## Relationship to Flash-MoE and MLX

| Component | Flash-MoE | MLX | This Project |
|-----------|-----------|-----|--------------|
| Expert streaming from SSD | Hand-tuned pread + GCD | Not supported | Inherits Flash-MoE's approach |
| GPU compute | Hand-written Metal shaders | Steel GEMM + auto-fusion | Uses MLX for non-expert compute |
| Expert cache | Rejected (no compression) | N/A | LZ4/ZSTD tiered cache (new) |
| KV cache | Raw f32 | Quantized 8-bit | Uses MLX's quantized cache |
| Interface | C/Objective-C | Python (NumPy-like) | Python with C compression |

The compressed cache is the **missing middle layer** that Flash-MoE couldn't make work (because they only tested uncompressed caching) and MLX doesn't need (because it assumes models fit in RAM).

## License

MIT
