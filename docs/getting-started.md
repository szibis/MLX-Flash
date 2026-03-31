# Getting Started

## Prerequisites

- **Mac with Apple Silicon** (M1, M2, M3, or M4 — any variant)
- **macOS 14+** (Sonoma or newer)
- **Python 3.10+**
- At least **16GB RAM** (more = better performance)

## Installation (2 minutes)

```bash
# Clone the repo
git clone https://github.com/szibis/MLX-Flash-compress.git
cd MLX-Flash-compress

# Create virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install lz4 zstandard numpy psutil tabulate pytest mlx mlx-lm

# Build C acceleration library (optional but recommended)
make -C csrc install
```

## Your First Run

### 1. Check your hardware

```bash
python -m mlx_flash_compress.hardware
```

This shows your Mac's specs and what models you can run:

```
  Detected: Apple M3 Max, 36GB RAM, 1TB SSD

  Model                           Fits?  Hit%   tok/s
  Qwen MoE (5GB)                  YES    100%   115
  Mixtral-8x7B (26GB)             YES    100%    16
  DeepSeek-V3 (170GB)              NO     68%   3.7
```

### 2. Run with a model

```bash
# Small model (downloads ~5GB, fits in RAM)
python -m mlx_flash_compress.run \
  --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit \
  --tokens 100

# With task-specific optimization
python -m mlx_flash_compress.run \
  --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit \
  --task coding \
  --tokens 100

# With adaptive profiling (learns what you need)
python -m mlx_flash_compress.run \
  --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit \
  --adaptive \
  --tokens 200
```

### 3. Find your optimal configuration

```bash
# For a specific model size on your hardware
python -m mlx_flash_compress.tier_optimizer \
  --total-ram 36 --model-gb 209 --layers 60 --experts 512

# Output: optimal RAM/SSD split, expected tok/s, cache hit rate
```

## Configuration

### Quick: Environment variables

```bash
# Set cache size (MB)
export FLASH_CACHE_RAM_MB=8192

# Enable/disable features
export FLASH_ENABLE_PREFETCH=1
export FLASH_MIXED_PRECISION=1
export FLASH_SKIP_FALLBACK=0

python -m mlx_flash_compress.run --model <path>
```

### Full: Config file

Create `~/.config/mlx-flash-compress/config.json`:

```json
{
  "cache": {
    "enable": true,
    "ram_mb": 0,
    "eviction": "lcp",
    "hot_algo": "lz4"
  },
  "prefetch": {
    "enable": true,
    "workers": 2
  },
  "mixed_precision": {
    "enable": true,
    "cold_bits": 2,
    "hot_bits": 4
  },
  "skip_fallback": {
    "enable": false
  },
  "ssd_protection": {
    "enable": true,
    "thermal_limit_c": 70
  },
  "engine": {
    "backend": "auto"
  }
}
```

Set `ram_mb` to `0` for auto-detection (uses 80% of available memory with safety margin).

## Running Tests

```bash
python -m pytest tests/ -v
# Expected: 43 passed
```

## Docker (for CI/testing only)

```bash
docker build -t mlx-flash-compress .
docker run mlx-flash-compress
# Runs synthetic benchmarks (MLX inference requires native macOS)
```

## Troubleshooting

**"MLX not available"**: You need Apple Silicon Mac. Intel Macs don't support MLX.

**"Model download fails"**: Set `HF_TOKEN` environment variable for Hugging Face authentication:
```bash
export HF_TOKEN=hf_your_token_here
```

**"libfastcache.dylib not found"**: Build it:
```bash
make -C csrc install
```

**"Out of memory"**: Reduce cache size:
```bash
python -m mlx_flash_compress.run --model <path> --cache-mb 2048
```

## What's Next

- Try different models to see scaling behavior
- Use `--task coding` or `--task writing` for task-specific optimization
- Run `python -m mlx_flash_compress.tier_optimizer` to find optimal settings
- Check `docs/technical-reference.md` for deep implementation details
