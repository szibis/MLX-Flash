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
# Expected: 89+ passed
```

## Rust Sidecar (optional, for production)

The Rust sidecar provides faster memory monitoring, SSE streaming, and expert caching.

### Build

```bash
cargo build --release -p mlx-flash-server
```

### Run

```bash
./mlx-flash-server/target/release/mlx-flash-server --launch-worker --preload --port 8080
```

### With expert caching

```bash
./mlx-flash-server/target/release/mlx-flash-server \
  --launch-worker --preload \
  --expert-dir /path/to/experts \
  --cache-mb 512 \
  --socket-path /tmp/mlx-flash-cache.sock
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

## Interactive Chat

The simplest way to use MLX-Flash-Compress:

```bash
python -m mlx_flash_compress.chat
```

Shows real-time memory status, tok/s per response, and warns when RAM is tight. Type `/status` to see memory info, `/clear` to reset conversation.

## API Server (LM Studio, continue.dev, OpenAI SDK)

Start the OpenAI-compatible API server:

```bash
python -m mlx_flash_compress.serve --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit --port 8080
```

### Connect from LM Studio

1. Open LM Studio
2. Go to Settings -> Server
3. Set custom endpoint: `http://localhost:8080/v1`
4. Chat normally — our server handles inference + memory management

### Connect from continue.dev (VS Code)

Add to your `~/.continue/config.json`:

```json
{
  "models": [{
    "title": "Local MoE",
    "provider": "openai",
    "model": "local",
    "apiBase": "http://localhost:8080/v1",
    "apiKey": "not-needed"
  }]
}
```

### Connect from any OpenAI SDK client

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Server endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat API (OpenAI-compatible) |
| `/v1/models` | GET | List available models |
| `/status` | GET | Memory, pressure, cache stats |
| `/health` | GET | Health check |

### Using with Ollama

Ollama uses `llama.cpp` as its backend, not MLX. Two options:

1. **Run our server alongside**: Our API server at `:8080`, Ollama at `:11434`. Use our server for MoE models that benefit from expert caching.
2. **Ollama with MLX backend**: If Ollama adds MLX support in the future, our memory management layer can integrate.

## Memory Management

The system automatically monitors your Mac's RAM:

```bash
# Check memory status anytime during chat
/status

# Or via the API
curl http://localhost:8080/status
```

**What it does:**
- Monitors macOS memory pressure in real-time
- Auto-sizes expert cache based on available RAM (2GB safety margin)
- Warns when pressure is critical ("close apps to prevent slowdown")
- Suggests actions: which apps to close, whether to use a smaller model

**For models that barely fit in RAM (the sweet spot):**

Mixed precision automatically reduces the model's memory footprint by ~20%:
- Hot experts stay at 4-bit (full quality)
- Cold experts compressed to 2-bit (minimal quality impact)
- Result: a model at 0.9x RAM goes from 43 tok/s -> 104 tok/s (measured)

## Benchmarks

```bash
# Memory pressure analysis (the key measurement)
python -m mlx_flash_compress.bench_memory_pressure --tokens 50

# ISP-like warm-up demo (watch cache fill in real-time)
python -m mlx_flash_compress.demo_warmup --topics coding writing coding math

# Real model routing with cache simulation
python -m mlx_flash_compress.cached_inference --tokens 80 --multi-topic
```

## What's Next

- Try different models to see scaling behavior
- Use `--task coding` or `--task writing` for task-specific optimization
- Run `python -m mlx_flash_compress.tier_optimizer` to find optimal settings
- Check `docs/integrations.md` for Claude Code, LM Studio, Cursor, Aider integration
- Check `docs/technical-reference.md` for deep implementation details
