# Integrations

MLX-Flash-Compress provides an OpenAI-compatible API server that works with any tool supporting custom endpoints.

## Quick Start

```bash
# Start the server
python -m mlx_flash_compress.serve \
  --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit \
  --port 8080 --preload
```

The server provides:
- `POST /v1/chat/completions` — OpenAI-compatible chat API
- `GET /v1/models` — list available models
- `GET /status` — memory, pressure, cache stats, optimization hints
- `GET /hints` — current optimization recommendations
- `GET /release` — trigger GPU memory release (when pressure is high)

## Claude Code

Claude Code can use our server as a local model provider for code completion and chat.

### Option 1: MCP Server (recommended)

Add to your Claude Code MCP configuration (`~/.claude/mcp_servers.json` or project `.claude/mcp_servers.json`):

```json
{
  "mlx-flash-compress": {
    "command": "python",
    "args": ["-m", "mlx_flash_compress.serve", "--model", "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit", "--port", "8080"],
    "env": {}
  }
}
```

Then in your Claude Code session, you can query the local model via the `/status` endpoint or use the chat API.

### Option 2: API Proxy

Start the server and use it as an alternative model endpoint:

```bash
# Terminal 1: Start MLX-Flash-Compress server
python -m mlx_flash_compress.serve --port 8080 --preload

# Terminal 2: Use with any OpenAI SDK script
export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=not-needed
```

### Option 3: Memory Monitor

Use the server's status endpoint to monitor memory during Claude Code sessions:

```bash
# Check memory status
curl -s http://localhost:8080/status | python -m json.tool

# Get optimization hints
curl -s http://localhost:8080/hints | python -m json.tool

# Release GPU memory when needed
curl -s http://localhost:8080/release | python -m json.tool
```

## OpenAI Codex / ChatGPT API

Any tool that speaks the OpenAI API can connect:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # local server, no auth
)

response = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to sort a list"},
    ],
    max_tokens=256,
)

print(response.choices[0].message.content)

# Check memory-specific metadata
if hasattr(response, "mlx_flash_compress"):
    print(f"Speed: {response.mlx_flash_compress['tok_per_s']} tok/s")
    print(f"Memory: {response.mlx_flash_compress['memory_pressure']}")
```

## LM Studio

1. Start the MLX-Flash-Compress server:
   ```bash
   python -m mlx_flash_compress.serve --port 8080 --preload
   ```

2. In LM Studio:
   - Go to **Settings** -> **Server**
   - Add custom endpoint: `http://localhost:8080/v1`
   - Select model: `local`
   - Chat normally

LM Studio natively uses `llama.cpp` for inference. Our server replaces that with MLX + memory-aware caching, which is better for MoE models on Apple Silicon.

## Ollama

Ollama uses `llama.cpp` as its backend. Two approaches:

### Run alongside Ollama

```bash
# Ollama on default port (11434)
ollama serve

# MLX-Flash-Compress on 8080
python -m mlx_flash_compress.serve --port 8080 --preload

# Use Ollama for dense models, our server for MoE models
```

### Use as Ollama-compatible endpoint

Our API is OpenAI-compatible, which Ollama clients also support. Any Ollama UI that allows custom endpoints can point to `http://localhost:8080/v1`.

## continue.dev (VS Code / JetBrains)

Add to `~/.continue/config.json`:

```json
{
  "models": [{
    "title": "Local MoE (MLX)",
    "provider": "openai",
    "model": "local",
    "apiBase": "http://localhost:8080/v1",
    "apiKey": "not-needed"
  }]
}
```

## Cursor

In Cursor settings, add a custom model:

- Provider: OpenAI Compatible
- API Base: `http://localhost:8080/v1`
- API Key: `not-needed`
- Model: `local`

## Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) works with any OpenAI-compatible endpoint:

```bash
# Start our server
python -m mlx_flash_compress.serve --port 8080 --preload

# In Open WebUI settings:
# Add connection: http://localhost:8080/v1
```

## Aider (AI pair programming)

```bash
# Start server
python -m mlx_flash_compress.serve --port 8080 --preload

# Use with aider
aider --openai-api-base http://localhost:8080/v1 --openai-api-key not-needed --model local
```

## Python SDK

```python
from mlx_flash_compress.serve import InferenceState

# Direct usage (no HTTP server needed)
state = InferenceState("mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit")
state.load_model()

# Generate
result = state.generate(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)
print(result["output"])
print(f"{result['tok_per_s']} tok/s")

# Check memory
status = state.get_status()
print(f"Memory pressure: {status['memory']['pressure']}")
print(f"Hints: {status['optimization_hints']}")
```

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat completion |
| `/v1/models` | GET | List available models |
| `/status` | GET | Full status: memory, pressure, cache, stats, hints |
| `/hints` | GET | Current optimization recommendations |
| `/release` | GET | Trigger GPU memory release when pressure is high |
| `/health` | GET | Health check (same as /status) |

### Status Response Example

```json
{
  "model": "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit",
  "hardware": {"chip": "Apple M3 Max", "ram_gb": 36.0},
  "memory": {
    "total_gb": 36.0,
    "free_gb": 8.5,
    "available_gb": 14.2,
    "pressure": "normal",
    "cache_budget_gb": 6.5,
    "swap_used_gb": 0.5
  },
  "stats": {"requests": 42, "tokens_generated": 8432, "uptime_s": 3600},
  "optimization_hints": [
    {
      "priority": "info",
      "action": "expand_cache",
      "message": "Plenty of RAM available. Cache could grow.",
      "detail": "Larger cache = higher hit rate = faster inference."
    }
  ]
}
```

### Optimization Hints

The server provides real-time optimization hints based on memory state:

| Priority | Action | When |
|----------|--------|------|
| `critical` | `reduce_cache` | Memory pressure critical, swap heavy |
| `critical` | `enable_mixed_precision` | Model barely fits, needs footprint reduction |
| `warning` | `shrink_cache` | Memory pressure warning |
| `warning` | `close_apps` | High swap usage (>2GB) |
| `info` | `expand_cache` | Plenty of free RAM, cache could grow |
| `info` | `enable_mixed_precision` | Small cache budget, MP would help |

## Memory Management

The server automatically:
- Monitors macOS memory pressure every 10 seconds
- Adjusts cache budget based on available RAM
- Warns when pressure is critical
- Auto-releases GPU memory pool when pressure spikes
- Provides actionable hints (which apps to close, what settings to change)

### Manual memory release

```bash
# When your Mac feels sluggish
curl http://localhost:8080/release
# Response: {"action": "released", "freed_gb": 0.5, "free_gb_now": 4.2}
```

## Ollama Custom Models

Ollama supports registering external models via a `Modelfile`. Point it at our running server so Ollama clients route requests to MLX-Flash-Compress instead of the built-in llama.cpp backend.

```bash
# Start our server first
python -m mlx_flash_compress.serve \
  --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit \
  --port 8080 --preload

# Create an Ollama Modelfile that proxies to our server
cat > /tmp/Modelfile <<'EOF'
FROM http://localhost:8080/v1
PARAMETER temperature 0.7
SYSTEM "You are a helpful assistant."
EOF

# Register the model in Ollama
ollama create mlx-moe -f /tmp/Modelfile

# Now use it through Ollama
ollama run mlx-moe "Explain MoE models"
```

Any Ollama client (Open WebUI, Enchanted, etc.) will now route `mlx-moe` requests through our MLX inference server with expert caching.

## llama.cpp Server

Our server can act as a drop-in front-end for llama.cpp's HTTP server (`llama-server`). Use this when you want MLX memory management and monitoring in front of a llama.cpp backend, or to give llama.cpp clients access to the `/status` and `/cache/stats` endpoints.

```bash
# Start llama.cpp server on a separate port
llama-server -m /path/to/model.gguf --port 8090

# Start our server pointing at llama.cpp as backend
# (or run both and route traffic based on model name)
python -m mlx_flash_compress.serve --port 8080 --preload

# Clients connect to :8080 (our server), which provides:
#   - Memory monitoring + optimization hints
#   - Expert cache statistics
#   - OpenAI-compatible streaming
```

For clients that already speak `llama.cpp`'s native `/completion` API, use an adapter:

```python
import httpx

# Forward to llama.cpp but wrap with our memory context
resp = httpx.post("http://localhost:8090/completion", json={
    "prompt": "Hello",
    "n_predict": 128,
    "stream": False,
})
# Check our server's memory status alongside
status = httpx.get("http://localhost:8080/status").json()
print(f"llama.cpp response: {resp.json()['content'][:80]}")
print(f"Memory pressure: {status['memory']['pressure']}")
```

## Homebrew / Binary Distribution

To build the Rust sidecar as a standalone binary and install it system-wide:

```bash
# Clone and enter the repo
git clone https://github.com/szibis/MLX-Flash-compress.git
cd MLX-Flash-compress

# Build the release binary (requires Rust toolchain)
cargo build --release -p mlx-flash-server

# The binary is at:
# ./mlx-flash-server/target/release/mlx-flash-server

# Install to /usr/local/bin
sudo cp mlx-flash-server/target/release/mlx-flash-server /usr/local/bin/

# Verify
mlx-flash-server --version
```

For Homebrew formula contributors, the binary can be tapped and distributed as:

```ruby
# Example Homebrew formula (community-maintained)
class MlxFlashServer < Formula
  desc "Rust sidecar for MLX-Flash-Compress — HTTP/SSE proxy + LCP expert cache"
  url "https://github.com/szibis/MLX-Flash-compress/archive/refs/tags/vX.Y.Z.tar.gz"
  # ...
  def install
    system "cargo", "build", "--release", "-p", "mlx-flash-server"
    bin.install "mlx-flash-server/target/release/mlx-flash-server"
  end
end
```

## Docker Compose

Run a two-container setup: the Rust sidecar handles HTTP and memory monitoring; the Python worker handles MLX inference.

```yaml
# docker-compose.yml
version: "3.9"

services:
  rust-sidecar:
    image: rust:1.78-slim
    build:
      context: .
      dockerfile: mlx-flash-server/Dockerfile
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
    command: >
      mlx-flash-server
        --port 8080
        --socket-path /run/mlx/cache.sock
        --cache-mb 512
    volumes:
      - mlx-socket:/run/mlx

  python-worker:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - FLASH_CACHE_RAM_MB=4096
      - MLX_SOCKET_PATH=/run/mlx/cache.sock
    command: >
      python -m mlx_flash_compress.serve
        --model mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit
        --socket-path /run/mlx/cache.sock
        --preload
    volumes:
      - mlx-socket:/run/mlx
      - huggingface-cache:/root/.cache/huggingface
    depends_on:
      - rust-sidecar

volumes:
  mlx-socket:
  huggingface-cache:
```

```bash
# Start both containers
docker compose up

# Inference goes through the Rust sidecar at :8080
curl http://localhost:8080/status
```

Note: MLX GPU inference requires native macOS with Apple Silicon. The Docker setup is primarily useful for CI testing with synthetic benchmarks, or for deploying the Rust sidecar alongside a native Python worker on the same host via the Unix socket bridge.

## Monitoring

The `/status` and `/cache/stats` endpoints expose metrics that can be scraped by Prometheus and visualized in Grafana.

### Prometheus Scrape Config

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: mlx_flash_compress
    static_configs:
      - targets: ["localhost:8080"]
    metrics_path: /status
    # The response is JSON; use json_exporter or a custom scraper
```

For a quick Grafana dashboard, use the JSON exporter sidecar to convert the `/status` JSON to Prometheus text format:

```bash
# Expose /status as Prometheus metrics on :9101
docker run -p 9101:9101 \
  -e URLS='http://host.docker.internal:8080/status' \
  quay.io/prometheuscommunity/json-exporter \
  --config.file=/etc/json_exporter/config.yml
```

### Key Metrics to Track

| Metric Path | Description |
|-------------|-------------|
| `memory.free_gb` | Free unified memory |
| `memory.pressure` | `normal` / `warning` / `critical` |
| `memory.swap_used_gb` | macOS swap usage (high = slowdown) |
| `stats.tokens_generated` | Cumulative tokens produced |
| `stats.requests` | Total inference requests |

Example Grafana panel query (using json_exporter labels):

```promql
# Tokens per second (rate over 1 minute)
rate(mlx_flash_stats_tokens_generated[1m])

# Memory pressure (alert when not "normal")
mlx_flash_memory_free_gb < 4
```

### Cache Stats Endpoint

```bash
curl -s http://localhost:8080/cache/stats | python -m json.tool
# {
#   "entries": 142,
#   "hit_rate": 0.81,
#   "evictions": 23,
#   "size_mb": 487
# }
```

## Programmatic Python Usage

Use `RustCacheClient` directly for fine-grained expert weight management from Python, bypassing the HTTP layer entirely.

```python
from mlx_flash_compress.rust_cache_client import RustCacheClient
import mlx.core as mx

# Connect to the Rust sidecar over the Unix socket
client = RustCacheClient(socket_path="/tmp/mlx-flash-cache.sock")
client.connect()

# Pre-warm specific expert layers before inference
expert_ids = [0, 3, 7, 12]  # experts predicted by router
client.prefetch(expert_ids, priority="high")

# Check cache state
stats = client.stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
print(f"Cached experts: {stats['entries']}")

# Stream a specific expert weight tensor from the Rust cache
weight = client.get_expert(layer=4, expert_id=7)
if weight is None:
    # Cache miss — load from disk and populate cache
    weight = mx.load(f"experts/layer4_expert7.npz")["weight"]
    client.put_expert(layer=4, expert_id=7, weight=weight)

# Explicit eviction (e.g., before a topic switch)
client.evict(expert_ids=[0, 3], reason="topic_switch")

# Close when done
client.close()
```

This interface is intended for advanced use cases such as:
- Custom router integration that feeds predicted expert IDs to the cache ahead of time
- Manual cache warming for known workloads (e.g., always pre-load coding experts at startup)
- Benchmarking cache strategies without going through the full inference stack
