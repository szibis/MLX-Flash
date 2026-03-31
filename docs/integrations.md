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
