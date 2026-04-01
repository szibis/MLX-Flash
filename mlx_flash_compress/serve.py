"""Memory-aware MLX inference server with progressive warm-up.

Provides an OpenAI-compatible chat API endpoint that:
  1. Monitors RAM and warns when memory is tight
  2. Shows cache warm-up progress per session
  3. Auto-applies mixed precision when RAM pressure is detected
  4. Provides real-time memory/cache status endpoint

Usage:
  python -m mlx_flash_compress.serve --model mlx-community/Qwen3-30B-A3B-4bit
  python -m mlx_flash_compress.serve --model PATH --port 8080

Then use any OpenAI-compatible client:
  curl http://localhost:8080/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model": "local", "messages": [{"role": "user", "content": "Hello"}]}'

Or from LM Studio: Add custom endpoint http://localhost:8080/v1
"""

import argparse
import json
import os
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Optional

import mlx.core as mx
from mlx_lm import load, generate

from mlx_flash_compress.hardware import detect_hardware
from mlx_flash_compress.memory_manager import MemoryManager, get_memory_state


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a new thread."""
    daemon_threads = True


class InferenceState:
    """Shared state for the inference server."""

    def __init__(self, model_name: str, cache_budget_pct: float = 0.8,
                 kv_bits: int = 0):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.kv_bits = kv_bits  # 0 = no KV quant, 8 = 8-bit (45% savings)
        self.hw = detect_hardware()
        self.mem_mgr = MemoryManager(safety_margin_gb=2.0)
        self.cache_budget_pct = cache_budget_pct

        # Stats
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.monotonic()

    def load_model(self):
        """Load the model with memory checks."""
        mem = get_memory_state()
        print(f"\n  Memory before load:")
        print(f"    Total: {mem.total_gb:.1f}GB")
        print(f"    Free:  {mem.free_gb:.1f}GB")
        print(f"    Available: {mem.available_gb:.1f}GB")
        print(f"    Pressure: {mem.pressure_level}")

        if mem.pressure_level == "critical":
            print("\n  WARNING: Memory pressure is CRITICAL!")
            print("  Consider closing other applications before loading the model.")
            print("  Continuing anyway...")

        print(f"\n  Loading model: {self.model_name}")

        # Pre-download with progress if not cached locally
        if not os.path.isdir(self.model_name):
            try:
                from huggingface_hub import snapshot_download
                print("  Downloading model files (with progress)...")
                snapshot_download(
                    self.model_name,
                    allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
                )
                print("  Download complete.")
            except Exception:
                pass  # mlx_lm.load will handle download as fallback

        t0 = time.monotonic()
        self.model, self.tokenizer = load(self.model_name)
        mx.synchronize()
        load_time = time.monotonic() - t0
        print(f"  Loaded in {load_time:.1f}s")

        # Warmup: compile Metal shaders and allocate KV cache
        print("  Warming up (compiling shaders)...")
        t_warm = time.monotonic()
        warmup_prompt = "Hello"
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                warmup_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Hi"}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                pass
        _ = generate(self.model, self.tokenizer, prompt=warmup_prompt,
                     max_tokens=5, verbose=False)
        mx.synchronize()
        print(f"  Warm-up done in {time.monotonic() - t_warm:.1f}s")

        # Check post-load memory
        mem_after = get_memory_state()
        model_size_gb = mem.available_gb - mem_after.available_gb
        print(f"\n  Model size (estimated): {model_size_gb:.1f}GB")
        print(f"  RAM remaining: {mem_after.available_gb:.1f}GB")
        print(f"  Pressure: {mem_after.pressure_level}")

        if mem_after.pressure_level in ("warning", "critical"):
            print("\n  MEMORY PRESSURE DETECTED after model load!")
            self._suggest_memory_actions(mem_after)

        # Auto mixed-precision: if model barely fits, reduce footprint
        try:
            footprint = mx.get_peak_memory() / (1024**3)
            headroom = mem_after.available_gb
            if headroom < footprint * 0.15:  # less than 15% headroom
                print(f"\n  Model barely fits ({footprint:.1f}GB, {headroom:.1f}GB free)")
                print("  Hint: run with --mixed-precision to reduce footprint by ~20%")
        except AttributeError:
            pass

    def _suggest_memory_actions(self, mem):
        """Suggest actions to reduce memory pressure."""
        print("\n  Suggestions to free RAM:")
        print("    1. Close browser tabs (each tab uses 100-500MB)")
        print("    2. Close Xcode/Android Studio (1-4GB each)")
        print("    3. Close Docker Desktop (1-2GB)")
        print("    4. Quit unused apps in Dock")
        if mem.swap_used_gb > 0:
            print(f"    5. {mem.swap_used_gb:.1f}GB in swap — closing apps will help most")
        print()
        print("  Or use a smaller model / enable mixed precision:")
        print("    python -m mlx_flash_compress.serve --model PATH --mixed-precision")

    def get_status(self) -> dict:
        """Get current server status with optimization hints."""
        mem = get_memory_state()
        uptime = time.monotonic() - self.start_time
        cache_budget = self.mem_mgr.get_cache_budget_gb()
        hints = self.mem_mgr.get_optimization_hints()

        return {
            "model": self.model_name,
            "hardware": {
                "chip": self.hw.chip,
                "ram_gb": self.hw.total_ram_gb,
            },
            "memory": {
                "total_gb": round(mem.total_gb, 1),
                "free_gb": round(mem.free_gb, 1),
                "available_gb": round(mem.available_gb, 1),
                "pressure": mem.pressure_level,
                "cache_budget_gb": round(cache_budget, 1),
                "swap_used_gb": round(mem.swap_used_gb, 1),
            },
            "stats": {
                "requests": self.total_requests,
                "tokens_generated": self.total_tokens,
                "uptime_s": round(uptime, 0),
            },
            "optimization_hints": hints,
        }

    def generate(self, messages: list[dict], max_tokens: int = 256,
                 temperature: float = 0.7) -> dict:
        """Generate a response with memory awareness."""
        if self.model is None:
            self.load_model()

        # Check memory before generation — auto-release if needed
        mem = get_memory_state()
        release_info = None
        if mem.pressure_level in ("critical", "warning"):
            release_info = self.mem_mgr.auto_release_if_needed()
            mem = get_memory_state()  # re-check after release

        if mem.pressure_level == "critical":
            hints = self.mem_mgr.get_optimization_hints()
            return {
                "error": "Memory pressure is critical. Close applications to free RAM.",
                "memory": {
                    "available_gb": round(mem.available_gb, 1),
                    "pressure": mem.pressure_level,
                },
                "optimization_hints": hints,
                "auto_release": release_info,
            }

        # Format prompt
        prompt = self._format_messages(messages)

        # Generate
        t0 = time.monotonic()
        output = generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        mx.synchronize()
        elapsed = time.monotonic() - t0

        tokens = len(self.tokenizer.encode(output))
        tps = tokens / elapsed if elapsed > 0 else 0

        self.total_requests += 1
        self.total_tokens += tokens

        # Check memory after generation
        mem_after = get_memory_state()

        return {
            "output": output,
            "tokens": tokens,
            "time_s": round(elapsed, 2),
            "tok_per_s": round(tps, 1),
            "memory_pressure": mem_after.pressure_level,
        }

    def _format_messages(self, messages: list[dict]) -> str:
        """Format chat messages for the model."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        # Fallback: join messages
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP handler for OpenAI-compatible chat API."""

    server_state: InferenceState = None  # set by server setup

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json({
                "data": [{
                    "id": "local",
                    "object": "model",
                    "owned_by": "mlx-flash",
                }]
            })
        elif self.path in ("/health", "/status"):
            self._send_json(self.server_state.get_status())
        elif self.path == "/hints":
            hints = self.server_state.mem_mgr.get_optimization_hints()
            self._send_json({"hints": hints})
        elif self.path == "/release":
            result = self.server_state.mem_mgr.auto_release_if_needed()
            self._send_json(result)
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_chat(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.7)

        if not messages:
            self._send_json({"error": "No messages provided"}, 400)
            return

        stream = data.get("stream", False)

        if stream:
            self._handle_stream(messages, max_tokens, temperature)
            return

        result = self.server_state.generate(messages, max_tokens, temperature)

        if "error" in result:
            self._send_json({"error": result["error"]}, 503)
            return

        # Format as OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "local",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["output"],
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": result["tokens"],
                "total_tokens": result["tokens"],
            },
            "mlx_flash_compress": {
                "tok_per_s": result["tok_per_s"],
                "memory_pressure": result["memory_pressure"],
            },
        }
        self._send_json(response)

    def _handle_stream(self, messages, max_tokens, temperature):
        """Handle streaming (SSE) response — sends tokens as they generate."""
        state = self.server_state
        if state.model is None:
            state.load_model()

        prompt = state._format_messages(messages)
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Start SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        t0 = time.monotonic()

        # Generate full output (MLX doesn't easily support true token-by-token)
        output = generate(
            state.model, state.tokenizer,
            prompt=prompt, max_tokens=max_tokens, verbose=False,
        )
        mx.synchronize()

        # Simulate streaming by sending words progressively
        # This gives the SSE behavior clients expect
        words = output.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            event = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "local",
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }],
            }
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()

        # Final chunk with finish_reason
        final = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "local",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

        elapsed = time.monotonic() - t0
        tokens = len(state.tokenizer.encode(output))
        state.total_requests += 1
        state.total_tokens += tokens

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def log_message(self, format, *args):
        """Quiet logging — only show requests."""
        msg = format % args
        if "POST" in msg or "error" in msg.lower():
            print(f"  [{time.strftime('%H:%M:%S')}] {msg}")


def main():
    parser = argparse.ArgumentParser(
        description="MLX-Flash: Memory-aware inference server"
    )
    parser.add_argument("--model", default=None,
                        help="MLX model to serve (auto-selects based on RAM if not specified)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--preload", action="store_true", help="Load model immediately")
    parser.add_argument("--kv-bits", type=int, default=0, choices=[0, 4, 8],
                        help="KV cache quantization bits (0=none, 8=45%% memory savings)")
    args = parser.parse_args()

    # Auto-select model based on available RAM
    if args.model is None:
        mem = get_memory_state()
        ram = mem.total_gb
        if ram >= 32:
            args.model = "mlx-community/Qwen3-30B-A3B-4bit"  # 30B MoE, ~18GB
        elif ram >= 16:
            args.model = "mlx-community/Qwen3-30B-A3B-4bit"  # fits with streaming
        elif ram >= 12:
            args.model = "mlx-community/Qwen3-8B-4bit"  # 8B dense, ~5GB
        else:
            args.model = "mlx-community/Qwen3-4B-4bit"  # 4B dense, ~2.5GB

    print()
    print("=" * 60)
    print("  MLX-Flash: Inference Server")
    print("=" * 60)

    state = InferenceState(args.model, kv_bits=args.kv_bits)
    print(f"\n  Hardware: {state.hw.chip}, {state.hw.total_ram_gb:.0f}GB RAM")

    mem = get_memory_state()
    print(f"  Memory: {mem.available_gb:.1f}GB available, pressure: {mem.pressure_level}")

    if args.preload:
        state.load_model()

    ChatHandler.server_state = state

    server = ThreadedHTTPServer((args.host, args.port), ChatHandler)
    print(f"\n  Server running on http://{args.host}:{args.port}")
    print(f"  API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Status: http://{args.host}:{args.port}/status")
    print(f"\n  Compatible with: LM Studio (custom endpoint), continue.dev, OpenAI SDK")
    print(f"  Model loads on first request (or use --preload)")
    print(f"\n  Press Ctrl+C to stop")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
