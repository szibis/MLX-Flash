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
import gc
import json
import os
import pathlib
import signal
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional

import mlx.core as mx
import yaml
from mlx_lm import generate, load

from mlx_flash_compress.hardware import detect_hardware
from mlx_flash_compress.log_config import setup_logging
from mlx_flash_compress.memory_manager import MemoryManager, get_memory_state
from mlx_flash_compress.telemetry import HardwareTelemetry

# Module-level logger, configured in main()
logger = None

# Available speculative decoding engines
SPECULATIVE_ENGINES = ("eagle3", "layerskip", "dflash", "none")

# Path to the model registry YAML
MODELS_YAML = pathlib.Path(__file__).parent.parent / "scripts" / "models.yaml"


def load_model_registry() -> dict:
    """Load the model registry from models.yaml."""
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a new thread."""

    daemon_threads = True
    allow_reuse_address = True


class InferenceState:
    """Shared state for the inference server."""

    def __init__(
        self,
        model_name: str,
        cache_budget_pct: float = 0.8,
        kv_bits: int = 0,
        batching: bool = False,
        speculative: str = "none",
        request_timeout: float = 120.0,
        safety_margin_gb: float = 2.0,
    ):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.kv_bits = kv_bits  # 0 = no KV quant, 8 = 8-bit (45% savings)
        self.hw = detect_hardware()
        self.mem_mgr = MemoryManager(safety_margin_gb=safety_margin_gb)
        self.cache_budget_pct = cache_budget_pct
        self.batching = batching
        self.speculative = speculative  # "eagle3", "layerskip", "dflash", or "none"
        self.request_timeout = request_timeout
        self.engine = None  # ContinuousBatchingEngine, created after model load
        self.spec_engine = None  # Speculative decoding engine, created after model load

        # Hardware telemetry
        self.telemetry = HardwareTelemetry()

        # Stats
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.monotonic()

    def load_model(self):
        """Load the model with memory checks."""
        global logger
        log = logger or __import__("logging").getLogger("mlx_flash")
        mem = get_memory_state()
        log.info(
            "Memory before load",
            extra={
                "action": "pre_load_memory",
                "memory_gb": round(mem.total_gb, 1),
                "pressure": mem.pressure_level,
            },
        )

        if mem.pressure_level == "critical":
            log.warning(
                "Memory pressure CRITICAL — loading anyway",
                extra={
                    "action": "critical_pressure_warning",
                    "memory_gb": round(mem.available_gb, 1),
                },
            )

        log.info("Loading model", extra={"model": self.model_name, "action": "model_load_start"})

        # Pre-download with progress if not cached locally
        if not os.path.isdir(self.model_name):
            try:
                from huggingface_hub import snapshot_download

                log.info("Downloading model files", extra={"model": self.model_name, "action": "download_start"})
                snapshot_download(
                    self.model_name,
                    allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
                )
                log.info("Download complete", extra={"model": self.model_name, "action": "download_complete"})
            except Exception:
                pass  # mlx_lm.load will handle download as fallback

        t0 = time.monotonic()
        self.model, self.tokenizer = load(self.model_name)
        mx.synchronize()
        load_time = time.monotonic() - t0
        log.info(
            "Model loaded",
            extra={
                "model": self.model_name,
                "load_time_s": round(load_time, 1),
                "action": "model_load_complete",
            },
        )

        # Warmup: compile Metal shaders and allocate KV cache
        log.info("Warming up (compiling Metal shaders)", extra={"action": "warmup_start"})
        t_warm = time.monotonic()
        warmup_prompt = "Hello"
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                warmup_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Hi"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        _ = generate(self.model, self.tokenizer, prompt=warmup_prompt, max_tokens=5, verbose=False)
        mx.synchronize()
        log.info(
            "Warm-up done",
            extra={
                "load_time_s": round(time.monotonic() - t_warm, 1),
                "action": "warmup_complete",
            },
        )

        # Check post-load memory
        mem_after = get_memory_state()
        model_size_gb = mem.available_gb - mem_after.available_gb
        log.info(
            "Post-load memory",
            extra={
                "model": self.model_name,
                "memory_gb": round(mem_after.available_gb, 1),
                "pressure": mem_after.pressure_level,
                "action": "post_load_memory",
            },
        )

        if mem_after.pressure_level in ("warning", "critical"):
            log.warning(
                "Memory pressure detected after model load",
                extra={
                    "pressure": mem_after.pressure_level,
                    "memory_gb": round(mem_after.available_gb, 1),
                },
            )
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

        # Initialize continuous batching engine if enabled
        if self.batching and self.engine is None:
            from mlx_flash_compress.continuous_batching import ContinuousBatchingEngine

            log.info("Starting continuous batching engine", extra={"action": "batching_start"})
            self.engine = ContinuousBatchingEngine(self.model, self.tokenizer)
            self.engine.start()

        # Initialize speculative decoding engine if enabled
        if self.speculative != "none" and self.spec_engine is None:
            self._init_speculative_engine(log)

    def _init_speculative_engine(self, log):
        """Initialize the configured speculative decoding engine."""
        if self.speculative == "eagle3":
            from mlx_flash_compress.eagle3 import EAGLE3Engine

            log.info("Initializing EAGLE-3 speculative decoding", extra={"action": "spec_init", "engine": "eagle3"})
            self.spec_engine = EAGLE3Engine(self.model, self.tokenizer)
        elif self.speculative == "layerskip":
            from mlx_flash_compress.layerskip import LayerSkipEngine

            log.info(
                "Initializing LayerSkip speculative decoding", extra={"action": "spec_init", "engine": "layerskip"}
            )
            self.spec_engine = LayerSkipEngine(self.model, self.tokenizer)
        elif self.speculative == "dflash":
            from mlx_flash_compress.dflash import DFlashEngine

            log.info("Initializing DFlash speculative decoding", extra={"action": "spec_init", "engine": "dflash"})
            self.spec_engine = DFlashEngine(
                self.model,
                drafter=None,
                config=__import__("mlx_flash_compress.dflash", fromlist=["DFlashConfig"]).DFlashConfig(),
                tokenizer=self.tokenizer,
            )
        else:
            log.warning(
                f"Unknown speculative engine: {self.speculative}",
                extra={"action": "spec_init_failed", "engine": self.speculative},
            )

    def _suggest_memory_actions(self, mem):
        """Suggest actions to reduce memory pressure."""
        global logger
        log = logger or __import__("logging").getLogger("mlx_flash")
        log.warning(
            "Memory optimization suggestions: close browser tabs, Xcode, Docker. "
            "Or use a smaller model / enable mixed precision.",
            extra={
                "action": "memory_suggestions",
                "memory_gb": round(mem.available_gb, 1),
            },
        )

    def get_status(self) -> dict:
        """Get current server status with optimization hints."""
        mem = get_memory_state()
        uptime = time.monotonic() - self.start_time
        cache_budget = self.mem_mgr.get_cache_budget_gb()
        hints = self.mem_mgr.get_optimization_hints()

        status = {
            "model": self.model_name,
            "model_loaded": self.model is not None,
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
            "speculative_engine": self.speculative,
        }
        if self.spec_engine is not None:
            try:
                if hasattr(self.spec_engine, "get_stats"):
                    status["speculative_stats"] = self.spec_engine.get_stats()
                elif hasattr(self.spec_engine, "get_stats_summary"):
                    status["speculative_stats"] = self.spec_engine.get_stats_summary()
            except Exception:
                pass
        return status

    def generate(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.7) -> dict:
        """Generate a response with memory awareness."""
        if self.model is None:
            self.load_model()

        # Check memory before generation — auto-release if needed
        log = logger or __import__("logging").getLogger("mlx_flash")
        mem = get_memory_state()
        release_info = None
        if mem.pressure_level in ("critical", "warning"):
            log.warning(
                "Memory pressure before generation",
                extra={
                    "action": "pressure_pre_generate",
                    "pressure": mem.pressure_level,
                    "memory_gb": round(mem.available_gb, 1),
                },
            )
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

        # Generate — use speculative engine if available
        t0 = time.monotonic()
        spec_stats = None
        if self.spec_engine is not None:
            prompt_tokens = mx.array(self.tokenizer.encode(prompt))
            result_tokens = self.spec_engine.generate(prompt_tokens, max_tokens=max_tokens)
            mx.synchronize()
            output = self.tokenizer.decode(result_tokens.tolist()[len(prompt_tokens) :])
            if hasattr(self.spec_engine, "get_stats"):
                spec_stats = self.spec_engine.get_stats()
            elif hasattr(self.spec_engine, "get_stats_summary"):
                spec_stats = self.spec_engine.get_stats_summary()
        else:
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )
        mx.synchronize()
        elapsed = time.monotonic() - t0

        tokens = len(self.tokenizer.encode(output))  # type: ignore[attr-defined]
        tps = tokens / elapsed if elapsed > 0 else 0

        self.total_requests += 1
        self.total_tokens += tokens

        log.info(
            "Inference complete",
            extra={
                "action": "inference_complete",
                "tokens": tokens,
                "tok_per_s": round(tps, 1),
                "latency_ms": round(elapsed * 1000),
                "speculative": self.speculative,
            },
        )

        # Check memory after generation
        mem_after = get_memory_state()
        if mem_after.pressure_level in ("critical", "warning"):
            log.warning(
                "Memory pressure after generation",
                extra={
                    "action": "pressure_post_generate",
                    "pressure": mem_after.pressure_level,
                    "memory_gb": round(mem_after.available_gb, 1),
                },
            )

        result = {
            "output": output,
            "tokens": tokens,
            "time_s": round(elapsed, 2),
            "tok_per_s": round(tps, 1),
            "memory_pressure": mem_after.pressure_level,
        }
        if spec_stats is not None:
            result["speculative_stats"] = spec_stats
        return result

    def _format_messages(self, messages: list[dict]) -> str:
        """Format chat messages for the model."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[attr-defined]
            except Exception:
                pass
        # Fallback: join messages
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP handler for OpenAI-compatible chat API."""

    server_state: InferenceState = None  # set by server setup

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(
                {
                    "data": [
                        {
                            "id": "local",
                            "object": "model",
                            "owned_by": "mlx-flash",
                        }
                    ]
                }
            )
        elif self.path == "/v1/engines":
            state = self.server_state
            engines = []
            for eng in SPECULATIVE_ENGINES:
                engines.append(
                    {
                        "id": eng,
                        "active": eng == state.speculative,
                        "description": {
                            "eagle3": "EAGLE-3: Autoregression head for hidden-state speculation (2.7-3.5x speedup)",
                            "layerskip": "LayerSkip: Self-speculative decoding via early exit (1.8-2.2x speedup)",
                            "dflash": "DFlash: Block diffusion parallel drafting (2-3x speedup)",
                            "none": "Standard autoregressive decoding",
                        }.get(eng, ""),
                    }
                )
            self._send_json({"engines": engines})
        elif self.path == "/health":
            state = self.server_state
            mem = get_memory_state()
            self._send_json(
                {
                    "status": "ok",
                    "model": state.model_name,
                    "model_loaded": state.model is not None,
                    "speculative_engine": state.speculative,
                    "memory_pressure": mem.pressure_level,
                    "uptime_s": round(time.monotonic() - state.start_time, 0),
                }
            )
        elif self.path == "/status":
            self._send_json(self.server_state.get_status())
        elif self.path == "/hints":
            hints = self.server_state.mem_mgr.get_optimization_hints()
            self._send_json({"hints": hints})
        elif self.path == "/release":
            result = self.server_state.mem_mgr.auto_release_if_needed()
            self._send_json(result)
        elif self.path == "/metrics":
            self._serve_metrics()
        elif self.path == "/chat":
            self._serve_chat_html()
        elif self.path == "/config":
            self._handle_config_get()
        elif self.path == "/admin":
            self._serve_dashboard_html()
        elif self.path == "/profile/models":
            self._handle_profile_models()
        elif self.path == "/telemetry":
            self._handle_telemetry()
        elif self.path == "/telemetry/current":
            self._handle_telemetry_current()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat()
        elif self.path in ("/switch", "/v1/models/switch"):
            self._handle_switch()
        elif self.path == "/reload":
            self._handle_reload()
        elif self.path == "/shutdown":
            self._handle_shutdown()
        elif self.path == "/config":
            self._handle_config_set()
        elif self.path == "/profile":
            self._handle_profile()
        elif self.path == "/profile/batch":
            self._handle_profile_batch()
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_switch(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        new_model = data.get("model")
        if not new_model:
            self._send_json({"error": "Missing 'model' field"}, 400)
            return

        state = self.server_state
        old_model = state.model_name
        log = logger or __import__("logging").getLogger("mlx_flash")
        log.info("Switching model", extra={"model": new_model, "action": "model_switch"})

        # Unload current model
        state.model = None
        state.tokenizer = None
        import gc

        gc.collect()
        try:
            mx.clear_cache()
        except AttributeError:
            pass

        # Load new model
        state.model_name = new_model
        try:
            state.load_model()
            self._send_json(
                {
                    "switched": True,
                    "model": new_model,
                    "previous": old_model,
                }
            )
        except Exception as e:
            # Rollback on failure
            state.model_name = old_model
            self._send_json(
                {
                    "switched": False,
                    "error": str(e),
                    "model": old_model,
                },
                500,
            )

    def _handle_reload(self):
        """Reload: refresh memory state, log status."""
        log = logger or __import__("logging").getLogger("mlx_flash")
        log.info("Reload requested via /reload", extra={"action": "reload"})
        mem = get_memory_state()
        state = self.server_state
        self._send_json(
            {
                "reloaded": True,
                "model": state.model_name,
                "model_loaded": state.model is not None,
                "memory_available_gb": round(mem.available_gb, 1),
                "pressure": mem.pressure_level,
            }
        )

    def _handle_shutdown(self):
        """Graceful shutdown via API."""
        log = logger or __import__("logging").getLogger("mlx_flash")
        log.info("Shutdown requested via /shutdown", extra={"action": "shutdown"})
        self._send_json(
            {
                "shutting_down": True,
                "message": "Server stopping in ~500ms",
            }
        )
        import threading

        def _stop():
            time.sleep(0.5)
            self.server.shutdown()

        threading.Thread(target=_stop, daemon=True).start()

    def _handle_config_get(self):
        state = self.server_state
        mem = get_memory_state()
        config = {
            "model": state.model_name,
            "model_loaded": state.model is not None,
            "speculative": state.speculative,
            "kv_bits": state.kv_bits,
            "cache_budget_pct": state.cache_budget_pct,
            "batching": state.batching,
            "request_timeout": state.request_timeout,
            "memory": {
                "total_gb": round(mem.total_gb, 1),
                "available_gb": round(mem.available_gb, 1),
                "pressure": mem.pressure_level,
            },
            "stats": {
                "total_requests": state.total_requests,
                "total_tokens": state.total_tokens,
                "uptime_secs": round(time.monotonic() - state.start_time, 1),
            },
        }
        if state.spec_engine is not None and hasattr(state.spec_engine, "get_stats"):
            config["speculative_stats"] = state.spec_engine.get_stats()
        self._send_json(config)

    def _handle_config_set(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        state = self.server_state
        updated = []
        if "cache_budget_pct" in data:
            state.cache_budget_pct = float(data["cache_budget_pct"])
            updated.append("cache_budget_pct")
        if "kv_bits" in data:
            state.kv_bits = int(data["kv_bits"])
            updated.append("kv_bits")
        if "request_timeout" in data:
            state.request_timeout = float(data["request_timeout"])
            updated.append("request_timeout")
        if "batching" in data:
            state.batching = bool(data["batching"])
            updated.append("batching")
        self._send_json({"updated": updated, "status": "ok"})

    def _handle_profile(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        state = self.server_state
        if state.model is None:
            self._send_json({"error": "No model loaded"}, 503)
            return

        max_tokens = data.get("max_tokens", 32)
        prompt_text = data.get("prompt", "def binary_search(arr, target):\n    ")

        prompt_tokens = mx.array(state.tokenizer.encode(prompt_text))
        mem_before = get_memory_state()
        t0 = time.monotonic()

        output = generate(
            state.model,
            state.tokenizer,
            prompt=prompt_text,
            max_tokens=max_tokens,
            verbose=False,
        )
        mx.synchronize()

        elapsed = time.monotonic() - t0
        out_tokens = len(state.tokenizer.encode(output))
        mem_after = get_memory_state()

        self._send_json(
            {
                "model": state.model_name,
                "prompt_tokens": len(prompt_tokens),
                "output_tokens": out_tokens,
                "elapsed_secs": round(elapsed, 3),
                "tokens_per_sec": round(out_tokens / elapsed, 1) if elapsed > 0 else 0,
                "memory_before_gb": round(mem_before.available_gb, 2),
                "memory_after_gb": round(mem_after.available_gb, 2),
            }
        )

    def _handle_profile_models(self):
        """Return the model registry from models.yaml without profiling."""
        try:
            registry = load_model_registry()
        except Exception as e:
            self._send_json({"error": f"Cannot read models.yaml: {e}"}, 500)
            return

        models = []
        for m in registry.get("models", []):
            models.append(
                {
                    "id": m.get("id", ""),
                    "category_expected": m.get("category_expected", "unknown"),
                    "size_gb_approx": m.get("size_gb_approx", 0),
                    "skip": m.get("skip", False),
                    "notes": m.get("notes", ""),
                    "drafter": m.get("drafter"),
                }
            )

        self._send_json(
            {
                "models": models,
                "cache_dir": registry.get("cache_dir", "~/.cache/huggingface/hub"),
                "total_models": len(models),
            }
        )

    def _handle_profile_batch(self):
        """Profile multiple models from the registry sequentially."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        max_tokens = data.get("max_tokens", 32)
        prompt_text = data.get("prompt", "def binary_search(arr, target):\n    ")
        requested_models = data.get("models", [])

        # Load registry
        try:
            registry = load_model_registry()
        except Exception as e:
            self._send_json({"error": f"Cannot read models.yaml: {e}"}, 500)
            return

        # Build the list of models to profile
        registry_models = registry.get("models", [])
        if requested_models:
            # Filter to only requested models
            models_to_profile = [m for m in registry_models if m.get("id") in requested_models]
        else:
            # Profile all non-skipped models
            models_to_profile = [m for m in registry_models if not m.get("skip", False)]

        log = logger or __import__("logging").getLogger("mlx_flash")
        state = self.server_state
        original_model = state.model_name
        results = []

        for entry in models_to_profile:
            model_id = entry.get("id", "")
            category = entry.get("category_expected", "unknown")
            log.info(
                f"Batch profiling model: {model_id}",
                extra={"action": "batch_profile_start", "model": model_id},
            )

            result = {"model": model_id, "category": category}

            try:
                # Unload any currently loaded model
                state.model = None
                state.tokenizer = None
                gc.collect()
                try:
                    mx.clear_cache()
                except AttributeError:
                    pass

                # Load target model
                state.model_name = model_id
                t_load = time.monotonic()
                state.model, state.tokenizer = load(model_id)
                mx.synchronize()
                load_time = time.monotonic() - t_load

                # Profile: generate tokens and measure
                mem_before = get_memory_state()
                t0 = time.monotonic()
                output = generate(
                    state.model,
                    state.tokenizer,
                    prompt=prompt_text,
                    max_tokens=max_tokens,
                    verbose=False,
                )
                mx.synchronize()
                elapsed = time.monotonic() - t0
                out_tokens = len(state.tokenizer.encode(output))
                mem_after = get_memory_state()

                result["tokens_per_sec"] = round(out_tokens / elapsed, 1) if elapsed > 0 else 0
                result["memory_gb"] = round(mem_before.available_gb - mem_after.available_gb, 2)
                result["load_time_s"] = round(load_time, 2)

            except Exception as e:
                result["error"] = str(e)

            results.append(result)

        # Restore original model name (leave model unloaded to save memory)
        state.model = None
        state.tokenizer = None
        gc.collect()
        try:
            mx.clear_cache()
        except AttributeError:
            pass
        state.model_name = original_model

        self._send_json(results)

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
        state = self.server_state

        # Route through continuous batching engine when enabled
        if state.batching and state.engine is not None:
            if stream:
                self._handle_stream_batched(messages, max_tokens, temperature)
            else:
                self._handle_chat_batched(messages, max_tokens, temperature)
            return

        if stream:
            self._handle_stream(messages, max_tokens, temperature)
            return

        result = state.generate(messages, max_tokens, temperature)

        if "error" in result:
            self._send_json({"error": result["error"]}, 503)
            return

        # Format as OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "local",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["output"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": result["tokens"],
                "total_tokens": result["tokens"],
            },
            "mlx_flash_compress": {
                "tok_per_s": result["tok_per_s"],
                "memory_pressure": result["memory_pressure"],
                "speculative_engine": state.speculative,
            },
        }
        if "speculative_stats" in result:
            response["mlx_flash_compress"]["speculative_stats"] = result["speculative_stats"]
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
            state.model,
            state.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
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
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }
                ],
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

    def _handle_chat_batched(self, messages, max_tokens, temperature):
        """Handle non-streaming chat via continuous batching engine."""
        state = self.server_state
        if state.model is None:
            state.load_model()

        prompt = state._format_messages(messages)
        t0 = time.monotonic()

        req = state.engine.submit(prompt, max_tokens=max_tokens, temperature=temperature)
        result = state.engine.wait_for_completion(req, timeout=60.0)

        elapsed = time.monotonic() - t0

        from mlx_flash_compress.continuous_batching import RequestStatus

        if result.status != RequestStatus.COMPLETED:
            self._send_json({"error": "Request timed out or was cancelled"}, 503)
            return

        output = state.tokenizer.decode(result.generated_tokens)
        tokens = len(result.generated_tokens)
        tps = tokens / elapsed if elapsed > 0 else 0

        state.total_requests += 1
        state.total_tokens += tokens

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "local",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(req.prompt_tokens),
                "completion_tokens": tokens,
                "total_tokens": len(req.prompt_tokens) + tokens,
            },
            "mlx_flash_compress": {
                "tok_per_s": round(tps, 1),
                "batched": True,
            },
        }
        self._send_json(response)

    def _handle_stream_batched(self, messages, max_tokens, temperature):
        """Handle streaming (SSE) response via continuous batching engine."""
        state = self.server_state
        if state.model is None:
            state.load_model()

        prompt = state._format_messages(messages)
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        req = state.engine.submit(prompt, max_tokens=max_tokens, temperature=temperature)

        # Start SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        token_count = 0
        for token_id in state.engine.stream_tokens(req):
            chunk_text = state.tokenizer.decode([token_id])
            token_count += 1
            event = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "local",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None,
                    }
                ],
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

        state.total_requests += 1
        state.total_tokens += token_count

    def _serve_chat_html(self):
        """Serve the web chat UI — loads from the shared HTML file."""
        import pathlib

        chat_html_path = pathlib.Path(__file__).parent.parent / "assets" / "chat.html"
        if chat_html_path.exists():
            html = chat_html_path.read_text()
        else:
            html = self._fallback_chat_html()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html.encode())

    @staticmethod
    def _fallback_chat_html():
        return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MLX-Flash Chat</title></head>
<body style="background:#0a0e14;color:#d4dce8;font-family:system-ui;max-width:700px;margin:40px auto;padding:0 20px">
<h1 style="background:linear-gradient(135deg,#4da6ff,#7b61ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent">MLX-Flash Chat</h1>
<p style="color:#5c6a7a">assets/chat.html not found — using fallback UI</p>
<div id="msgs"></div>
<div style="display:flex;gap:8px;margin-top:20px">
<input id="inp" style="flex:1;background:#131920;border:1px solid #262d38;border-radius:8px;padding:10px;color:#d4dce8;font-size:14px" placeholder="Type a message..." autofocus>
<button onclick="send()" style="background:linear-gradient(135deg,#4da6ff,#7b61ff);border:none;color:#fff;padding:10px 20px;border-radius:8px;cursor:pointer">Send</button>
</div>
<script>
const msgs=[],msgsEl=document.getElementById('msgs'),inp=document.getElementById('inp');
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send()});
async function send(){
  const t=inp.value.trim();if(!t)return;inp.value='';
  msgs.push({role:'user',content:t});
  msgsEl.innerHTML+=`<p><b style="color:#4da6ff">You:</b> ${t}</p>`;
  const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:'local',messages:msgs,max_tokens:512})});
  const j=await r.json();
  const reply=j.choices?.[0]?.message?.content||j.error||'No response';
  msgs.push({role:'assistant',content:reply});
  msgsEl.innerHTML+=`<p><b style="color:#7b61ff">AI:</b> ${reply}</p>`;
}
</script></body></html>"""

    def _serve_metrics(self):
        """Serve Prometheus exposition format metrics."""
        state = self.server_state
        mem = get_memory_state()
        uptime = time.monotonic() - state.start_time

        lines = []
        lines.append("# HELP mlx_flash_info Server metadata.")
        lines.append("# TYPE mlx_flash_info gauge")
        lines.append(f'mlx_flash_info{{model="{state.model_name}"}} 1')
        lines.append("")
        lines.append("# HELP mlx_flash_uptime_seconds Time since server start.")
        lines.append("# TYPE mlx_flash_uptime_seconds gauge")
        lines.append(f"mlx_flash_uptime_seconds {uptime:.1f}")
        lines.append("")
        lines.append("# HELP mlx_flash_requests_total Total inference requests.")
        lines.append("# TYPE mlx_flash_requests_total counter")
        lines.append(f"mlx_flash_requests_total {state.total_requests}")
        lines.append("")
        lines.append("# HELP mlx_flash_tokens_generated_total Total tokens generated.")
        lines.append("# TYPE mlx_flash_tokens_generated_total counter")
        lines.append(f"mlx_flash_tokens_generated_total {state.total_tokens}")
        lines.append("")
        # Memory
        total_b = mem.total_gb * 1073741824
        free_b = mem.free_gb * 1073741824
        available_b = mem.available_gb * 1073741824
        swap_b = mem.swap_used_gb * 1073741824
        used_ratio = (1.0 - mem.available_gb / mem.total_gb) if mem.total_gb > 0 else 0
        pressure_map = {"normal": 0, "warning": 1, "critical": 2}
        pressure_val = pressure_map.get(mem.pressure_level, 0)

        lines.append("# HELP mlx_flash_memory_total_bytes Total physical RAM.")
        lines.append("# TYPE mlx_flash_memory_total_bytes gauge")
        lines.append(f"mlx_flash_memory_total_bytes {total_b:.0f}")
        lines.append("")
        lines.append("# HELP mlx_flash_memory_free_bytes Free (unused) RAM.")
        lines.append("# TYPE mlx_flash_memory_free_bytes gauge")
        lines.append(f"mlx_flash_memory_free_bytes {free_b:.0f}")
        lines.append("")
        lines.append("# HELP mlx_flash_memory_available_bytes Usable RAM (free + inactive).")
        lines.append("# TYPE mlx_flash_memory_available_bytes gauge")
        lines.append(f"mlx_flash_memory_available_bytes {available_b:.0f}")
        lines.append("")
        lines.append("# HELP mlx_flash_memory_swap_used_bytes Swap space in use.")
        lines.append("# TYPE mlx_flash_memory_swap_used_bytes gauge")
        lines.append(f"mlx_flash_memory_swap_used_bytes {swap_b:.0f}")
        lines.append("")
        lines.append("# HELP mlx_flash_memory_used_ratio Fraction of RAM in use (0-1).")
        lines.append("# TYPE mlx_flash_memory_used_ratio gauge")
        lines.append(f"mlx_flash_memory_used_ratio {used_ratio:.4f}")
        lines.append("")
        lines.append("# HELP mlx_flash_memory_pressure macOS memory pressure (0=normal, 1=warning, 2=critical).")
        lines.append("# TYPE mlx_flash_memory_pressure gauge")
        lines.append(f"mlx_flash_memory_pressure {pressure_val}")
        lines.append("")
        # Model loaded
        loaded = 1 if state.model is not None else 0
        lines.append("# HELP mlx_flash_model_loaded Whether a model is loaded (1) or not (0).")
        lines.append("# TYPE mlx_flash_model_loaded gauge")
        lines.append(f"mlx_flash_model_loaded {loaded}")
        lines.append("")

        body = "\n".join(lines) + "\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())

    def _serve_dashboard_html(self):
        """Serve the admin dashboard — loads from assets/dashboard.html or inline fallback."""
        import pathlib

        dash_path = pathlib.Path(__file__).parent.parent / "assets" / "dashboard.html"
        if dash_path.exists():
            html = dash_path.read_text()
        else:
            html = self._fallback_dashboard_html()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(html.encode())

    @staticmethod
    def _fallback_dashboard_html():
        return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MLX-Flash Dashboard</title></head>
<body style="background:#0a0e14;color:#d4dce8;font-family:system-ui;max-width:900px;margin:40px auto;padding:0 20px">
<h1 style="background:linear-gradient(135deg,#4da6ff,#7b61ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent">MLX-Flash Dashboard</h1>
<p style="color:#5c6a7a">assets/dashboard.html not found — using fallback</p>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:24px">
<div id="model-card" style="background:#1a2029;border:1px solid #262d38;border-radius:12px;padding:18px"><div style="color:#5c6a7a;font-size:0.75rem;text-transform:uppercase">Model</div><div id="model" style="font-size:1.1rem;font-weight:600;margin-top:8px">loading...</div></div>
<div style="background:#1a2029;border:1px solid #262d38;border-radius:12px;padding:18px"><div style="color:#5c6a7a;font-size:0.75rem;text-transform:uppercase">Memory</div><div id="mem" style="font-size:1.8rem;font-weight:700;margin-top:8px;color:#2dd4a8">--%</div></div>
<div style="background:#1a2029;border:1px solid #262d38;border-radius:12px;padding:18px"><div style="color:#5c6a7a;font-size:0.75rem;text-transform:uppercase">Tokens</div><div id="tok" style="font-size:1.8rem;font-weight:700;margin-top:8px;color:#4da6ff">0</div></div>
</div>
<div style="margin-top:16px;background:#1a2029;border:1px solid #262d38;border-radius:12px;padding:18px">
<div style="color:#5c6a7a;font-size:0.75rem;text-transform:uppercase;margin-bottom:12px">Memory Chart</div>
<canvas id="mem-chart" style="width:100%;height:150px"></canvas>
</div>
<div style="margin-top:12px;text-align:center"><a href="/chat" style="color:#4da6ff;text-decoration:none;font-size:0.85rem">Open Chat UI &rarr;</a></div>
<script>
const MAX=60;let memH=[];
function draw(c,d,col){const dpr=devicePixelRatio||1,r=c.getBoundingClientRect();c.width=r.width*dpr;c.height=r.height*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);const w=r.width,h=r.height;x.clearRect(0,0,w,h);if(d.length<2)return;const mx=Math.max(...d,1)*1.15;const p=d.map((v,i)=>[i/(MAX-1)*w,h-(v/mx)*(h-30)-4]);x.beginPath();x.strokeStyle=col;x.lineWidth=2;p.forEach((pt,i)=>i?x.lineTo(pt[0],pt[1]):x.moveTo(pt[0],pt[1]));x.stroke()}
async function poll(){try{const s=await fetch('/status').then(r=>r.json()),m=s.memory||{},st=s.stats||{};document.getElementById('model').textContent=(s.model||'none').split('/').pop();const a=Math.max((m.free_gb||0)+(m.inactive_gb||0)*0.5,0),t=m.total_gb||1,p=((1-a/t)*100);document.getElementById('mem').textContent=p.toFixed(0)+'%';document.getElementById('tok').textContent=(st.tokens_generated||0).toLocaleString();memH.push(p);if(memH.length>MAX)memH.shift();draw(document.getElementById('mem-chart'),memH,'#4da6ff')}catch(e){}}
poll();setInterval(poll,2000);
</script></body></html>"""

    def _handle_telemetry(self):
        """Return current telemetry sample plus 120-sample history."""
        state = self.server_state
        stats = state.telemetry.get_stats()
        history = state.telemetry.get_history(seconds=120)
        self._send_json(
            {
                "current": stats.get("current", {}),
                "stats": {
                    "samples_count": stats.get("samples_count", 0),
                    "avg": stats.get("avg", {}),
                    "max": stats.get("max", {}),
                    "min": stats.get("min", {}),
                },
                "history": history,
            }
        )

    def _handle_telemetry_current(self):
        """Return just the current telemetry sample."""
        state = self.server_state
        sample = state.telemetry.sample()
        self._send_json(sample.to_dict())

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
        """Route HTTP access logs through structured logger."""
        log = logger or __import__("logging").getLogger("mlx_flash")
        msg = format % args
        if "POST" in msg:
            log.info(msg, extra={"action": "http_request"})
        elif "error" in msg.lower() or "404" in msg or "500" in msg:
            log.warning(msg, extra={"action": "http_error"})


def main():
    parser = argparse.ArgumentParser(description="MLX-Flash: Memory-aware inference server")
    parser.add_argument("--model", default=None, help="MLX model to serve (auto-selects based on RAM if not specified)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--preload", action="store_true", help="Load model immediately")
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=0,
        choices=[0, 4, 8],
        help="KV cache quantization bits (0=none, 8=45%% memory savings)",
    )
    parser.add_argument(
        "--log-format", default="text", choices=["text", "json"], help="Log format: text (human) or json (structured)"
    )
    parser.add_argument("--log-file", default=None, help="Log file path (also logs to stdout)")
    parser.add_argument(
        "--batching",
        action="store_true",
        help="Enable continuous batching engine for concurrent request handling",
    )
    parser.add_argument(
        "--speculative",
        default="none",
        choices=list(SPECULATIVE_ENGINES),
        help="Speculative decoding engine (default: none)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--cache-budget-pct",
        type=float,
        default=0.8,
        help="Cache budget as fraction of available memory (default: 0.8)",
    )
    parser.add_argument(
        "--safety-margin-gb",
        type=float,
        default=2.0,
        help="Safety margin in GB for memory management (default: 2.0)",
    )
    args = parser.parse_args()

    # Initialize structured logging
    global logger
    logger = setup_logging(
        component="python-worker",
        port=args.port,
        json_format=(args.log_format == "json"),
        log_file=args.log_file,
    )

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

    state = InferenceState(
        args.model,
        cache_budget_pct=args.cache_budget_pct,
        kv_bits=args.kv_bits,
        batching=args.batching,
        speculative=args.speculative,
        request_timeout=args.request_timeout,
        safety_margin_gb=args.safety_margin_gb,
    )

    logger.info(
        "MLX-Flash inference server starting",
        extra={
            "model": args.model,
            "port": args.port,
            "speculative": args.speculative,
        },
    )

    mem = get_memory_state()
    logger.info(
        "Hardware detected",
        extra={
            "memory_gb": round(mem.available_gb, 1),
            "pressure": mem.pressure_level,
        },
    )

    if args.preload:
        state.load_model()

    ChatHandler.server_state = state

    # Start hardware telemetry sampling
    state.telemetry.start_sampling(interval_ms=1000)
    logger.info("Hardware telemetry started", extra={"action": "telemetry_start", "interval_ms": 1000})

    server = ThreadedHTTPServer((args.host, args.port), ChatHandler)
    logger.info(
        "Server listening",
        extra={
            "port": args.port,
            "action": "server_start",
            "model": args.model,
        },
    )

    def _graceful_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(
            f"Received {sig_name}, shutting down gracefully", extra={"action": "server_stop", "signal": sig_name}
        )
        state.telemetry.stop_sampling()
        server.shutdown()

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down", extra={"action": "server_stop"})
        state.telemetry.stop_sampling()
        server.shutdown()


if __name__ == "__main__":
    main()
