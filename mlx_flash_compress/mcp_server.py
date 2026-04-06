"""MCP stdio server for Claude Code integration.

Implements the Model Context Protocol (MCP) over stdio so Claude Code
can launch MLX-Flash as a tool server and call generate/check_memory/etc.

Setup in Claude Code:
  Add to ~/.claude/mcp.json:
  {
    "mcpServers": {
      "mlx-flash": {
        "command": "python",
        "args": ["-m", "mlx_flash_compress.mcp_server"]
      }
    }
  }

Protocol: JSON-RPC 2.0 over stdin/stdout
  - initialize → returns server info + tool list
  - tools/list → returns available tools
  - tools/call → executes a tool and returns result
"""

import json
import sys
from typing import Optional

from mlx_flash_compress.mcp_tools import MCP_TOOLS, get_mcp_manifest
from mlx_flash_compress.hardware import detect_hardware
from mlx_flash_compress.memory_manager import get_memory_state


SERVER_INFO = {
    "name": "mlx-flash",
    "version": "0.6.2",
    "description": "Run AI models too large for your Mac's memory at near-full speed",
}

# Lazy-loaded model state
_model = None
_tokenizer = None
_model_name = None


def _ensure_model(model_name: Optional[str] = None):
    """Lazy-load the model on first generate call."""
    global _model, _tokenizer, _model_name

    if _model is not None and (model_name is None or model_name == _model_name):
        return

    try:
        from mlx_lm import load
        from mlx_flash_compress.chat import auto_select_model

        if model_name is None:
            hw = detect_hardware()
            model_name = auto_select_model(hw.total_ram_gb)

        _model, _tokenizer = load(model_name)
        _model_name = model_name
    except ImportError:
        raise RuntimeError("mlx-lm required. Install: pip install mlx-lm")


def handle_tool_call(name: str, arguments: dict) -> dict:
    """Execute an MCP tool and return the result."""

    if name == "generate":
        _ensure_model()
        from mlx_lm import generate as mlx_generate
        prompt = arguments.get("prompt", "")
        max_tokens = arguments.get("max_tokens", 256)
        system = arguments.get("system", "You are a helpful assistant.")

        if _tokenizer and hasattr(_tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            try:
                formatted = _tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = prompt
        else:
            formatted = prompt

        result = mlx_generate(_model, _tokenizer, prompt=formatted, max_tokens=max_tokens)
        return {"type": "text", "text": result}

    elif name == "check_memory":
        hw = detect_hardware()
        mem = get_memory_state()
        return {
            "type": "text",
            "text": json.dumps({
                "chip": hw.chip,
                "total_ram_gb": hw.total_ram_gb,
                "available_gb": mem.available_gb,
                "pressure": mem.pressure_level,
                "model_loaded": _model_name or "none",
            }, indent=2)
        }

    elif name == "switch_model":
        model_name = arguments.get("model", "")
        if not model_name:
            return {"type": "text", "text": "Error: model name required"}
        _ensure_model(model_name)
        return {"type": "text", "text": f"Switched to {model_name}"}

    elif name == "release_memory":
        global _model, _tokenizer, _model_name
        fraction = arguments.get("fraction", 0.5)
        if fraction >= 1.0:
            _model = None
            _tokenizer = None
            _model_name = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except (ImportError, AttributeError):
                pass
            import gc
            gc.collect()
            return {"type": "text", "text": "Released all model memory"}
        return {"type": "text", "text": f"Partial release ({fraction}) not yet implemented"}

    elif name == "list_models":
        from mlx_flash_compress.chat import MODELS
        lines = []
        for name_m, total, active, size, mtype, desc in MODELS:
            lines.append(f"{name_m} — {total} params, {size}GB, {mtype}: {desc}")
        return {"type": "text", "text": "\n".join(lines)}

    elif name == "get_status":
        hw = detect_hardware()
        mem = get_memory_state()
        from mlx_flash_compress.kernels.ops import get_kernel_status
        return {
            "type": "text",
            "text": json.dumps({
                "model": _model_name or "none",
                "hardware": hw.chip,
                "ram_gb": hw.total_ram_gb,
                "available_gb": mem.available_gb,
                "pressure": mem.pressure_level,
                "kernels": get_kernel_status(),
            }, indent=2)
        }

    else:
        return {"type": "text", "text": f"Unknown tool: {name}"}


def handle_jsonrpc(request: dict) -> dict:
    """Handle a JSON-RPC 2.0 request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": SERVER_INFO,
                "capabilities": {
                    "tools": {"listChanged": False},
                },
            }
        }

    elif method == "notifications/initialized":
        return None  # no response needed for notifications

    elif method == "tools/list":
        tools = []
        for tool in MCP_TOOLS:
            tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
            })
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools}
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        try:
            content = handle_tool_call(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [content]}
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                }
            }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }


def main():
    """Run the MCP stdio server."""
    # MCP uses newline-delimited JSON over stdin/stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle_jsonrpc(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
