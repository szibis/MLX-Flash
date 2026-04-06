"""MCP (Model Context Protocol) tool definitions for Claude Code integration.

Exposes MLX-Flash server capabilities as MCP tools that Claude Code
can discover and call directly without manual API configuration.

Usage:
  # In ~/.claude/.mcp.json:
  {
    "mlx-flash": {
      "command": "mlx-flash",
      "args": ["--port", "8080", "--mcp"],
      "tools": "auto"
    }
  }

  # Or standalone MCP server:
  python -m mlx_flash_compress.mcp_tools --port 8080
"""

import json
from typing import Optional

MCP_TOOLS = [
    {
        "name": "generate",
        "description": "Generate text completion from the loaded MLX model",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The input text prompt"
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate",
                    "default": 256
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature (0.0 = deterministic)",
                    "default": 0.7
                },
                "system": {
                    "type": "string",
                    "description": "System prompt for the model",
                    "default": "You are a helpful assistant."
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "check_memory",
        "description": "Check current memory pressure, cache stats, and available RAM",
        "inputSchema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "switch_model",
        "description": "Switch to a different MLX model (downloads if needed)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "HuggingFace model ID (e.g., mlx-community/gemma-4-31b-it-4bit)"
                }
            },
            "required": ["model"]
        }
    },
    {
        "name": "release_memory",
        "description": "Release cached expert weights to free RAM for other apps",
        "inputSchema": {
            "type": "object",
            "properties": {
                "fraction": {
                    "type": "number",
                    "description": "Fraction of cache to release (0.0-1.0)",
                    "default": 0.5
                }
            }
        }
    },
    {
        "name": "list_models",
        "description": "List available models with size info and hardware compatibility",
        "inputSchema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "get_status",
        "description": "Get full server status: model, hardware, memory, optimization hints",
        "inputSchema": {
            "type": "object",
            "properties": {},
        }
    }
]


def get_mcp_manifest() -> dict:
    """Return the MCP tool manifest for Claude Code discovery."""
    return {
        "name": "mlx-flash",
        "version": "0.6.1",
        "description": "Run AI models too large for your Mac's memory at near-full speed",
        "tools": MCP_TOOLS,
    }


def get_tool_names() -> list[str]:
    """Return list of available tool names."""
    return [t["name"] for t in MCP_TOOLS]


def format_mcp_response(tool_name: str, result: dict) -> dict:
    """Format a tool execution result as an MCP response."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_name,
        "content": json.dumps(result, indent=2),
    }
