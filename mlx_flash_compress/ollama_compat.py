"""Ollama API compatibility layer.

Translates Ollama's /api/generate and /api/chat requests to OpenAI format
and vice versa, so MLX-Flash can serve both Ollama and OpenAI clients
on the same port.

Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md

Routes added:
  POST /api/generate  → translates to /v1/completions
  POST /api/chat      → translates to /v1/chat/completions
  GET  /api/tags      → returns loaded model info
  POST /api/show      → returns model details
"""

import json
import time
from typing import Optional


def ollama_generate_to_openai(body: dict) -> dict:
    """Convert Ollama /api/generate request to OpenAI /v1/completions format."""
    return {
        "model": body.get("model", "local"),
        "prompt": body.get("prompt", ""),
        "max_tokens": body.get("options", {}).get("num_predict", 256),
        "temperature": body.get("options", {}).get("temperature", 0.7),
        "top_p": body.get("options", {}).get("top_p", 0.9),
        "stream": body.get("stream", False),
        "stop": body.get("options", {}).get("stop", None),
    }


def ollama_chat_to_openai(body: dict) -> dict:
    """Convert Ollama /api/chat request to OpenAI /v1/chat/completions format."""
    messages = body.get("messages", [])
    return {
        "model": body.get("model", "local"),
        "messages": messages,
        "max_tokens": body.get("options", {}).get("num_predict", 256),
        "temperature": body.get("options", {}).get("temperature", 0.7),
        "top_p": body.get("options", {}).get("top_p", 0.9),
        "stream": body.get("stream", False),
    }


def openai_completion_to_ollama(response: dict, model_name: str = "local") -> dict:
    """Convert OpenAI completion response to Ollama /api/generate format."""
    choice = response.get("choices", [{}])[0]
    text = choice.get("text", "") or choice.get("message", {}).get("content", "")

    return {
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "response": text,
        "done": True,
        "context": [],  # Ollama context tokens (not used)
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": response.get("usage", {}).get("prompt_tokens", 0),
        "prompt_eval_duration": 0,
        "eval_count": response.get("usage", {}).get("completion_tokens", 0),
        "eval_duration": 0,
    }


def openai_chat_to_ollama(response: dict, model_name: str = "local") -> dict:
    """Convert OpenAI chat response to Ollama /api/chat format."""
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {"role": "assistant", "content": ""})

    return {
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "message": message,
        "done": True,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": response.get("usage", {}).get("prompt_tokens", 0),
        "eval_count": response.get("usage", {}).get("completion_tokens", 0),
    }


def ollama_tags_response(model_name: str, model_size_gb: float = 0) -> dict:
    """Build Ollama /api/tags response (list loaded models)."""
    return {
        "models": [
            {
                "name": model_name,
                "model": model_name,
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                "size": int(model_size_gb * 1024 * 1024 * 1024),
                "digest": "mlx-flash",
                "details": {
                    "parent_model": "",
                    "format": "mlx",
                    "family": model_name.split("/")[-1].split("-")[0] if "/" in model_name else model_name,
                    "parameter_size": f"{model_size_gb:.1f}GB",
                    "quantization_level": "Q4_0",
                },
            }
        ]
    }


def ollama_show_response(model_name: str) -> dict:
    """Build Ollama /api/show response (model details)."""
    return {
        "modelfile": f"FROM {model_name}",
        "parameters": "temperature 0.7\ntop_p 0.9\n",
        "template": "{{ .System }}\n{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "mlx",
            "family": model_name.split("/")[-1].split("-")[0] if "/" in model_name else model_name,
            "quantization_level": "Q4_0",
        },
    }


def is_ollama_request(path: str) -> bool:
    """Check if an HTTP path is an Ollama API request."""
    return path.startswith("/api/")
