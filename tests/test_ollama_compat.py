"""Tests for Ollama API compatibility layer."""

import pytest

from mlx_flash_compress.ollama_compat import (
    ollama_generate_to_openai,
    ollama_chat_to_openai,
    openai_completion_to_ollama,
    openai_chat_to_ollama,
    ollama_tags_response,
    ollama_show_response,
    is_ollama_request,
)


class TestOllamaToOpenAI:
    def test_generate_basic(self):
        body = {"prompt": "Hello", "model": "test"}
        result = ollama_generate_to_openai(body)
        assert result["prompt"] == "Hello"
        assert result["model"] == "test"

    def test_generate_with_options(self):
        body = {
            "prompt": "Hi",
            "options": {"num_predict": 100, "temperature": 0.5, "top_p": 0.8}
        }
        result = ollama_generate_to_openai(body)
        assert result["max_tokens"] == 100
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.8

    def test_generate_defaults(self):
        body = {"prompt": "Hi"}
        result = ollama_generate_to_openai(body)
        assert result["model"] == "local"
        assert result["max_tokens"] == 256
        assert result["temperature"] == 0.7

    def test_generate_stream(self):
        body = {"prompt": "Hi", "stream": True}
        result = ollama_generate_to_openai(body)
        assert result["stream"] is True

    def test_chat_basic(self):
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test"
        }
        result = ollama_chat_to_openai(body)
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["model"] == "test"

    def test_chat_defaults(self):
        body = {"messages": []}
        result = ollama_chat_to_openai(body)
        assert result["model"] == "local"
        assert result["max_tokens"] == 256


class TestOpenAIToOllama:
    def test_completion_to_ollama(self):
        response = {
            "choices": [{"text": "World"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
        }
        result = openai_completion_to_ollama(response, "test-model")
        assert result["response"] == "World"
        assert result["model"] == "test-model"
        assert result["done"] is True
        assert result["prompt_eval_count"] == 5
        assert result["eval_count"] == 1

    def test_chat_to_ollama(self):
        response = {
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}
        }
        result = openai_chat_to_ollama(response, "test-model")
        assert result["message"]["content"] == "Hi!"
        assert result["done"] is True

    def test_empty_response(self):
        result = openai_completion_to_ollama({})
        assert result["response"] == ""
        assert result["done"] is True


class TestOllamaEndpoints:
    def test_tags_response(self):
        result = ollama_tags_response("mlx-community/gemma-4-31b-it-4bit", 20.0)
        assert len(result["models"]) == 1
        model = result["models"][0]
        assert "gemma" in model["name"]
        assert model["size"] > 0

    def test_show_response(self):
        result = ollama_show_response("mlx-community/gemma-4-31b-it-4bit")
        assert "modelfile" in result
        assert "details" in result
        assert result["details"]["format"] == "mlx"


class TestIsOllamaRequest:
    def test_ollama_paths(self):
        assert is_ollama_request("/api/generate") is True
        assert is_ollama_request("/api/chat") is True
        assert is_ollama_request("/api/tags") is True
        assert is_ollama_request("/api/show") is True

    def test_openai_paths(self):
        assert is_ollama_request("/v1/chat/completions") is False
        assert is_ollama_request("/v1/models") is False

    def test_other_paths(self):
        assert is_ollama_request("/health") is False
        assert is_ollama_request("/") is False
