"""Tests for MCP stdio server: JSON-RPC handling, tool dispatch, parameter validation."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mlx_flash_compress.mcp_server import (
    SERVER_INFO,
    handle_jsonrpc,
    handle_tool_call,
)


class TestServerInfo:
    def test_has_required_fields(self):
        assert "name" in SERVER_INFO
        assert "version" in SERVER_INFO
        assert "description" in SERVER_INFO

    def test_server_name(self):
        assert SERVER_INFO["name"] == "mlx-flash"


class TestHandleJsonRPC:
    """Test JSON-RPC 2.0 protocol handling."""

    def test_initialize(self):
        req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        resp = handle_jsonrpc(req)
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "result" in resp
        result = resp["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"] == SERVER_INFO
        assert "tools" in result["capabilities"]

    def test_notifications_initialized_returns_none(self):
        req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        resp = handle_jsonrpc(req)
        assert resp is None

    def test_tools_list(self):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        resp = handle_jsonrpc(req)
        assert resp["id"] == 2
        tools = resp["result"]["tools"]
        assert isinstance(tools, list)
        assert len(tools) > 0
        # Each tool must have name, description, inputSchema
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_tools_list_includes_expected_tools(self):
        req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
        resp = handle_jsonrpc(req)
        tool_names = {t["name"] for t in resp["result"]["tools"]}
        assert "generate" in tool_names
        assert "check_memory" in tool_names
        assert "switch_model" in tool_names
        assert "release_memory" in tool_names
        assert "list_models" in tool_names
        assert "get_status" in tool_names

    def test_unknown_method(self):
        req = {"jsonrpc": "2.0", "id": 4, "method": "nonexistent/method", "params": {}}
        resp = handle_jsonrpc(req)
        assert "error" in resp
        assert resp["error"]["code"] == -32601
        assert "nonexistent/method" in resp["error"]["message"]

    def test_tools_call_unknown_tool(self):
        req = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
        resp = handle_jsonrpc(req)
        assert resp["id"] == 5
        content = resp["result"]["content"]
        assert len(content) == 1
        assert "Unknown tool" in content[0]["text"]

    def test_tools_call_error_handling(self):
        """tools/call should catch exceptions and return isError."""
        # Use a tool that will reliably raise (switch_model with a bogus model name)
        # We mock _ensure_model to raise, testing the error path
        with patch("mlx_flash_compress.mcp_server._ensure_model", side_effect=RuntimeError("test error")):
            req = {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "switch_model", "arguments": {"model": "bogus/model"}},
            }
            resp = handle_jsonrpc(req)
            assert resp["id"] == 6
            result = resp["result"]
            assert result["isError"] is True
            assert "Error" in result["content"][0]["text"]

    def test_preserves_request_id(self):
        for req_id in [1, "abc-123", 999]:
            req = {"jsonrpc": "2.0", "id": req_id, "method": "initialize", "params": {}}
            resp = handle_jsonrpc(req)
            assert resp["id"] == req_id

    def test_missing_method(self):
        req = {"jsonrpc": "2.0", "id": 7}
        resp = handle_jsonrpc(req)
        # empty method → unknown method error
        assert "error" in resp

    def test_missing_params(self):
        req = {"jsonrpc": "2.0", "id": 8, "method": "tools/list"}
        resp = handle_jsonrpc(req)
        # Should still work with default empty params
        assert "result" in resp


class TestHandleToolCall:
    """Test individual tool dispatch logic."""

    def test_unknown_tool(self):
        result = handle_tool_call("bogus_tool", {})
        assert result["type"] == "text"
        assert "Unknown tool" in result["text"]

    def test_switch_model_missing_name(self):
        result = handle_tool_call("switch_model", {})
        assert "Error" in result["text"] or "model name required" in result["text"]

    def test_switch_model_empty_name(self):
        result = handle_tool_call("switch_model", {"model": ""})
        assert "model name required" in result["text"]

    @patch("mlx_flash_compress.mcp_server._model", "fake_model")
    @patch("mlx_flash_compress.mcp_server._tokenizer", "fake_tokenizer")
    @patch("mlx_flash_compress.mcp_server._model_name", "test-model")
    def test_release_memory_full(self):
        """release_memory with fraction >= 1.0 should clear model state."""
        import mlx_flash_compress.mcp_server as srv

        srv._model = "fake"
        srv._tokenizer = "fake"
        srv._model_name = "test"

        result = handle_tool_call("release_memory", {"fraction": 1.0})
        assert "Released all" in result["text"]
        assert srv._model is None
        assert srv._tokenizer is None
        assert srv._model_name is None

    def test_release_memory_partial(self):
        result = handle_tool_call("release_memory", {"fraction": 0.5})
        assert "not yet implemented" in result["text"]

    def test_release_memory_default_fraction(self):
        result = handle_tool_call("release_memory", {})
        # default fraction is 0.5 (< 1.0), so partial release
        assert "not yet implemented" in result["text"]

    @patch("mlx_flash_compress.mcp_server.detect_hardware")
    @patch("mlx_flash_compress.mcp_server.get_memory_state")
    def test_check_memory(self, mock_mem, mock_hw):
        mock_hw.return_value = MagicMock(chip="M2 Ultra", total_ram_gb=192.0)
        mock_mem.return_value = MagicMock(available_gb=100.0, pressure_level="nominal")

        import mlx_flash_compress.mcp_server as srv

        srv._model_name = None

        result = handle_tool_call("check_memory", {})
        assert result["type"] == "text"
        data = json.loads(result["text"])
        assert data["chip"] == "M2 Ultra"
        assert data["total_ram_gb"] == 192.0
        assert data["available_gb"] == 100.0
        assert data["pressure"] == "nominal"
