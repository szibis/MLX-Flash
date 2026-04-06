"""Tests for MCP tool definitions."""

import json
import pytest

from mlx_flash_compress.mcp_tools import (
    MCP_TOOLS,
    get_mcp_manifest,
    get_tool_names,
    format_mcp_response,
)


class TestMCPTools:
    def test_all_tools_have_required_fields(self):
        for tool in MCP_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_tool_names(self):
        names = get_tool_names()
        assert "generate" in names
        assert "check_memory" in names
        assert "switch_model" in names
        assert "release_memory" in names
        assert "list_models" in names
        assert "get_status" in names

    def test_generate_schema(self):
        gen = next(t for t in MCP_TOOLS if t["name"] == "generate")
        props = gen["inputSchema"]["properties"]
        assert "prompt" in props
        assert "max_tokens" in props
        assert "temperature" in props
        assert gen["inputSchema"]["required"] == ["prompt"]

    def test_switch_model_requires_model(self):
        sw = next(t for t in MCP_TOOLS if t["name"] == "switch_model")
        assert sw["inputSchema"]["required"] == ["model"]

    def test_check_memory_no_required(self):
        cm = next(t for t in MCP_TOOLS if t["name"] == "check_memory")
        assert "required" not in cm["inputSchema"]


class TestMCPManifest:
    def test_manifest_structure(self):
        m = get_mcp_manifest()
        assert m["name"] == "mlx-flash"
        assert "version" in m
        assert "tools" in m
        assert len(m["tools"]) >= 6

    def test_manifest_is_json_serializable(self):
        m = get_mcp_manifest()
        serialized = json.dumps(m)
        assert len(serialized) > 100


class TestMCPResponse:
    def test_format_response(self):
        r = format_mcp_response("generate", {"text": "Hello world"})
        assert r["type"] == "tool_result"
        assert r["tool_use_id"] == "generate"
        assert "Hello world" in r["content"]
