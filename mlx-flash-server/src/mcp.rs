//! MCP (Model Context Protocol) stdio server for Claude Code / Codex integration.
//!
//! When launched with `--mcp`, the server runs in MCP stdio mode instead of HTTP.
//! JSON-RPC 2.0 messages arrive on stdin, responses go to stdout.
//!
//! This allows Claude Code and Codex to discover and call MLX-Flash tools
//! without any manual API configuration.

use serde_json::{json, Value};

/// MCP tool definitions matching mlx_flash_compress/mcp_tools.py
pub fn tool_list() -> Value {
    json!({
        "tools": [
            {
                "name": "generate",
                "description": "Generate text completion from the loaded MLX model",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "The input text prompt"},
                        "max_tokens": {"type": "integer", "description": "Maximum tokens to generate", "default": 256},
                        "temperature": {"type": "number", "description": "Sampling temperature", "default": 0.7},
                        "system": {"type": "string", "description": "System prompt", "default": "You are a helpful assistant."}
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "check_memory",
                "description": "Check current memory pressure, cache stats, and available RAM",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "switch_model",
                "description": "Switch to a different MLX model (downloads if needed)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string", "description": "HuggingFace model ID"}
                    },
                    "required": ["model"]
                }
            },
            {
                "name": "release_memory",
                "description": "Release cached expert weights to free RAM",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "fraction": {"type": "number", "description": "Fraction of cache to release (0.0-1.0)", "default": 0.5}
                    }
                }
            },
            {
                "name": "list_models",
                "description": "List available models with size info and hardware compatibility",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "get_status",
                "description": "Get full server status: model, hardware, memory, optimization hints",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]
    })
}

/// Handle a JSON-RPC 2.0 request for MCP protocol
pub fn handle_request(request: &Value, python_port: u16) -> Option<Value> {
    let method = request["method"].as_str().unwrap_or("");
    let id = &request["id"];
    let params = &request["params"];

    match method {
        "initialize" => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "mlx-flash",
                    "version": "0.6.2",
                    "description": "Run AI models too large for your Mac's memory at near-full speed"
                },
                "capabilities": {
                    "tools": {"listChanged": false}
                }
            }
        })),

        "notifications/initialized" => None, // no response for notifications

        "tools/list" => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": tool_list()
        })),

        "tools/call" => {
            let tool_name = params["name"].as_str().unwrap_or("");
            let _arguments = &params["arguments"];

            // For tool calls, proxy to the Python worker
            let result = match tool_name {
                "check_memory" => {
                    match crate::memory::get_memory_state() {
                        Ok(mem) => json!({
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string_pretty(&mem).unwrap_or_default()
                            }]
                        }),
                        Err(e) => json!({
                            "content": [{"type": "text", "text": format!("Error: {e}")}],
                            "isError": true
                        }),
                    }
                },
                "get_status" => {
                    match crate::memory::get_memory_state() {
                        Ok(mem) => json!({
                            "content": [{
                                "type": "text",
                                "text": format!(
                                    "Memory: {:.1}GB total, {:.1}GB available, pressure: {:?}",
                                    mem.total_gb, mem.available_gb(),
                                    mem.pressure
                                )
                            }]
                        }),
                        Err(e) => json!({
                            "content": [{"type": "text", "text": format!("Error: {e}")}],
                            "isError": true
                        }),
                    }
                },
                // generate, switch_model, release_memory, list_models
                // → proxy to Python worker at python_port
                _ => json!({
                    "content": [{
                        "type": "text",
                        "text": format!(
                            "Tool '{}' requires Python worker. Proxy to http://127.0.0.1:{}/v1/chat/completions",
                            tool_name, python_port
                        )
                    }]
                }),
            };

            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result
            }))
        },

        _ => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": -32601, "message": format!("Method not found: {method}")}
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_list_has_6_tools() {
        let list = tool_list();
        let tools = list["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 6);
    }

    #[test]
    fn test_tool_names() {
        let list = tool_list();
        let names: Vec<&str> = list["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"generate"));
        assert!(names.contains(&"check_memory"));
        assert!(names.contains(&"switch_model"));
    }

    #[test]
    fn test_initialize() {
        let req = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}});
        let resp = handle_request(&req, 8081).unwrap();
        assert_eq!(resp["result"]["serverInfo"]["name"], "mlx-flash");
        assert!(resp["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn test_tools_list() {
        let req = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}});
        let resp = handle_request(&req, 8081).unwrap();
        assert!(resp["result"]["tools"].is_array());
    }

    #[test]
    fn test_tools_call_check_memory() {
        let req = json!({
            "jsonrpc": "2.0", "id": 3,
            "method": "tools/call",
            "params": {"name": "check_memory", "arguments": {}}
        });
        let resp = handle_request(&req, 8081).unwrap();
        assert!(resp["result"]["content"].is_array());
    }

    #[test]
    fn test_notification_returns_none() {
        let req = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        let resp = handle_request(&req, 8081);
        assert!(resp.is_none());
    }

    #[test]
    fn test_unknown_method() {
        let req = json!({"jsonrpc": "2.0", "id": 4, "method": "unknown", "params": {}});
        let resp = handle_request(&req, 8081).unwrap();
        assert!(resp["error"].is_object());
    }
}
