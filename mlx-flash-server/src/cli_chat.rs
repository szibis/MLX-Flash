//! CLI interactive chat that talks to the local Rust server's API.
//!
//! Usage: mlx-flash-server --chat [--port 8080]
//! Sends messages to /v1/chat/completions on the local server.

use std::io::{self, Write, BufRead};

pub async fn run_chat(port: u16) {
    println!("\n  \x1b[36m╔══════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m\x1b[1m     ⚡ MLX-Flash CLI Chat ⚡         \x1b[0m\x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚══════════════════════════════════════╝\x1b[0m");
    println!("  \x1b[2mConnecting to http://127.0.0.1:{}...\x1b[0m", port);

    // Check if server is running
    let client = reqwest::Client::new();
    match client.get(format!("http://127.0.0.1:{}/status", port)).send().await {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                let model = json["model"].as_str().unwrap_or("unknown");
                println!("  \x1b[32m✓\x1b[0m Connected — model: \x1b[1m{}\x1b[0m", model);
            }
        }
        _ => {
            println!("  \x1b[31m✗\x1b[0m Server not running on port {}", port);
            println!("  \x1b[2m  Start it: mlx-flash-server --port {} --launch-worker --preload\x1b[0m", port);
            return;
        }
    }

    println!("  \x1b[2mType a message. /quit to exit. /status for memory info.\x1b[0m\n");

    let mut messages: Vec<serde_json::Value> = vec![
        serde_json::json!({"role": "system", "content": "You are a helpful AI assistant."})
    ];

    let stdin = io::stdin();
    let reader = stdin.lock();
    let mut lines = reader.lines();

    loop {
        print!("  \x1b[32m▶ You:\x1b[0m ");
        io::stdout().flush().ok();

        let line = match lines.next() {
            Some(Ok(l)) => l.trim().to_string(),
            _ => break,
        };

        if line.is_empty() { continue; }
        if line == "/quit" || line == "/exit" { println!("  \x1b[36mGoodbye!\x1b[0m"); break; }

        if line == "/status" {
            match client.get(format!("http://127.0.0.1:{}/status", port)).send().await {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        let mem = &json["memory"];
                        let free = mem["free_gb"].as_f64().unwrap_or(0.0);
                        let inactive = mem["inactive_gb"].as_f64().unwrap_or(0.0);
                        let total = mem["total_gb"].as_f64().unwrap_or(0.0);
                        let avail = free + inactive * 0.5;
                        println!("  \x1b[2mMemory: {:.1}GB free / {:.0}GB total, pressure: {}\x1b[0m",
                            avail, total, mem["pressure"].as_str().unwrap_or("unknown"));
                    }
                }
                Err(e) => println!("  \x1b[31mError: {}\x1b[0m", e),
            }
            continue;
        }

        messages.push(serde_json::json!({"role": "user", "content": line}));

        let payload = serde_json::json!({
            "model": "local",
            "messages": messages,
            "max_tokens": 1024,
            "stream": false,
        });

        print!("  \x1b[36m⚡ AI:\x1b[0m ");
        io::stdout().flush().ok();

        match client.post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
            .json(&payload)
            .send()
            .await
        {
            Ok(resp) => {
                if resp.status().is_success() {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        let content = json["choices"][0]["message"]["content"]
                            .as_str().unwrap_or("No response");
                        println!("{}\n", content);
                        messages.push(serde_json::json!({"role": "assistant", "content": content}));
                    } else {
                        println!("\x1b[31mFailed to parse response\x1b[0m\n");
                    }
                } else {
                    let status = resp.status().as_u16();
                    let body = resp.text().await.unwrap_or_default();
                    if status == 502 {
                        println!("\x1b[31mPython worker not running\x1b[0m");
                        println!("  \x1b[2mStart: mlx-flash --port 8081 --preload\x1b[0m\n");
                    } else {
                        println!("\x1b[31mError {}: {}\x1b[0m\n", status, &body[..body.len().min(200)]);
                    }
                    messages.pop(); // remove failed user message
                }
            }
            Err(e) => {
                println!("\x1b[31mConnection error: {}\x1b[0m\n", e);
                messages.pop();
            }
        }
    }
}
