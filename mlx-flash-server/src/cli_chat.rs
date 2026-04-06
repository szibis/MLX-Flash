//! CLI interactive chat — shares the same Rust server API as the web chat UI.
//!
//! Usage:
//!   mlx-flash-server --chat             # connect to running server on :8080
//!   mlx-flash-server --chat --port 9090 # connect to server on :9090
//!
//! All slash commands work the same as in the web chat UI, routing through
//! the server's /commands/run and /v1/models/switch endpoints.

use std::io::{self, Write, BufRead};

pub async fn run_chat(port: u16) {
    println!("\n  \x1b[36m╔══════════════════════════════════════╗\x1b[0m");
    println!("  \x1b[36m║\x1b[0m\x1b[1m     ⚡ MLX-Flash CLI Chat ⚡         \x1b[0m\x1b[36m║\x1b[0m");
    println!("  \x1b[36m╚══════════════════════════════════════╝\x1b[0m");

    let base = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .unwrap();

    // Connect + show model
    println!("  \x1b[2mConnecting to {}...\x1b[0m", base);
    match client.get(format!("{}/status", base)).send().await {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                let model = json["model"].as_str().unwrap_or("unknown");
                let mem = &json["memory"];
                let avail = mem["free_gb"].as_f64().unwrap_or(0.0) + mem["inactive_gb"].as_f64().unwrap_or(0.0) * 0.5;
                let pressure = mem["pressure"].as_str().unwrap_or("unknown");
                println!("  \x1b[32m✓\x1b[0m Connected");
                println!("  \x1b[2m  Model:    \x1b[0m\x1b[1m{}\x1b[0m", model);
                println!("  \x1b[2m  Memory:   {:.1}GB available, pressure: {}\x1b[0m", avail, pressure);
                let workers = &json["workers"];
                let healthy = workers["healthy_count"].as_u64().unwrap_or(0);
                let total = workers["total_count"].as_u64().unwrap_or(0);
                println!("  \x1b[2m  Workers:  {}/{} healthy\x1b[0m", healthy, total);
            }
        }
        _ => {
            println!("  \x1b[31m✗\x1b[0m Server not running on port {}", port);
            println!("  \x1b[2m  Start: mlx-flash-server --port {}\x1b[0m", port);
            return;
        }
    }

    println!("\n  \x1b[2mType a message, or / for commands. /help for all commands. /quit to exit.\x1b[0m\n");

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

        // Slash commands — route through server's /commands/run
        if line.starts_with("/") {
            let cmd = line.to_lowercase();

            // /clear is handled locally (conversation state is client-side)
            if cmd == "/clear" {
                messages = vec![serde_json::json!({"role": "system", "content": "You are a helpful AI assistant."})];
                println!("  \x1b[32m✓\x1b[0m Conversation cleared\n");
                continue;
            }

            // /model <name> — use /v1/models/switch
            if cmd.starts_with("/model ") && cmd != "/models" {
                let model_name = line[7..].trim();
                print!("  \x1b[2mSwitching to {}...\x1b[0m", model_name);
                io::stdout().flush().ok();
                let switch_body = serde_json::json!({"model": model_name});
                match client.post(format!("{}/v1/models/switch", base))
                    .json(&switch_body).send().await
                {
                    Ok(resp) => {
                        if let Ok(json) = resp.json::<serde_json::Value>().await {
                            if json["switched"].as_bool().unwrap_or(false) {
                                println!("\r  \x1b[32m✓\x1b[0m Switched to \x1b[1m{}\x1b[0m\n", json["model"].as_str().unwrap_or(model_name));
                            } else {
                                println!("\r  \x1b[31m✗\x1b[0m Switch failed: {}\n", json["error"].as_str().unwrap_or("unknown"));
                            }
                        }
                    }
                    Err(e) => println!("\r  \x1b[31m✗\x1b[0m Error: {}\n", e),
                }
                continue;
            }

            // All other commands → /commands/run
            let cmd_body = serde_json::json!({"command": &line});
            match client.post(format!("{}/commands/run", base))
                .json(&cmd_body).send().await
            {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        let cmd_type = json["type"].as_str().unwrap_or("unknown");
                        match cmd_type {
                            "status" => {
                                let model = json["model"].as_str().unwrap_or("?");
                                let mem = &json["memory"];
                                let avail = mem["available_gb"].as_f64().unwrap_or(0.0);
                                let pressure = mem["pressure"].as_str().unwrap_or("?");
                                let swap = mem["swap_gb"].as_f64().unwrap_or(0.0);
                                let reqs = json["requests"].as_u64().unwrap_or(0);
                                let toks = json["tokens_generated"].as_u64().unwrap_or(0);
                                let uptime = json["uptime_s"].as_u64().unwrap_or(0);
                                let workers = &json["workers"];
                                println!("  \x1b[1m┌─ Status ─────────────────────────┐\x1b[0m");
                                println!("  \x1b[1m│\x1b[0m Model:      \x1b[1m{}\x1b[0m", model);
                                println!("  \x1b[1m│\x1b[0m Memory:     \x1b[{}m{:.1}GB available\x1b[0m (pressure: {})",
                                    if pressure == "Critical" { "31" } else if pressure == "Warning" { "33" } else { "32" },
                                    avail, pressure);
                                if swap > 0.1 { println!("  \x1b[1m│\x1b[0m Swap:       \x1b[31m{:.1}GB in use\x1b[0m", swap); }
                                println!("  \x1b[1m│\x1b[0m Workers:    {}/{} healthy, {} sessions",
                                    workers["healthy"].as_u64().unwrap_or(0),
                                    workers["total"].as_u64().unwrap_or(0),
                                    workers["sessions"].as_u64().unwrap_or(0));
                                println!("  \x1b[1m│\x1b[0m Requests:   {}  Tokens: {}  Uptime: {}m",
                                    reqs, toks, uptime / 60);
                                println!("  \x1b[1m└──────────────────────────────────┘\x1b[0m\n");
                            }
                            "models" => {
                                let current = json["current"].as_str().unwrap_or("?");
                                println!("  \x1b[1m┌─ Available Models ───────────────┐\x1b[0m");
                                if let Some(models) = json["available"].as_array() {
                                    for (i, m) in models.iter().enumerate() {
                                        let id = m["id"].as_str().unwrap_or("?");
                                        let label = m["label"].as_str().unwrap_or("?");
                                        let size = m["size_gb"].as_f64().unwrap_or(0.0);
                                        let active = if id == current { " \x1b[32m◀ active\x1b[0m" } else { "" };
                                        println!("  \x1b[1m│\x1b[0m \x1b[36m{:>2}.\x1b[0m {} ({:.0}GB){}", i + 1, label, size, active);
                                    }
                                }
                                println!("  \x1b[1m│\x1b[0m");
                                println!("  \x1b[1m│\x1b[0m \x1b[2mSwitch: /model <number> or /model <name>\x1b[0m");
                                println!("  \x1b[1m└──────────────────────────────────┘\x1b[0m\n");
                            }
                            "help" => {
                                println!("  \x1b[1m┌─ Commands ───────────────────────┐\x1b[0m");
                                if let Some(cmds) = json["commands"].as_object() {
                                    for (k, v) in cmds {
                                        println!("  \x1b[1m│\x1b[0m \x1b[36m{:<20}\x1b[0m {}", k, v.as_str().unwrap_or(""));
                                    }
                                }
                                println!("  \x1b[1m│\x1b[0m \x1b[36m{:<20}\x1b[0m Exit the chat", "/quit");
                                println!("  \x1b[1m└──────────────────────────────────┘\x1b[0m\n");
                            }
                            "workers" => {
                                println!("  \x1b[1m┌─ Workers ────────────────────────┐\x1b[0m");
                                if let Some(workers) = json["workers"].as_array() {
                                    for w in workers {
                                        let port = w["port"].as_u64().unwrap_or(0);
                                        let healthy = w["healthy"].as_bool().unwrap_or(false);
                                        let inflight = w["inflight"].as_u64().unwrap_or(0);
                                        let total = w["total_requests"].as_u64().unwrap_or(0);
                                        let dot = if healthy { "\x1b[32m●\x1b[0m" } else { "\x1b[31m●\x1b[0m" };
                                        println!("  \x1b[1m│\x1b[0m {} :{} — {} inflight, {} total", dot, port, inflight, total);
                                    }
                                }
                                println!("  \x1b[1m└──────────────────────────────────┘\x1b[0m\n");
                            }
                            "error" => {
                                println!("  \x1b[31m{}\x1b[0m\n", json["error"].as_str().unwrap_or("Unknown error"));
                            }
                            _ => {
                                println!("  \x1b[2m{}\x1b[0m\n", serde_json::to_string_pretty(&json).unwrap_or_default());
                            }
                        }
                    }
                }
                Err(e) => println!("  \x1b[31mError: {}\x1b[0m\n", e),
            }
            continue;
        }

        // Regular message — send to /v1/chat/completions
        messages.push(serde_json::json!({"role": "user", "content": line}));

        let model = match client.get(format!("{}/status", base)).send().await {
            Ok(resp) => {
                resp.json::<serde_json::Value>().await
                    .ok()
                    .and_then(|st| st["model"].as_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| "local".to_string())
            }
            Err(_) => "local".to_string(),
        };

        let payload = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
            "stream": false,
        });

        print!("  \x1b[36m⚡ AI:\x1b[0m ");
        io::stdout().flush().ok();

        let t0 = std::time::Instant::now();
        match client.post(format!("{}/v1/chat/completions", base))
            .json(&payload)
            .send()
            .await
        {
            Ok(resp) => {
                if resp.status().is_success() {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        let content = json["choices"][0]["message"]["content"]
                            .as_str().unwrap_or("No response");
                        let elapsed = t0.elapsed().as_secs_f64();
                        let mlx = &json["mlx_flash_compress"];
                        let tps = mlx["tok_per_s"].as_f64().unwrap_or(0.0);
                        let usage = &json["usage"];
                        let tok_count = usage["completion_tokens"].as_u64().unwrap_or(0);

                        println!("{}", content);
                        let mut stats = vec![format!("{:.1}s", elapsed)];
                        if tps > 0.0 { stats.push(format!("{:.0} tok/s", tps)); }
                        if tok_count > 0 { stats.push(format!("{} tokens", tok_count)); }
                        println!("  \x1b[2m  [{}]\x1b[0m\n", stats.join(", "));

                        messages.push(serde_json::json!({"role": "assistant", "content": content}));
                    } else {
                        println!("\x1b[31mFailed to parse response\x1b[0m\n");
                    }
                } else {
                    let status = resp.status().as_u16();
                    let body = resp.text().await.unwrap_or_default();
                    if status == 502 {
                        println!("\x1b[31mPython worker not running\x1b[0m");
                        println!("  \x1b[2mStart: mlx-flash-server --port {} --preload\x1b[0m\n", port);
                    } else {
                        println!("\x1b[31mError {}: {}\x1b[0m\n", status, &body[..body.len().min(200)]);
                    }
                    messages.pop();
                }
            }
            Err(e) => {
                println!("\x1b[31mConnection error: {}\x1b[0m\n", e);
                messages.pop();
            }
        }
    }
}
