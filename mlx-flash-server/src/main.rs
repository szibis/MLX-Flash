mod cache;
mod chat_ui;
mod cli_chat;
mod dashboard;
mod expert_store;
mod log_buffer;
mod mcp;
mod memory;
mod protocol;
mod proxy;
mod server;
mod socket_server;
mod worker_pool;

use clap::Parser;
use std::io::BufRead as _;
use std::net::TcpStream;
use std::process::{Child, Command};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "mlx-flash-server", about = "Rust sidecar for MLX-Flash-Compress")]
struct Args {
    #[arg(long, default_value = "8080")]
    port: u16,
    #[arg(long, default_value = "8081")]
    python_port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value = "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit")]
    model: String,
    #[arg(long, default_value = "true", help = "Launch Python workers automatically (use --no-launch-worker to disable)")]
    launch_worker: bool,
    #[arg(long, help = "Don't launch Python workers (connect to existing)")]
    no_launch_worker: bool,
    #[arg(long, help = "Preload model in Python worker")]
    preload: bool,
    #[arg(long, help = "Directory with expert weight files")]
    expert_dir: Option<String>,
    #[arg(long, default_value = "512", help = "Cache size in MB")]
    cache_mb: u32,
    #[arg(long, default_value = "/tmp/mlx-flash-cache.sock", help = "Unix socket path for expert cache")]
    socket_path: String,
    #[arg(long, help = "Run as MCP stdio server for Claude Code / Codex")]
    mcp: bool,
    #[arg(long, help = "Start interactive CLI chat (connects to local server)")]
    chat: bool,
    #[arg(long, default_value = "1", help = "Number of Python inference workers")]
    workers: usize,
    #[arg(long, help = "Path to Python interpreter (auto-detects venv if not set)")]
    python: Option<String>,
    #[arg(long, default_value = "text", help = "Log format: text or json")]
    log_format: String,
    #[arg(long, help = "Log file path (also logs to stdout)")]
    log_file: Option<String>,
}

fn run_mcp_stdio(python_port: u16) {
    let stdin = std::io::stdin();
    let reader = stdin.lock();
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let request: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(response) = mcp::handle_request(&request, python_port) {
            println!("{}", serde_json::to_string(&response).unwrap_or_default());
        }
    }
}

/// Check if a port is already in use by attempting a TCP connect.
fn is_port_in_use(port: u16) -> bool {
    TcpStream::connect(("127.0.0.1", port)).is_ok()
}

/// Probe the /health endpoint to check if the existing listener is our worker.
async fn is_our_worker(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap();
    match client.get(&url).send().await {
        Ok(resp) => {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                // Our serve.py returns "model" in the status/health response
                body.get("model").is_some()
            } else {
                false
            }
        }
        Err(_) => false,
    }
}

/// Wait for the Python worker to become ready, polling /health.
async fn wait_for_worker(port: u16, timeout_secs: u64) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    while std::time::Instant::now() < deadline {
        if is_our_worker(port).await {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    }
    false
}

/// Find the best Python interpreter: explicit --python, then venv, then system.
fn find_python(explicit: &Option<String>) -> String {
    // 1. Explicit --python flag
    if let Some(p) = explicit {
        return p.clone();
    }

    // 2. VIRTUAL_ENV environment variable
    if let Ok(venv) = std::env::var("VIRTUAL_ENV") {
        let venv_python = format!("{}/bin/python3", venv);
        if std::path::Path::new(&venv_python).exists() {
            return venv_python;
        }
    }

    // 3. Auto-detect .venv* in current dir or parent dirs
    let cwd = std::env::current_dir().unwrap_or_default();
    for dir in [&cwd, &cwd.join(".."), &cwd.join("../..")]  {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with(".venv") || name == "venv" {
                    let candidate = entry.path().join("bin/python3");
                    if candidate.exists() {
                        tracing::info!("Auto-detected Python venv: {}", candidate.display());
                        return candidate.to_string_lossy().to_string();
                    }
                }
            }
        }
    }

    // 4. System python3
    "python3".to_string()
}

fn check_python_module(python: &str) -> bool {
    let project_root = std::env::current_dir().unwrap_or_default();
    Command::new(python)
        .env("PYTHONPATH", &project_root)
        .args(["-c", "import mlx_flash_compress"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn launch_python_worker(port: u16, model: &str, preload: bool, python: &str) -> Option<Child> {
    let mut cmd = Command::new(python);
    let project_root = std::env::current_dir().unwrap_or_default();
    cmd.env("PYTHONPATH", &project_root);
    cmd.args([
        "-m", "mlx_flash_compress.serve",
        "--port", &port.to_string(),
        "--host", "127.0.0.1",
        "--model", model,
    ]);
    if preload {
        cmd.arg("--preload");
    }
    tracing::info!("Launching Python worker: {} -m mlx_flash_compress.serve --port {}", python, port);
    match cmd.spawn() {
        Ok(child) => {
            tracing::info!("Python worker started (PID {}) on port {}", child.id(), port);
            Some(child)
        }
        Err(e) => {
            tracing::error!("Failed to start Python worker: {} (python={})", e, python);
            None
        }
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // MCP stdio mode — no HTTP, just JSON-RPC over stdin/stdout
    if args.mcp {
        run_mcp_stdio(args.python_port);
        return;
    }

    // CLI chat mode — connects to running server
    if args.chat {
        cli_chat::run_chat(args.port).await;
        return;
    }

    // Log buffer for /logs/recent endpoint (dashboard live logs)
    let log_buf = log_buffer::LogBuffer::new();
    let buf_layer = log_buffer::BufferLayer::new(log_buf.clone());

    // Structured logging: JSON or text, stdout + optional file
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    if args.log_format == "json" {
        if let Some(ref log_path) = args.log_file {
            let dir = std::path::Path::new(log_path).parent().unwrap_or(std::path::Path::new("."));
            let filename = std::path::Path::new(log_path).file_name().unwrap_or_default();
            let file_appender = tracing_appender::rolling::never(dir, filename);
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
            tracing_subscriber::registry()
                .with(env_filter)
                .with(buf_layer)
                .with(tracing_subscriber::fmt::layer().json()
                    .with_target(true)
                    .with_thread_ids(true)
                    .flatten_event(true))
                .with(tracing_subscriber::fmt::layer().json()
                    .with_writer(non_blocking)
                    .with_target(true)
                    .flatten_event(true))
                .init();
            Box::leak(Box::new(_guard));
        } else {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(buf_layer)
                .with(tracing_subscriber::fmt::layer().json()
                    .with_target(true)
                    .with_thread_ids(true)
                    .flatten_event(true))
                .init();
        }
    } else {
        if let Some(ref log_path) = args.log_file {
            let dir = std::path::Path::new(log_path).parent().unwrap_or(std::path::Path::new("."));
            let filename = std::path::Path::new(log_path).file_name().unwrap_or_default();
            let file_appender = tracing_appender::rolling::never(dir, filename);
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
            tracing_subscriber::registry()
                .with(env_filter)
                .with(buf_layer)
                .with(tracing_subscriber::fmt::layer())
                .with(tracing_subscriber::fmt::layer().with_writer(non_blocking).with_ansi(false))
                .init();
            Box::leak(Box::new(_guard));
        } else {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(buf_layer)
                .with(tracing_subscriber::fmt::layer())
                .init();
        }
    }

    match memory::get_memory_state() {
        Ok(mem) => tracing::info!(
            "Memory: {:.1}GB total, {:.1}GB free, pressure: {:?}",
            mem.total_gb, mem.free_gb, mem.pressure
        ),
        Err(e) => tracing::warn!("Could not read memory state: {}", e),
    }

    let python_path = find_python(&args.python);
    tracing::info!("Python interpreter: {}", python_path);

    let mut _workers: Vec<Child> = Vec::new();
    let worker_count = args.workers.max(1);

    let should_launch = args.launch_worker && !args.no_launch_worker;
    if should_launch {
        // Pre-flight: verify Python can import mlx_flash_compress
        if !check_python_module(&python_path) {
            tracing::error!(
                "Python at '{}' cannot import mlx_flash_compress. Install it:\n\
                 \n  pip install mlx-flash          # from PyPI\n\
                 \n  pip install -e .               # from source\n\
                 \n  brew install mlx-flash         # via Homebrew\n\
                 \nOr use --no-launch-worker to connect to an existing Python worker.",
                python_path
            );
            std::process::exit(1);
        }
        for i in 0..worker_count {
            let port = args.python_port + i as u16;
            if is_port_in_use(port) {
                if is_our_worker(port).await {
                    tracing::info!("Reusing existing MLX-Flash worker on port {}", port);
                    continue;
                } else {
                    tracing::error!(
                        "Port {} is already in use by another process. \
                         Free the port or use --python-port to pick a different base port.",
                        port
                    );
                    std::process::exit(1);
                }
            }

            if let Some(child) = launch_python_worker(port, &args.model, args.preload, &python_path) {
                _workers.push(child);
            }
        }

        // Wait for all workers to become ready
        let timeout = if args.preload { 120 } else { 15 };
        for i in 0..worker_count {
            let port = args.python_port + i as u16;
            tracing::info!("Waiting for worker on port {} (up to {}s)...", port, timeout);
            if wait_for_worker(port, timeout).await {
                tracing::info!("Worker on port {} is ready", port);
            } else {
                tracing::warn!(
                    "Worker on port {} not yet responding — may still be loading model",
                    port
                );
            }
        }
    }

    let pool = Arc::new(worker_pool::WorkerPool::new(args.python_port, worker_count));
    tracing::info!(
        "Worker pool: {} worker(s) on ports {}-{}, strategy: least-connections + cache-affinity",
        worker_count,
        args.python_port,
        args.python_port + worker_count as u16 - 1,
    );

    let cache_arc = if let Some(ref expert_dir) = args.expert_dir {
        let cache = Arc::new(cache::LcpCache::new(args.cache_mb as usize * 1024 * 1024));
        let store = Arc::new(expert_store::ExpertStore::new(std::path::PathBuf::from(expert_dir)));
        let socket_cache = cache.clone();
        let socket_store = store.clone();
        let socket_path = args.socket_path.clone();
        tokio::spawn(async move {
            socket_server::run_socket_server(&socket_path, socket_cache, socket_store).await;
        });
        tracing::info!("Expert cache: {}MB, socket: {}", args.cache_mb, args.socket_path);
        Some(cache)
    } else {
        None
    };

    let model_for_restart = args.model.clone();
    let preload_for_restart = args.preload;

    let state = server::AppState {
        python_port: args.python_port,
        model_name: std::sync::Arc::new(tokio::sync::RwLock::new(args.model)),
        cache: cache_arc,
        pool: pool.clone(),
        log_buffer: log_buf,
        ..Default::default()
    };

    // Background: periodic health check + auto-restart dead workers
    let health_pool = pool.clone();
    let health_model = model_for_restart.clone();
    let health_python = python_path.clone();
    let health_base_port = args.python_port;
    let health_preload = preload_for_restart;
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();

        loop {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;

            for port in health_pool.ports() {
                let url = format!("http://127.0.0.1:{port}/health");
                match client.get(&url).send().await {
                    Ok(resp) => {
                        if resp.status().is_success() {
                            if let Ok(body) = resp.json::<serde_json::Value>().await {
                                if body.get("model").is_some() {
                                    health_pool.mark_healthy(port);
                                    continue;
                                }
                            }
                        }
                        tracing::warn!(port, "Worker health check failed — marking unhealthy");
                        health_pool.mark_unhealthy(port);
                    }
                    Err(_) => {
                        tracing::warn!(port, "Worker unreachable — attempting auto-restart");
                        health_pool.mark_unhealthy(port);
                        // Auto-restart: launch a new worker on this port
                        if !is_port_in_use(port) {
                            if let Some(_child) = launch_python_worker(port, &health_model, health_preload, &health_python) {
                                tracing::info!(port, "Auto-restarted Python worker");
                                // Wait for it to come up
                                if wait_for_worker(port, 30).await {
                                    health_pool.mark_healthy(port);
                                    tracing::info!(port, "Auto-restarted worker is ready");
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    let app = server::create_router(state);
    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr).await.unwrap();
    tracing::info!("Listening on http://{}", addr);
    tracing::info!("Python worker expected at http://127.0.0.1:{}", args.python_port);
    tracing::info!("Health checks every 10s — unhealthy workers auto-restart");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await
        .unwrap();

    tracing::info!("Shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener as StdTcpListener;

    #[test]
    fn test_is_port_in_use_detects_occupied_port() {
        // Bind a port, then check it's detected as in use
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(is_port_in_use(port));
        drop(listener);
    }

    #[test]
    fn test_is_port_in_use_returns_false_for_free_port() {
        // Bind and immediately release to get a known-free port
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        assert!(!is_port_in_use(port));
    }

    #[tokio::test]
    async fn test_is_our_worker_returns_false_for_non_worker() {
        // Start a bare TCP listener (no HTTP) — should not be recognized as our worker
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        // Keep listener alive but don't serve HTTP
        assert!(!is_our_worker(port).await);
        drop(listener);
    }

    #[tokio::test]
    async fn test_is_our_worker_returns_false_for_closed_port() {
        assert!(!is_our_worker(19876).await);
    }

    #[tokio::test]
    async fn test_is_our_worker_recognizes_health_with_model() {
        // Spin up a minimal axum server that returns {"model": "test"} on /health
        let app = axum::Router::new().route(
            "/health",
            axum::routing::get(|| async {
                axum::Json(serde_json::json!({"model": "test-model"}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        // Give the server a moment to start
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(is_our_worker(port).await);
        handle.abort();
    }

    #[tokio::test]
    async fn test_is_our_worker_rejects_health_without_model() {
        // Server returns valid JSON but without "model" key
        let app = axum::Router::new().route(
            "/health",
            axum::routing::get(|| async {
                axum::Json(serde_json::json!({"status": "ok"}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!is_our_worker(port).await);
        handle.abort();
    }

    #[tokio::test]
    async fn test_wait_for_worker_times_out_on_closed_port() {
        // Should return false quickly (1s timeout, no server)
        let result = wait_for_worker(19877, 1).await;
        assert!(!result);
    }

    #[tokio::test]
    async fn test_wait_for_worker_succeeds_when_ready() {
        let app = axum::Router::new().route(
            "/health",
            axum::routing::get(|| async {
                axum::Json(serde_json::json!({"model": "test"}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let result = wait_for_worker(port, 5).await;
        assert!(result);
        handle.abort();
    }
}
