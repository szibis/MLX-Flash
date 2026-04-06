mod cache;
mod expert_store;
mod mcp;
mod memory;
mod protocol;
mod proxy;
mod server;
mod socket_server;

use clap::Parser;
use std::io::BufRead as _;
use std::process::{Child, Command};
use std::sync::Arc;
use tokio::net::TcpListener;

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
    #[arg(long, help = "Launch Python worker automatically")]
    launch_worker: bool,
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

fn launch_python_worker(port: u16, model: &str, preload: bool) -> Option<Child> {
    let mut cmd = Command::new("python3");
    cmd.args([
        "-m", "mlx_flash_compress.serve",
        "--port", &port.to_string(),
        "--host", "127.0.0.1",
        "--model", model,
    ]);
    if preload {
        cmd.arg("--preload");
    }
    match cmd.spawn() {
        Ok(child) => {
            tracing::info!("Python worker started (PID {})", child.id());
            Some(child)
        }
        Err(e) => {
            tracing::error!("Failed to start Python worker: {}", e);
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

    tracing_subscriber::fmt::init();

    match memory::get_memory_state() {
        Ok(mem) => tracing::info!(
            "Memory: {:.1}GB total, {:.1}GB free, pressure: {:?}",
            mem.total_gb, mem.free_gb, mem.pressure
        ),
        Err(e) => tracing::warn!("Could not read memory state: {}", e),
    }

    let mut _worker: Option<Child> = None;
    if args.launch_worker {
        _worker = launch_python_worker(args.python_port, &args.model, args.preload);
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }

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

    let state = server::AppState {
        python_port: args.python_port,
        model_name: args.model,
        cache: cache_arc,
        ..Default::default()
    };

    let app = server::create_router(state);
    let addr = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&addr).await.unwrap();
    tracing::info!("Listening on http://{}", addr);
    tracing::info!("Python worker expected at http://127.0.0.1:{}", args.python_port);

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            tokio::signal::ctrl_c().await.ok();
        })
        .await
        .unwrap();

    tracing::info!("Shutting down");
}
