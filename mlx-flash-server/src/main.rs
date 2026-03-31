mod memory;
mod proxy;
mod server;

use clap::Parser;
use std::process::{Child, Command};
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

    let state = server::AppState {
        python_port: args.python_port,
        model_name: args.model,
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
