pub mod memory;
pub mod proxy;
pub mod server;

use clap::Parser;

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

#[tokio::main]
async fn main() {
    let args = Args::parse();
    tracing_subscriber::fmt::init();
    tracing::info!(
        "mlx-flash-server v{} starting on {}:{}",
        env!("CARGO_PKG_VERSION"), args.host, args.port
    );
}
