# MLX-Flash Logging

Both the Rust proxy and Python workers emit structured logs in a unified format. Logs go to stdout by default (cloud-native) and optionally to a file.

## Quick Start

```bash
# Text format (default, human-readable)
mlx-flash --port 8080 --launch-worker

# JSON format (for Loki, Vector, Datadog, ELK)
mlx-flash --port 8080 --launch-worker --log-format json

# JSON + file output
mlx-flash --port 8080 --log-format json --log-file /var/log/mlx-flash.log

# Python worker standalone
python -m mlx_flash_compress.serve --port 8081 --log-format json --log-file /var/log/mlx-flash-worker.log
```

## Environment Variables

Override CLI flags via environment:

| Variable | Values | Description |
|----------|--------|-------------|
| `MLX_FLASH_LOG_FORMAT` | `text`, `json` | Log format (overrides `--log-format`) |
| `MLX_FLASH_LOG_FILE` | path | Log file (overrides `--log-file`) |
| `MLX_FLASH_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | Python log level |
| `RUST_LOG` | `info`, `debug`, `warn`, `mlx_flash_server=debug` | Rust log level (tracing-subscriber EnvFilter) |

## Log Format

### JSON (for machines)

Every log line is a single JSON object:

```json
{"timestamp":"2026-04-06T14:32:01.123Z","level":"info","component":"rust-proxy","target":"mlx_flash_server::server","message":"Listening on http://127.0.0.1:8080"}
{"timestamp":"2026-04-06T14:32:01.456Z","level":"info","component":"python-worker","worker_port":8081,"message":"Model loaded","model":"mlx-community/Qwen3-30B-A3B-4bit","load_time_s":4.2}
{"timestamp":"2026-04-06T14:32:05.789Z","level":"info","component":"python-worker","worker_port":8081,"message":"Inference complete","tokens":128,"tok_per_s":82.6,"latency_ms":1549}
{"timestamp":"2026-04-06T14:32:06.001Z","level":"warning","component":"rust-proxy","message":"Worker on port 8082 not yet responding","worker_port":8082}
```

### Text (for humans)

```
2026-04-06T14:32:01Z INFO  rust-proxy: Listening on http://127.0.0.1:8080
2026-04-06T14:32:01Z INFO  python-worker :8081: Model loaded model=mlx-community/Qwen3-30B-A3B-4bit load_time_s=4.2
2026-04-06T14:32:05Z INFO  python-worker :8081: Inference complete tokens=128 tok_per_s=82.6
2026-04-06T14:32:06Z WARN  rust-proxy: Worker on port 8082 not yet responding
```

## Unified Fields

Both Rust and Python components emit these fields:

| Field | Type | Present | Description |
|-------|------|---------|-------------|
| `timestamp` | ISO 8601 | always | UTC timestamp |
| `level` | string | always | `debug`, `info`, `warning`, `error` |
| `component` | string | always | `rust-proxy` or `python-worker` |
| `target` | string | always | Module path (Rust: `mlx_flash_server::server`, Python: `mlx_flash`) |
| `message` | string | always | Human-readable message |
| `worker_port` | int | when applicable | Which Python worker port |
| `model` | string | model events | Model ID |
| `load_time_s` | float | load events | Time to load/warm-up in seconds |
| `tokens` | int | inference events | Tokens generated |
| `tok_per_s` | float | inference events | Token throughput |
| `pressure` | string | memory events | `normal`, `warning`, `critical` |
| `memory_gb` | float | memory events | Available RAM in GB |
| `action` | string | lifecycle events | `server_start`, `model_load_start`, `model_load_complete`, `warmup_start`, `warmup_complete`, `model_switch`, `server_stop` |
| `error` | string | error events | Error description |

## Collecting Logs

### With Vector (recommended for Loki)

```toml
[sources.mlx_flash]
type = "file"
include = ["/var/log/mlx-flash*.log"]

[transforms.parse]
type = "remap"
inputs = ["mlx_flash"]
source = '. = parse_json!(.message)'

[sinks.loki]
type = "loki"
inputs = ["parse"]
endpoint = "http://loki:3100"
labels.job = "mlx-flash"
labels.component = "{{ component }}"
```

### With Docker Compose (stdout → Loki)

When running via `docker compose --profile monitoring up`, logs go to stdout and can be collected by any Docker log driver (json-file, fluentd, loki).

### With Grafana + Loki

Query logs by component:
```logql
{job="mlx-flash"} | json | component="python-worker"
{job="mlx-flash"} | json | level="error"
{job="mlx-flash"} | json | action="model_load_complete" | line_format "{{.model}} loaded in {{.load_time_s}}s"
```

## Output Destinations

| Destination | How | Config |
|-------------|-----|--------|
| **stdout** (default) | Built-in, always on | No config needed |
| **File** | `--log-file /path/to/file.log` | Both stdout + file |
| **Loki** | Vector/Promtail reads file or Docker stdout | See Vector config above |
| **Datadog** | JSON format → Datadog Agent | `--log-format json` |
| **CloudWatch** | JSON stdout → CloudWatch Logs agent | `--log-format json` |
| **ELK** | JSON → Filebeat → Elasticsearch | `--log-format json` |
