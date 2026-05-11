#!/usr/bin/env python3
"""Load test harness for the MLX-Flash continuous batching engine.

Sends concurrent requests to a running server and reports throughput,
latency percentiles, and error rate.

Usage:
    # Against a running server:
    python scripts/load_test_batching.py --url http://localhost:8080 --concurrent 50 --total 200

    # Lighter smoke test:
    python scripts/load_test_batching.py --url http://localhost:8080 --concurrent 10 --total 30

    # With custom prompt and token count:
    python scripts/load_test_batching.py --url http://localhost:8080 --total 100 \
        --prompt "Explain quantum computing" --max-tokens 128
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field


@dataclass
class RequestResult:
    """Outcome of a single load-test request."""

    success: bool
    latency_ms: float
    tokens: int = 0
    error: str = ""
    status_code: int = 0


@dataclass
class LoadTestReport:
    """Aggregated results from a load test run."""

    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    error_rate_pct: float = 0.0
    total_time_s: float = 0.0
    throughput_rps: float = 0.0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_mean_ms: float = 0.0
    errors: dict = field(default_factory=dict)


def send_request(
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> RequestResult:
    """Send a single chat completion request to the server."""
    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            latency_ms = (time.monotonic() - t0) * 1000
            tokens = body.get("usage", {}).get("completion_tokens", 0)
            return RequestResult(
                success=True,
                latency_ms=latency_ms,
                tokens=tokens,
                status_code=resp.status,
            )
    except urllib.error.HTTPError as e:
        latency_ms = (time.monotonic() - t0) * 1000
        error_body = ""
        try:
            error_body = e.read().decode()[:200]
        except Exception:
            pass
        return RequestResult(
            success=False,
            latency_ms=latency_ms,
            error=f"HTTP {e.code}: {error_body}",
            status_code=e.code,
        )
    except Exception as e:
        latency_ms = (time.monotonic() - t0) * 1000
        return RequestResult(
            success=False,
            latency_ms=latency_ms,
            error=str(e),
        )


def run_load_test(
    url: str,
    total: int,
    concurrent: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> LoadTestReport:
    """Run the load test and return aggregated results."""
    results: list[RequestResult] = []

    print(f"\nLoad test: {total} requests, {concurrent} concurrent")
    print(f"Target: {url}")
    print(f"Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    print("-" * 60)

    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        futures = [
            pool.submit(send_request, url, prompt, max_tokens, temperature, timeout)
            for _ in range(total)
        ]

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % max(1, total // 10) == 0 or completed == total:
                pct = completed * 100 // total
                print(f"  Progress: {completed}/{total} ({pct}%)", flush=True)

    t_end = time.monotonic()
    total_time = t_end - t_start

    # Build report
    report = LoadTestReport()
    report.total_requests = total
    report.total_time_s = round(total_time, 3)

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    report.successful = len(successes)
    report.failed = len(failures)
    report.error_rate_pct = round(len(failures) / total * 100, 2) if total > 0 else 0.0
    report.throughput_rps = round(total / total_time, 2) if total_time > 0 else 0.0

    # Token stats
    report.total_tokens = sum(r.tokens for r in successes)
    report.tokens_per_second = (
        round(report.total_tokens / total_time, 2) if total_time > 0 else 0.0
    )

    # Latency percentiles (from successful requests only)
    if successes:
        latencies = sorted(r.latency_ms for r in successes)
        report.latency_min_ms = round(latencies[0], 2)
        report.latency_max_ms = round(latencies[-1], 2)
        report.latency_mean_ms = round(statistics.mean(latencies), 2)

        def percentile(data, pct):
            k = (len(data) - 1) * (pct / 100)
            f = int(k)
            c = f + 1
            if c >= len(data):
                return data[f]
            return data[f] + (k - f) * (data[c] - data[f])

        report.latency_p50_ms = round(percentile(latencies, 50), 2)
        report.latency_p95_ms = round(percentile(latencies, 95), 2)
        report.latency_p99_ms = round(percentile(latencies, 99), 2)

    # Error breakdown
    if failures:
        error_counts: dict[str, int] = {}
        for r in failures:
            key = r.error[:80] if r.error else "unknown"
            error_counts[key] = error_counts.get(key, 0) + 1
        report.errors = error_counts

    return report


def print_report(report: LoadTestReport) -> None:
    """Print a formatted report table."""
    print("\n" + "=" * 60)
    print("  LOAD TEST RESULTS")
    print("=" * 60)
    print(f"  {'Total requests:':<30} {report.total_requests}")
    print(f"  {'Successful:':<30} {report.successful}")
    print(f"  {'Failed:':<30} {report.failed}")
    print(f"  {'Error rate:':<30} {report.error_rate_pct}%")
    print(f"  {'Total time:':<30} {report.total_time_s}s")
    print("-" * 60)
    print("  THROUGHPUT")
    print("-" * 60)
    print(f"  {'Requests/sec:':<30} {report.throughput_rps}")
    print(f"  {'Total tokens:':<30} {report.total_tokens}")
    print(f"  {'Tokens/sec:':<30} {report.tokens_per_second}")
    print("-" * 60)
    print("  LATENCY (successful requests)")
    print("-" * 60)
    print(f"  {'Min:':<30} {report.latency_min_ms} ms")
    print(f"  {'Mean:':<30} {report.latency_mean_ms} ms")
    print(f"  {'p50 (median):':<30} {report.latency_p50_ms} ms")
    print(f"  {'p95:':<30} {report.latency_p95_ms} ms")
    print(f"  {'p99:':<30} {report.latency_p99_ms} ms")
    print(f"  {'Max:':<30} {report.latency_max_ms} ms")

    if report.errors:
        print("-" * 60)
        print("  ERRORS")
        print("-" * 60)
        for err, count in sorted(report.errors.items(), key=lambda x: -x[1]):
            print(f"  [{count}x] {err}")

    print("=" * 60)


def check_server(url: str) -> bool:
    """Check if the server is reachable."""
    try:
        req = urllib.request.Request(f"{url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            print(f"Server OK: model={data.get('model', '?')}, "
                  f"loaded={data.get('model_loaded', '?')}")
            return True
    except Exception as e:
        print(f"Server unreachable at {url}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Load test for MLX-Flash continuous batching engine"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Server base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=50,
        help="Number of concurrent workers (default: 50)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=200,
        help="Total number of requests to send (default: 200)",
    )
    parser.add_argument(
        "--prompt",
        default="What is 2+2? Answer in one word.",
        help="Prompt to send in each request",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens per response (default: 32)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of table",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip server health check before running",
    )
    args = parser.parse_args()

    if not args.skip_check:
        if not check_server(args.url):
            print(
                "\nServer not reachable. Start it with:\n"
                f"  python -m mlx_flash_compress.serve --batching --port "
                f"{args.url.split(':')[-1] if ':' in args.url else '8080'}",
                file=sys.stderr,
            )
            sys.exit(1)

    report = run_load_test(
        url=args.url,
        total=args.total,
        concurrent=args.concurrent,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    if args.json:
        output = {
            "total_requests": report.total_requests,
            "successful": report.successful,
            "failed": report.failed,
            "error_rate_pct": report.error_rate_pct,
            "total_time_s": report.total_time_s,
            "throughput_rps": report.throughput_rps,
            "total_tokens": report.total_tokens,
            "tokens_per_second": report.tokens_per_second,
            "latency_ms": {
                "min": report.latency_min_ms,
                "mean": report.latency_mean_ms,
                "p50": report.latency_p50_ms,
                "p95": report.latency_p95_ms,
                "p99": report.latency_p99_ms,
                "max": report.latency_max_ms,
            },
            "errors": report.errors,
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report)

    # Exit with error if error rate is too high
    if report.error_rate_pct > 10:
        sys.exit(2)


if __name__ == "__main__":
    main()
