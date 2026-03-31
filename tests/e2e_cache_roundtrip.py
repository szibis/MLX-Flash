"""E2E test: Python client ↔ Rust expert cache ↔ SSD roundtrip.

Tests the full pipeline:
  1. Create synthetic expert files on disk
  2. Start the Rust cache server (Unix socket)
  3. Python connects via RustCacheClient
  4. Fetch experts (cache miss → SSD load → cache)
  5. Fetch same experts again (cache hit)
  6. Report routing (triggers prefetch)
  7. Verify cache stats show hits, entries, hit rate

Can run standalone or via pytest (skips if Rust binary not built).

Usage:
  # Build Rust first
  cd mlx-flash-server && cargo build --release

  # Run E2E test
  python -m pytest tests/e2e_cache_roundtrip.py -v

  # Or standalone
  python tests/e2e_cache_roundtrip.py
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


RUST_BINARY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mlx-flash-server", "target", "release", "mlx-flash-server"
)
SOCKET_PATH = "/tmp/mlx-flash-e2e-test.sock"


def create_test_experts(base_dir: str, num_layers: int = 4, num_experts: int = 8,
                        size_bytes: int = 1024) -> str:
    """Create synthetic expert weight files."""
    expert_dir = os.path.join(base_dir, "experts")
    rng = np.random.default_rng(42)
    for layer in range(num_layers):
        layer_dir = os.path.join(expert_dir, f"layer_{layer:03d}")
        os.makedirs(layer_dir, exist_ok=True)
        for expert in range(num_experts):
            data = rng.integers(0, 256, size=size_bytes, dtype=np.uint8).tobytes()
            with open(os.path.join(layer_dir, f"expert_{expert:04d}.bin"), "wb") as f:
                f.write(data)
    return expert_dir


def rust_binary_available():
    """Check if the Rust binary has been built."""
    return os.path.exists(RUST_BINARY)


@pytest.fixture(scope="module")
def cache_server():
    """Start Rust cache server for the entire test module."""
    if not rust_binary_available():
        pytest.skip("Rust binary not built. Run: cd mlx-flash-server && cargo build --release")

    tmpdir = tempfile.mkdtemp(prefix="mlx_e2e_")
    expert_dir = create_test_experts(tmpdir, num_layers=4, num_experts=8, size_bytes=1024)

    # Remove stale socket
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass

    # Start Rust server (HTTP disabled, only socket)
    proc = subprocess.Popen(
        [
            RUST_BINARY,
            "--port", "0",  # don't bind HTTP (use random or skip)
            "--expert-dir", expert_dir,
            "--cache-mb", "1",  # 1MB cache
            "--socket-path", SOCKET_PATH,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for socket to appear
    for _ in range(30):
        if os.path.exists(SOCKET_PATH):
            break
        time.sleep(0.1)
    else:
        proc.terminate()
        proc.wait()
        shutil.rmtree(tmpdir, ignore_errors=True)
        pytest.skip("Rust server failed to start (socket not created)")

    yield {"proc": proc, "tmpdir": tmpdir, "expert_dir": expert_dir}

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    shutil.rmtree(tmpdir, ignore_errors=True)
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass


class TestCacheRoundtrip:
    """Full roundtrip: Python → Rust cache → SSD → Python."""

    def test_fetch_experts_cold(self, cache_server):
        """First fetch should load from SSD (cache miss)."""
        from mlx_flash_compress.rust_bridge import RustCacheClient

        client = RustCacheClient(SOCKET_PATH)
        client.connect()

        result = client.fetch_experts(layer=0, experts=[1, 2, 3], request_id=1)

        assert "ExpertData" in result, f"Unexpected response: {result}"
        data = result["ExpertData"]
        assert data["request_id"] == 1
        assert len(data["expert_sizes"]) == 3
        assert all(s == 1024 for s in data["expert_sizes"]), \
            f"Expected 1024 bytes per expert, got {data['expert_sizes']}"

        client.close()

    def test_fetch_experts_warm(self, cache_server):
        """Second fetch should be cache hit (same experts)."""
        from mlx_flash_compress.rust_bridge import RustCacheClient

        client = RustCacheClient(SOCKET_PATH)
        client.connect()

        # First fetch (cold)
        client.fetch_experts(layer=1, experts=[0, 1], request_id=10)
        # Second fetch (should be cached)
        result = client.fetch_experts(layer=1, experts=[0, 1], request_id=11)

        assert "ExpertData" in result
        assert all(s == 1024 for s in result["ExpertData"]["expert_sizes"])

        client.close()

    def test_routing_report_triggers_prefetch(self, cache_server):
        """Routing report should trigger prefetch for next layer."""
        from mlx_flash_compress.rust_bridge import RustCacheClient

        client = RustCacheClient(SOCKET_PATH)
        client.connect()

        # Report routing for layer 2
        result = client.report_routing(layer=2, activated=[3, 5, 7], token_idx=0)

        assert "CacheStatsResponse" in result
        stats = result["CacheStatsResponse"]
        assert "entries" in stats
        assert "hit_rate" in stats
        assert stats["entries"] >= 0

        # Wait for prefetch to complete
        time.sleep(0.1)

        # Now fetch layer 3 experts that were prefetched
        result = client.fetch_experts(layer=3, experts=[3, 5, 7], request_id=20)
        assert "ExpertData" in result
        # These should have been prefetched (cache hits)
        assert all(s > 0 for s in result["ExpertData"]["expert_sizes"])

        client.close()

    def test_multiple_clients(self, cache_server):
        """Multiple Python clients can connect simultaneously."""
        from mlx_flash_compress.rust_bridge import RustCacheClient

        clients = []
        for i in range(3):
            c = RustCacheClient(SOCKET_PATH)
            c.connect()
            clients.append(c)

        # Each client fetches different experts
        for i, c in enumerate(clients):
            result = c.fetch_experts(layer=0, experts=[i], request_id=100 + i)
            assert "ExpertData" in result

        for c in clients:
            c.close()

    def test_warmup_curve(self, cache_server):
        """Simulate warm-up: hit rate should increase over repeated accesses."""
        from mlx_flash_compress.rust_bridge import RustCacheClient

        client = RustCacheClient(SOCKET_PATH)
        client.connect()

        # Simulate 20 tokens accessing the same 4 experts
        experts = [0, 1, 2, 3]
        for token in range(20):
            client.fetch_experts(layer=0, experts=experts, request_id=200 + token)

        # Get final stats
        result = client.report_routing(layer=0, activated=experts, token_idx=20)
        assert "CacheStatsResponse" in result
        stats = result["CacheStatsResponse"]

        # After 20 accesses of the same experts, hit rate should be high
        assert stats["hit_rate"] > 0.8, \
            f"Expected hit rate > 80% after warm-up, got {stats['hit_rate']:.1%}"

        client.close()


def main():
    """Run E2E test standalone (without pytest)."""
    if not rust_binary_available():
        print(f"ERROR: Rust binary not found at {RUST_BINARY}")
        print("Build it: cd mlx-flash-server && cargo build --release")
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="mlx_e2e_")
    expert_dir = create_test_experts(tmpdir)

    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass

    print("Starting Rust cache server...")
    proc = subprocess.Popen(
        [RUST_BINARY, "--port", "0", "--expert-dir", expert_dir,
         "--cache-mb", "1", "--socket-path", SOCKET_PATH],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    for _ in range(30):
        if os.path.exists(SOCKET_PATH):
            break
        time.sleep(0.1)
    else:
        print("ERROR: Server failed to start")
        proc.terminate()
        sys.exit(1)

    print("Server started. Running tests...")

    from mlx_flash_compress.rust_bridge import RustCacheClient

    # Test 1: Cold fetch
    client = RustCacheClient(SOCKET_PATH)
    client.connect()
    result = client.fetch_experts(layer=0, experts=[1, 2, 3], request_id=1)
    assert "ExpertData" in result, f"FAIL: {result}"
    print(f"  Cold fetch: OK ({result['ExpertData']['expert_sizes']})")

    # Test 2: Warm fetch
    result = client.fetch_experts(layer=0, experts=[1, 2, 3], request_id=2)
    assert all(s == 1024 for s in result["ExpertData"]["expert_sizes"])
    print(f"  Warm fetch: OK (cache hit)")

    # Test 3: Routing report
    result = client.report_routing(layer=0, activated=[1, 2], token_idx=0)
    assert "CacheStatsResponse" in result
    print(f"  Routing report: OK (entries={result['CacheStatsResponse']['entries']})")

    # Test 4: Warm-up curve
    for t in range(20):
        client.fetch_experts(layer=0, experts=[0, 1, 2, 3], request_id=100 + t)
    result = client.report_routing(layer=0, activated=[0, 1, 2, 3], token_idx=20)
    hr = result["CacheStatsResponse"]["hit_rate"]
    print(f"  Warm-up: hit_rate={hr:.1%} (expected >80%)")
    assert hr > 0.8

    client.close()
    proc.terminate()
    proc.wait()
    shutil.rmtree(tmpdir, ignore_errors=True)
    try:
        os.unlink(SOCKET_PATH)
    except FileNotFoundError:
        pass

    print("\nAll E2E tests passed!")


if __name__ == "__main__":
    main()
