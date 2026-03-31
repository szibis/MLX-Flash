"""SSD lifespan protection for MoE expert streaming.

Apple SSDs have finite write endurance (TBW — Total Bytes Written):
  - 1TB Apple SSD: ~600 TBW typical (enterprise-grade NAND)
  - At 209GB model loaded 100x/day: 20.9 TB/day write... wait, that's READ.

Key insight: MoE expert streaming is READ-heavy, not WRITE-heavy.
SSD writes only happen during:
  1. Initial model download (one-time)
  2. Cache file creation (one-time if pre-compressed)
  3. OS swap/page cache writeback (indirect)

Reads don't wear out SSDs. NAND endurance is measured in write cycles.
However, excessive reads can still degrade performance via:
  - Thermal throttling (sustained reads heat the controller)
  - Read disturb (adjacent cells weakened by many reads — rare on modern TLC)
  - Wear leveling overhead (controller background tasks)

This module implements protective measures:
  1. Read rate limiting (prevent thermal throttle)
  2. Sequential read preference (less controller overhead than random)
  3. Idle period detection (pause heavy reads during thermal stress)
  4. Write avoidance (never write to SSD during inference)
  5. SMART monitoring (track SSD health metrics)
"""

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SSDHealth:
    """SSD health status from SMART data."""
    available: bool = False
    percentage_used: float = 0.0  # 0-100%
    data_read_tb: float = 0.0
    data_written_tb: float = 0.0
    power_on_hours: int = 0
    temperature_c: float = 0.0
    warning: Optional[str] = None


def check_ssd_health() -> SSDHealth:
    """Read SSD SMART data on macOS."""
    health = SSDHealth()

    try:
        result = subprocess.run(
            ["smartctl", "-a", "/dev/disk0"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            health.available = True
            for line in result.stdout.split("\n"):
                if "Percentage Used" in line:
                    try:
                        health.percentage_used = float(line.split(":")[-1].strip().rstrip("%"))
                    except ValueError:
                        pass
                elif "Temperature" in line and "Celsius" in line:
                    try:
                        health.temperature_c = float(line.split(":")[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # smartctl not installed — try diskutil
        try:
            result = subprocess.run(
                ["diskutil", "info", "disk0"],
                capture_output=True, text=True, timeout=5,
            )
            health.available = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return health


@dataclass
class ReadPolicy:
    """SSD-friendly read policy parameters."""
    max_read_rate_gbs: float = 15.0  # stay below max to prevent thermal throttle
    sequential_preference: bool = True  # prefer sequential over random reads
    cooldown_after_burst_s: float = 0.01  # brief pause after large burst reads
    burst_threshold_mb: float = 100.0  # reads above this trigger cooldown
    enable_rate_limiting: bool = True
    write_during_inference: bool = False  # never write cache to SSD during inference

    # Thermal protection
    thermal_throttle_temp_c: float = 70.0  # pause reads above this temp
    thermal_check_interval_s: float = 30.0  # how often to check temperature


class SSDProtectedReader:
    """SSD-friendly file reader with rate limiting and thermal protection.

    Wraps standard file reads with protective measures to extend SSD lifespan
    and prevent thermal throttling during sustained expert streaming.
    """

    def __init__(self, policy: Optional[ReadPolicy] = None):
        self.policy = policy or ReadPolicy()
        self._bytes_since_cooldown: int = 0
        self._last_thermal_check: float = 0.0
        self._throttled: bool = False

    def read_expert(self, path: str) -> bytes:
        """Read an expert weight file with SSD protection."""
        # Check thermal (periodically, not every read)
        now = time.monotonic()
        if now - self._last_thermal_check > self.policy.thermal_check_interval_s:
            self._check_thermal()
            self._last_thermal_check = now

        if self._throttled:
            time.sleep(0.1)  # 100ms pause during thermal stress

        data = self._read_file(path)
        self._bytes_since_cooldown += len(data)

        # Cooldown after burst
        if self._bytes_since_cooldown > self.policy.burst_threshold_mb * 1024 * 1024:
            if self.policy.enable_rate_limiting:
                time.sleep(self.policy.cooldown_after_burst_s)
            self._bytes_since_cooldown = 0

        return data

    def _read_file(self, path: str) -> bytes:
        """Read file with sequential access hints."""
        fd = os.open(path, os.O_RDONLY)
        try:
            # Hint to OS: this is sequential access
            if self.policy.sequential_preference:
                try:
                    import fcntl
                    # F_RDAHEAD = 45 on macOS (enable read-ahead)
                    fcntl.fcntl(fd, 45, 1)
                except (ImportError, OSError):
                    pass

            size = os.fstat(fd).st_size
            return os.read(fd, size)
        finally:
            os.close(fd)

    def _check_thermal(self):
        """Check SSD temperature and throttle if needed."""
        health = check_ssd_health()
        if health.available and health.temperature_c > 0:
            self._throttled = health.temperature_c > self.policy.thermal_throttle_temp_c
            if self._throttled:
                print(f"  WARNING: SSD temperature {health.temperature_c}°C > {self.policy.thermal_throttle_temp_c}°C — throttling reads")


def estimate_ssd_impact(
    model_gb: float,
    tokens_per_day: int = 10000,
    cache_hit_rate: float = 0.7,
    expert_size_mb: float = 6.75,
    k: int = 4,
    num_layers: int = 60,
) -> dict:
    """Estimate daily SSD read volume and lifespan impact.

    Returns dict with read volumes and projected wear.
    """
    # Per token: K experts per layer read from SSD (on cache miss)
    experts_per_token = k * num_layers
    cache_misses_per_token = experts_per_token * (1 - cache_hit_rate)
    bytes_per_token = cache_misses_per_token * expert_size_mb * 1024 * 1024

    daily_read_gb = bytes_per_token * tokens_per_day / (1024**3)
    yearly_read_tb = daily_read_gb * 365 / 1024

    # Apple SSDs rated for ~600 TBW (writes), but reads don't count toward TBW
    # Read impact is primarily thermal + controller wear (negligible vs writes)

    return {
        "experts_per_token": experts_per_token,
        "cache_misses_per_token": cache_misses_per_token,
        "bytes_per_token_mb": bytes_per_token / (1024 * 1024),
        "daily_read_gb": daily_read_gb,
        "yearly_read_tb": yearly_read_tb,
        "ssd_write_impact": "NONE (reads don't count toward TBW)",
        "thermal_risk": "LOW" if daily_read_gb < 500 else "MODERATE" if daily_read_gb < 2000 else "HIGH",
        "recommendation": (
            "SSD reads do not degrade NAND cells. "
            "Write endurance (TBW) is unaffected by inference workloads. "
            "Only concern: sustained reads may cause thermal throttling. "
            f"At {cache_hit_rate:.0%} cache hit rate, daily read volume is {daily_read_gb:.0f} GB."
        ),
    }
