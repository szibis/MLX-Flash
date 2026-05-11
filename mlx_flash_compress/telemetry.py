"""Hardware telemetry for Apple Silicon: GPU, power, thermal, and memory metrics.

Uses IOReport-based sampling via ioreg and vm_stat to collect real-time
hardware metrics without requiring sudo or external dependencies.

Usage:
    from mlx_flash_compress.telemetry import HardwareTelemetry

    tel = HardwareTelemetry()
    tel.start_sampling(interval_ms=1000)
    sample = tel.sample()
    stats = tel.get_stats()
    tel.stop_sampling()
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class TelemetrySample:
    """A single point-in-time hardware telemetry snapshot."""

    timestamp: float = 0.0
    gpu_util_pct: float = 0.0
    gpu_renderer_pct: float = 0.0
    gpu_tiler_pct: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    power_watts: float = 0.0
    cpu_temp_c: float = 0.0
    gpu_temp_c: float = 0.0
    memory_pressure: str = "normal"
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    ane_util_pct: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


# Maximum number of samples kept in the circular buffer (2 minutes at 1s interval)
_MAX_HISTORY = 120


class HardwareTelemetry:
    """Thread-safe hardware telemetry collector for Apple Silicon.

    Samples GPU utilization, power draw, temperatures, and memory metrics
    using macOS system commands (ioreg, vm_stat, sysctl).
    """

    def __init__(self) -> None:
        self._history: deque[TelemetrySample] = deque(maxlen=_MAX_HISTORY)
        self._lock = threading.Lock()
        self._sampling = False
        self._thread: Optional[threading.Thread] = None
        self._total_memory_gb: float = self._detect_total_memory()

    # ── Public API ───────────────────────────────────────────────

    def sample(self) -> TelemetrySample:
        """Capture a single telemetry sample (current metrics)."""
        s = TelemetrySample(timestamp=time.time())

        # GPU metrics from IOAccelerator
        self._read_gpu_metrics(s)

        # Power and thermal from SMC via ioreg
        self._read_power_thermal(s)

        # System memory from vm_stat + sysctl
        self._read_memory(s)

        with self._lock:
            self._history.append(s)

        return s

    def start_sampling(self, interval_ms: int = 1000) -> None:
        """Start continuous background sampling at the given interval."""
        if self._sampling:
            return
        self._sampling = True
        self._thread = threading.Thread(
            target=self._sampling_loop,
            args=(interval_ms / 1000.0,),
            daemon=True,
            name="telemetry-sampler",
        )
        self._thread.start()

    def stop_sampling(self) -> None:
        """Stop the background sampling thread."""
        self._sampling = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_history(self, seconds: int = 120) -> list[dict]:
        """Return recent telemetry samples from the last N seconds."""
        cutoff = time.time() - seconds
        with self._lock:
            return [s.to_dict() for s in self._history if s.timestamp >= cutoff]

    def get_stats(self) -> dict:
        """Return current values plus aggregates over the history window."""
        with self._lock:
            samples = list(self._history)

        if not samples:
            return {
                "current": TelemetrySample(timestamp=time.time()).to_dict(),
                "samples_count": 0,
                "avg": {},
                "max": {},
                "min": {},
            }

        current = samples[-1]
        count = len(samples)

        def _avg(attr: str) -> float:
            vals = [getattr(s, attr) for s in samples]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        def _max(attr: str) -> float:
            vals = [getattr(s, attr) for s in samples]
            return round(max(vals), 2) if vals else 0.0

        def _min(attr: str) -> float:
            vals = [getattr(s, attr) for s in samples]
            return round(min(vals), 2) if vals else 0.0

        numeric_fields = [
            "gpu_util_pct",
            "gpu_renderer_pct",
            "gpu_tiler_pct",
            "gpu_memory_used_gb",
            "power_watts",
            "cpu_temp_c",
            "gpu_temp_c",
            "memory_used_gb",
            "ane_util_pct",
        ]

        return {
            "current": current.to_dict(),
            "samples_count": count,
            "avg": {f: _avg(f) for f in numeric_fields},
            "max": {f: _max(f) for f in numeric_fields},
            "min": {f: _min(f) for f in numeric_fields},
        }

    # ── Internal helpers ─────────────────────────────────────────

    def _sampling_loop(self, interval_s: float) -> None:
        """Background sampling loop."""
        while self._sampling:
            try:
                self.sample()
            except Exception:
                pass  # Never crash the sampling thread
            time.sleep(interval_s)

    def _run_cmd(self, args: list[str], timeout: float = 3.0) -> str:
        """Run a subprocess and return stdout, or empty string on failure."""
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ""

    def _extract_ioreg_int(self, text: str, key: str) -> int:
        """Extract an integer value from ioreg output by key name."""
        for line in text.splitlines():
            stripped = line.strip()
            pos = stripped.find(key)
            if pos >= 0:
                after = stripped[pos + len(key) :]
                digits = ""
                started = False
                for ch in after:
                    if ch.isdigit():
                        digits += ch
                        started = True
                    elif started:
                        break
                if digits:
                    return int(digits)
        return 0

    def _read_gpu_metrics(self, s: TelemetrySample) -> None:
        """Read GPU utilization and memory from IOAccelerator."""
        text = self._run_cmd(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
        if not text:
            return

        s.gpu_util_pct = float(self._extract_ioreg_int(text, "Device Utilization %"))
        s.gpu_renderer_pct = float(self._extract_ioreg_int(text, "Renderer Utilization %"))
        s.gpu_tiler_pct = float(self._extract_ioreg_int(text, "Tiler Utilization %"))

        gpu_mem_used = self._extract_ioreg_int(text, '"In use system memory"=')
        gpu_mem_alloc = self._extract_ioreg_int(text, '"Alloc system memory"=')
        s.gpu_memory_used_gb = round(gpu_mem_used / (1024**3), 2) if gpu_mem_used else 0.0
        s.gpu_memory_total_gb = round(gpu_mem_alloc / (1024**3), 2) if gpu_mem_alloc else 0.0

    def _read_power_thermal(self, s: TelemetrySample) -> None:
        """Read power and temperature from AppleSMC via ioreg."""
        text = self._run_cmd(["ioreg", "-r", "-n", "AppleSmartBattery"])
        if text:
            # Try to extract instantaneous power from battery info
            watts = self._extract_ioreg_int(text, '"InstantAmperage"=')
            voltage = self._extract_ioreg_int(text, '"Voltage"=')
            if watts and voltage:
                # Convert mA * mV -> W (both are in milli units)
                s.power_watts = round(abs(watts) * voltage / 1_000_000, 1)

        # Try to get temperatures from thermal sensors
        thermal_text = self._run_cmd(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AppleARMIODevice"])
        if thermal_text:
            # Look for CPU/GPU temperature entries
            cpu_temp = self._extract_temperature(thermal_text, "cpu")
            gpu_temp = self._extract_temperature(thermal_text, "gpu")
            if cpu_temp > 0:
                s.cpu_temp_c = cpu_temp
            if gpu_temp > 0:
                s.gpu_temp_c = gpu_temp

        # Fallback: try sysctl for CPU temperature (macOS 14+)
        if s.cpu_temp_c == 0.0:
            temp_text = self._run_cmd(["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"])
            if temp_text.strip():
                try:
                    level = int(temp_text.strip())
                    # Thermal level is 0-100 scale, approximate to Celsius
                    s.cpu_temp_c = round(35.0 + level * 0.6, 1)
                except ValueError:
                    pass

    def _extract_temperature(self, text: str, sensor_type: str) -> float:
        """Extract temperature value from ioreg thermal sensor output."""
        # Look for temperature patterns like "temperature" = XX or sensor readings
        pattern = re.compile(
            rf'"(?:.*{sensor_type}.*temp|{sensor_type}.*die).*?".*?=\s*(\d+)',
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            raw = int(match.group(1))
            # Values might be in centi-degrees or direct Celsius
            if raw > 200:
                return round(raw / 100.0, 1)
            elif raw > 0:
                return float(raw)
        return 0.0

    def _read_memory(self, s: TelemetrySample) -> None:
        """Read system memory stats from vm_stat and sysctl."""
        s.memory_total_gb = self._total_memory_gb

        # Try psutil first (most accurate), fall back to vm_stat
        try:
            import psutil

            vm = psutil.virtual_memory()
            s.memory_used_gb = round((vm.total - vm.available) / (1024**3), 2)
            s.memory_total_gb = round(vm.total / (1024**3), 2)
            pct = vm.percent
            if pct > 90:
                s.memory_pressure = "critical"
            elif pct > 75:
                s.memory_pressure = "warn"
            else:
                s.memory_pressure = "normal"
            return
        except ImportError:
            pass

        # Fallback: parse vm_stat
        text = self._run_cmd(["vm_stat"])
        if not text:
            return

        page_size = 16384  # Default for Apple Silicon
        ps_match = re.search(r"page size of (\d+) bytes", text)
        if ps_match:
            page_size = int(ps_match.group(1))

        def _pages(key: str) -> int:
            m = re.search(rf"{key}:\s+(\d+)", text)
            return int(m.group(1)) if m else 0

        free = _pages("Pages free")
        active = _pages("Pages active")
        inactive = _pages("Pages inactive")
        wired = _pages("Pages wired down")
        compressed = _pages("Pages occupied by compressor")

        used_pages = active + wired + compressed
        total_pages = free + active + inactive + wired + compressed
        if total_pages > 0:
            s.memory_used_gb = round(used_pages * page_size / (1024**3), 2)
            used_pct = used_pages / total_pages * 100
            if used_pct > 90:
                s.memory_pressure = "critical"
            elif used_pct > 75:
                s.memory_pressure = "warn"
            else:
                s.memory_pressure = "normal"

        # Memory pressure from macOS memory_pressure command
        pressure_text = self._run_cmd(["memory_pressure"])
        if "critical" in pressure_text.lower():
            s.memory_pressure = "critical"
        elif "warn" in pressure_text.lower():
            s.memory_pressure = "warn"

    def _detect_total_memory(self) -> float:
        """Detect total system memory via sysctl."""
        text = self._run_cmd(["sysctl", "-n", "hw.memsize"])
        if text.strip():
            try:
                return round(int(text.strip()) / (1024**3), 2)
            except ValueError:
                pass
        return 0.0
