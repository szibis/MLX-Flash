"""Adaptive Memory Manager: Use maximum safe RAM without harming user workloads.

Monitors system memory pressure in real-time and dynamically adjusts cache
size to use as much RAM as safely available.

macOS provides memory pressure levels via `vm_stat` and `memory_pressure`:
  - GREEN (normal): plenty of free RAM → use aggressively
  - YELLOW (warning): system starting to compress → reduce cache
  - RED (critical): heavy swapping → shrink cache immediately

The manager:
  1. Detects initial available memory (free + inactive pages)
  2. Reserves a safety margin (default 2GB) for user apps
  3. Allocates the rest to expert cache
  4. Monitors pressure every N seconds
  5. Grows cache when memory frees up (user closes apps)
  6. Shrinks cache when pressure increases (user opens apps)

This ensures the LLM never interferes with the user's work while
extracting maximum performance from available memory.

Usage:
  mgr = MemoryManager(safety_margin_gb=2.0)
  budget = mgr.get_cache_budget()  # returns bytes
  mgr.start_monitoring()  # background thread
  # ... later ...
  new_budget = mgr.get_cache_budget()  # may have changed
"""

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class MemoryState:
    """Current system memory state."""
    total_gb: float = 0.0
    free_gb: float = 0.0
    active_gb: float = 0.0
    inactive_gb: float = 0.0
    wired_gb: float = 0.0
    compressed_gb: float = 0.0
    swap_used_gb: float = 0.0
    pressure_level: str = "unknown"  # green, yellow, red
    app_memory_gb: float = 0.0

    @property
    def available_gb(self) -> float:
        """Memory safely available for cache (free + reclaimable inactive)."""
        return max(self.free_gb + self.inactive_gb * 0.5, 0)

    @property
    def pressure_score(self) -> float:
        """0.0 = no pressure, 1.0 = critical. Used for cache sizing."""
        if self.pressure_level in ("green", "normal"):
            return 0.0
        elif self.pressure_level in ("yellow", "warning"):
            return 0.5
        elif self.pressure_level in ("red", "critical"):
            return 1.0
        # Fallback: estimate from swap usage
        if self.total_gb > 0:
            swap_ratio = self.swap_used_gb / self.total_gb
            return min(swap_ratio * 5, 1.0)  # 20% swap = critical
        return 0.3


def get_memory_state() -> MemoryState:
    """Get current system memory state from macOS vm_stat."""
    state = MemoryState()

    # Total RAM from sysctl
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        state.total_gb = int(result.stdout.strip()) / (1024**3)
    except (subprocess.TimeoutExpired, ValueError):
        pass

    # Detailed stats from vm_stat
    try:
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5,
        )
        page_size = 16384  # default on Apple Silicon
        ps_match = re.search(r"page size of (\d+) bytes", result.stdout)
        if ps_match:
            page_size = int(ps_match.group(1))

        def parse_pages(label):
            m = re.search(rf"{label}:\s+(\d+)", result.stdout)
            return int(m.group(1)) * page_size / (1024**3) if m else 0.0

        state.free_gb = parse_pages("Pages free")
        state.active_gb = parse_pages("Pages active")
        state.inactive_gb = parse_pages("Pages inactive")
        state.wired_gb = parse_pages("Pages wired down")
        state.compressed_gb = parse_pages("Pages occupied by compressor")

        # Swap
        swap_match = re.search(r"Swapins:\s+(\d+)", result.stdout)
        # Use sysctl for swap usage
        try:
            swap_result = subprocess.run(
                ["sysctl", "-n", "vm.swapusage"],
                capture_output=True, text=True, timeout=5,
            )
            swap_used_match = re.search(r"used\s*=\s*([\d.]+)M", swap_result.stdout)
            if swap_used_match:
                state.swap_used_gb = float(swap_used_match.group(1)) / 1024
        except (subprocess.TimeoutExpired, ValueError):
            pass

    except subprocess.TimeoutExpired:
        pass

    # Memory pressure level
    try:
        result = subprocess.run(
            ["memory_pressure", "-Q"],
            capture_output=True, text=True, timeout=5,
        )
        output = result.stdout.lower()
        if "normal" in output or "green" in output:
            state.pressure_level = "normal"
        elif "warn" in output or "yellow" in output:
            state.pressure_level = "warning"
        elif "critical" in output or "red" in output:
            state.pressure_level = "critical"
        else:
            # Parse "free percentage: N%" format
            pct_match = re.search(r"free percentage:\s*(\d+)%", output)
            if pct_match:
                free_pct = int(pct_match.group(1))
                if free_pct >= 40:
                    state.pressure_level = "normal"
                elif free_pct >= 15:
                    state.pressure_level = "warning"
                else:
                    state.pressure_level = "critical"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: estimate from swap usage
    if state.pressure_level == "unknown":
        if state.swap_used_gb > 4.0:
            state.pressure_level = "critical"
        elif state.swap_used_gb > 1.0:
            state.pressure_level = "warning"
        else:
            state.pressure_level = "normal"

    # App memory (active + wired - our process)
    state.app_memory_gb = state.active_gb + state.wired_gb

    return state


class MemoryManager:
    """Adaptive memory manager that maximizes cache without harming user.

    Usage:
        mgr = MemoryManager(safety_margin_gb=2.0)
        cache_bytes = mgr.get_cache_budget()

        # Start background monitoring (adjusts budget dynamically)
        mgr.start_monitoring(interval_s=5.0)

        # Check if budget changed
        if mgr.budget_changed:
            new_budget = mgr.get_cache_budget()
            cache.resize(new_budget)
    """

    def __init__(
        self,
        safety_margin_gb: float = 2.0,
        min_cache_gb: float = 0.5,
        max_cache_pct: float = 0.80,  # never use more than 80% of total
        on_resize: Optional[Callable[[int], None]] = None,
    ):
        self.safety_margin_gb = safety_margin_gb
        self.min_cache_gb = min_cache_gb
        self.max_cache_pct = max_cache_pct
        self.on_resize = on_resize

        self._current_budget_bytes: int = 0
        self._budget_changed: bool = False
        self._monitoring: bool = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Initial calculation
        self._update_budget()

    @property
    def budget_changed(self) -> bool:
        with self._lock:
            changed = self._budget_changed
            self._budget_changed = False
            return changed

    def get_cache_budget(self) -> int:
        """Get current cache budget in bytes."""
        with self._lock:
            return self._current_budget_bytes

    def get_cache_budget_gb(self) -> float:
        return self.get_cache_budget() / (1024**3)

    def _update_budget(self) -> bool:
        """Recalculate cache budget. Returns True if changed significantly."""
        state = get_memory_state()

        # Calculate available memory
        available = state.available_gb - self.safety_margin_gb
        max_allowed = state.total_gb * self.max_cache_pct

        # Adjust based on pressure
        if state.pressure_level == "yellow":
            available *= 0.5  # cut in half under warning
        elif state.pressure_level == "red":
            available = self.min_cache_gb  # minimum under critical

        # Clamp
        budget_gb = max(self.min_cache_gb, min(available, max_allowed))
        budget_bytes = int(budget_gb * 1024**3)

        with self._lock:
            old = self._current_budget_bytes
            self._current_budget_bytes = budget_bytes

            # Consider "changed" if delta > 10%
            if old > 0 and abs(budget_bytes - old) / old > 0.10:
                self._budget_changed = True
                if self.on_resize:
                    self.on_resize(budget_bytes)
                return True

            if old == 0:
                self._current_budget_bytes = budget_bytes
                return True

        return False

    def start_monitoring(self, interval_s: float = 10.0):
        """Start background thread that monitors memory pressure."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_s,),
            daemon=True,
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        self._monitoring = False

    def _monitor_loop(self, interval_s: float):
        while self._monitoring:
            self._update_budget()
            time.sleep(interval_s)

    def get_status(self) -> dict:
        """Get human-readable memory status."""
        state = get_memory_state()
        budget = self.get_cache_budget()
        return {
            "total_ram_gb": f"{state.total_gb:.1f}",
            "free_gb": f"{state.free_gb:.1f}",
            "available_gb": f"{state.available_gb:.1f}",
            "pressure": state.pressure_level,
            "swap_gb": f"{state.swap_used_gb:.1f}",
            "cache_budget_gb": f"{budget / (1024**3):.1f}",
            "safety_margin_gb": f"{self.safety_margin_gb:.1f}",
        }
