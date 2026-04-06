"""Metal kernel compilation and loading infrastructure.

Compiles .metal shader files into Metal libraries at first use,
caches the compiled .metallib for subsequent runs.

Usage:
  loader = get_kernel_loader()
  if loader.available:
      loader.compile_all()
      fn = loader.get_function("flash_dequant_gemv_q4")
"""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


KERNEL_DIR = Path(__file__).parent
CACHE_DIR = Path.home() / ".cache" / "mlx-flash" / "kernels"


@dataclass
class MetalKernelLoader:
    """Compile and load custom Metal kernels."""

    compiled: dict = field(default_factory=dict)
    _available: Optional[bool] = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = (
                sys.platform == "darwin"
                and _has_metal_compiler()
            )
        return self._available

    def compile_shader(self, name: str) -> Optional[Path]:
        """Compile a .metal file to .metallib."""
        if not self.available:
            return None

        source = KERNEL_DIR / f"{name}.metal"
        if not source.exists():
            return None

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        lib_path = CACHE_DIR / f"{name}.metallib"

        # Skip if already compiled and source hasn't changed
        if lib_path.exists() and lib_path.stat().st_mtime > source.stat().st_mtime:
            self.compiled[name] = lib_path
            return lib_path

        try:
            # Compile .metal → .air (intermediate)
            air_path = CACHE_DIR / f"{name}.air"
            subprocess.run([
                "xcrun", "-sdk", "macosx", "metal",
                "-c", str(source),
                "-o", str(air_path),
            ], check=True, capture_output=True)

            # Link .air → .metallib
            subprocess.run([
                "xcrun", "-sdk", "macosx", "metallib",
                str(air_path),
                "-o", str(lib_path),
            ], check=True, capture_output=True)

            # Clean up intermediate
            air_path.unlink(missing_ok=True)

            self.compiled[name] = lib_path
            return lib_path

        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def compile_all(self) -> dict:
        """Compile all .metal files in the kernels directory."""
        results = {}
        for metal_file in KERNEL_DIR.glob("*.metal"):
            name = metal_file.stem
            path = self.compile_shader(name)
            results[name] = path
        return results

    def get_compiled_path(self, name: str) -> Optional[Path]:
        """Get path to compiled metallib, compiling if needed."""
        if name not in self.compiled:
            self.compile_shader(name)
        return self.compiled.get(name)


def _has_metal_compiler() -> bool:
    """Check if the Metal compiler toolchain is available."""
    try:
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_loader: Optional[MetalKernelLoader] = None


def get_kernel_loader() -> MetalKernelLoader:
    """Get the singleton kernel loader."""
    global _loader
    if _loader is None:
        _loader = MetalKernelLoader()
    return _loader
