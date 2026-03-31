"""Configuration system for toggling features on/off.

Supports: YAML file, JSON file, CLI flags, environment variables, and auto-detection.

Usage:
  # From Python
  cfg = FlashConfig.auto_detect()
  cfg.enable_lcp_cache = True
  cfg.cache_ram_mb = 4096

  # From CLI
  python -m mlx_flash_compress.bench_final --config config.yaml

  # From env vars
  FLASH_CACHE_RAM_MB=4096 FLASH_ENABLE_PREFETCH=1 python -m ...

  # From file (config.yaml)
  cache:
    enable: true
    ram_mb: 4096
    eviction: lcp        # lcp, lfu, least_stale
    hot_algo: lz4        # lz4, lzfse, none
  prefetch:
    enable: true
    workers: 2
  mixed_precision:
    enable: true
    cold_bits: 2
    hot_bits: 4
    threshold: 0.05      # frequency below this = cold
  skip_fallback:
    enable: false        # disabled by default (quality trade-off)
  ssd_protection:
    enable: true
    thermal_limit_c: 70
    rate_limit: true
  engine:
    backend: auto        # auto, python, c_gcd
  router_hook:
    enable: false        # intercept MLX gate for real routing
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class CacheConfig:
    enable: bool = True
    ram_mb: int = 0          # 0 = auto (use 80% of available)
    eviction: str = "lcp"    # lcp, lfu, least_stale
    hot_algo: str = "lz4"    # lz4, lzfse, none
    lcp_base: float = 0.25
    lcp_decay: int = 128

@dataclass
class PrefetchConfig:
    enable: bool = True
    workers: int = 2

@dataclass
class MixedPrecisionConfig:
    enable: bool = True
    cold_bits: int = 2
    hot_bits: int = 4
    threshold: float = 0.05  # activation frequency below this = cold

@dataclass
class SkipFallbackConfig:
    enable: bool = False     # off by default (quality trade-off)
    renormalize: bool = True

@dataclass
class SSDProtectionConfig:
    enable: bool = True
    thermal_limit_c: float = 70.0
    rate_limit: bool = True
    cooldown_ms: float = 10.0

@dataclass
class EngineConfig:
    backend: str = "auto"    # auto, python, c_gcd

@dataclass
class RouterHookConfig:
    enable: bool = False     # off by default (requires MLX model)

@dataclass
class FlashConfig:
    """Master configuration for MLX-Flash."""
    cache: CacheConfig = field(default_factory=CacheConfig)
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    skip_fallback: SkipFallbackConfig = field(default_factory=SkipFallbackConfig)
    ssd_protection: SSDProtectionConfig = field(default_factory=SSDProtectionConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    router_hook: RouterHookConfig = field(default_factory=RouterHookConfig)

    # Hardware (auto-detected)
    detected_chip: str = ""
    detected_ram_gb: float = 0.0
    detected_ssd_gbs: float = 0.0

    @classmethod
    def auto_detect(cls) -> "FlashConfig":
        """Create config with auto-detected hardware and sensible defaults."""
        cfg = cls()

        try:
            from mlx_flash_compress.hardware import detect_hardware
            hw = detect_hardware()
            cfg.detected_chip = hw.chip
            cfg.detected_ram_gb = hw.total_ram_gb
            cfg.detected_ssd_gbs = hw.estimated_ssd_read_gbs

            # Auto-size cache: 80% of available RAM
            if cfg.cache.ram_mb == 0:
                cfg.cache.ram_mb = int(hw.available_ram_gb * 0.8 * 1024)

            # Auto-select engine backend
            if cfg.engine.backend == "auto":
                try:
                    from mlx_flash_compress.fast_cache_bindings import is_available
                    cfg.engine.backend = "c_gcd" if is_available() else "python"
                except ImportError:
                    cfg.engine.backend = "python"

        except Exception:
            # Fallback if hardware detection fails
            cfg.cache.ram_mb = 2048  # 2GB default
            cfg.engine.backend = "python"

        return cfg

    @classmethod
    def from_file(cls, path: str) -> "FlashConfig":
        """Load config from YAML or JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = p.read_text()

        if p.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                data = yaml.safe_load(text)
            except ImportError:
                raise ImportError("PyYAML required for YAML config: pip install pyyaml")
        else:
            data = json.loads(text)

        return cls._from_dict(data)

    @classmethod
    def from_env(cls) -> "FlashConfig":
        """Load config from environment variables (FLASH_ prefix)."""
        cfg = cls.auto_detect()

        env_map = {
            "FLASH_CACHE_ENABLE": ("cache", "enable", bool),
            "FLASH_CACHE_RAM_MB": ("cache", "ram_mb", int),
            "FLASH_CACHE_EVICTION": ("cache", "eviction", str),
            "FLASH_CACHE_HOT_ALGO": ("cache", "hot_algo", str),
            "FLASH_ENABLE_PREFETCH": ("prefetch", "enable", bool),
            "FLASH_PREFETCH_WORKERS": ("prefetch", "workers", int),
            "FLASH_MIXED_PRECISION": ("mixed_precision", "enable", bool),
            "FLASH_COLD_BITS": ("mixed_precision", "cold_bits", int),
            "FLASH_SKIP_FALLBACK": ("skip_fallback", "enable", bool),
            "FLASH_SSD_PROTECTION": ("ssd_protection", "enable", bool),
            "FLASH_ENGINE": ("engine", "backend", str),
            "FLASH_ROUTER_HOOK": ("router_hook", "enable", bool),
        }

        for env_key, (section, field_name, typ) in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                section_obj = getattr(cfg, section)
                if typ == bool:
                    setattr(section_obj, field_name, val.lower() in ('1', 'true', 'yes'))
                elif typ == int:
                    setattr(section_obj, field_name, int(val))
                else:
                    setattr(section_obj, field_name, val)

        return cfg

    @classmethod
    def _from_dict(cls, data: dict) -> "FlashConfig":
        cfg = cls.auto_detect()
        for section_name in ('cache', 'prefetch', 'mixed_precision', 'skip_fallback',
                            'ssd_protection', 'engine', 'router_hook'):
            if section_name in data:
                section_obj = getattr(cfg, section_name)
                for k, v in data[section_name].items():
                    if hasattr(section_obj, k):
                        setattr(section_obj, k, v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str):
        Path(path).write_text(self.to_json())

    def summary(self) -> str:
        lines = [
            f"MLX-Flash Configuration",
            f"  Hardware: {self.detected_chip}, {self.detected_ram_gb:.0f}GB RAM",
            f"  Cache:    {'ON' if self.cache.enable else 'OFF'} — {self.cache.ram_mb}MB, {self.cache.eviction} eviction",
            f"  Prefetch: {'ON' if self.prefetch.enable else 'OFF'} — {self.prefetch.workers} workers",
            f"  Mixed:    {'ON' if self.mixed_precision.enable else 'OFF'} — {self.mixed_precision.hot_bits}b hot / {self.mixed_precision.cold_bits}b cold",
            f"  Skip:     {'ON' if self.skip_fallback.enable else 'OFF'}",
            f"  SSD prot: {'ON' if self.ssd_protection.enable else 'OFF'} — thermal limit {self.ssd_protection.thermal_limit_c}°C",
            f"  Engine:   {self.engine.backend}",
            f"  Router:   {'HOOKED' if self.router_hook.enable else 'simulated'}",
        ]
        return "\n".join(lines)


# Default config file location
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "mlx-flash-compress" / "config.json"


def get_config(config_path: Optional[str] = None) -> FlashConfig:
    """Get configuration from file, env vars, or auto-detect.

    Priority: explicit path > env vars > default file > auto-detect
    """
    if config_path:
        return FlashConfig.from_file(config_path)

    # Check env vars
    if any(k.startswith("FLASH_") for k in os.environ):
        return FlashConfig.from_env()

    # Check default config file
    if DEFAULT_CONFIG_PATH.exists():
        return FlashConfig.from_file(str(DEFAULT_CONFIG_PATH))

    # Auto-detect
    return FlashConfig.auto_detect()
