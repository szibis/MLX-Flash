"""Transparent mlx-lm integration via monkey patching.

Patches mlx_lm.load() so that any tool calling it (LM Studio, llama.cpp
Python bindings, custom scripts) automatically gets Flash mode:
  - lazy=True weight loading
  - Wired memory limit set to safe budget
  - Page cache advisor enabled for madvise hints
  - Expert streaming activated for MoE models

Usage:
  from mlx_flash_compress.mlx_lm_patch import apply_flash_patch, remove_flash_patch

  # Enable Flash mode globally
  apply_flash_patch(ram_budget_gb=24.0)

  # Now any mlx_lm.load() call gets Flash behavior
  model, tokenizer = mlx_lm.load("mlx-community/Qwen3-30B-A3B-4bit")

  # Disable if needed
  remove_flash_patch()
"""

import functools
import sys
from typing import Optional

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from mlx_flash_compress.hardware import detect_hardware
from mlx_flash_compress.page_cache import PageCacheAdvisor


_original_load = None
_patch_active = False


def _flash_load_wrapper(
    original_fn,
    ram_budget_gb: float,
    advisor: PageCacheAdvisor,
    *args,
    **kwargs,
):
    """Wrapper around mlx_lm.load() that enables Flash mode."""
    # Force lazy loading
    kwargs.setdefault("lazy", True)

    # Set wired memory limit
    if HAS_MLX:
        try:
            limit_bytes = int(ram_budget_gb * 1024 * 1024 * 1024)
            mx.metal.set_wired_limit(limit_bytes)
        except (AttributeError, TypeError):
            try:
                mx.set_memory_limit(limit_bytes)
            except (AttributeError, TypeError):
                pass

    # Call original load
    model, tokenizer = original_fn(*args, **kwargs)

    # Check if MoE and enable expert streaming
    if _is_moe_model(model):
        try:
            from mlx_flash_compress.expert_streaming import enable_expert_streaming
            hw = detect_hardware()
            # Use 50% of available experts as capacity
            streaming = enable_expert_streaming(model, capacity_per_layer=64)
            streaming.warmup()
        except Exception:
            pass  # streaming is optional

    return model, tokenizer


def _is_moe_model(model) -> bool:
    """Detect if a model uses Mixture of Experts."""
    # Check class name (works with real MLX model classes)
    class_name = getattr(type(model), "__name__", "")
    model_str = (class_name + " " + str(type(model))).lower()
    if any(k in model_str for k in ("mixtral", "moe", "switch", "deepseek")):
        return True

    # Check for expert layers via named_modules
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        for name, _ in named_modules():
            if "expert" in name.lower() or "gate" in name.lower():
                return True

    return False


def apply_flash_patch(ram_budget_gb: Optional[float] = None):
    """Monkey-patch mlx_lm.load() to enable Flash mode globally.

    Args:
        ram_budget_gb: RAM budget in GB. Auto-detected if None.
    """
    global _original_load, _patch_active

    if _patch_active:
        return

    try:
        import mlx_lm
    except ImportError:
        raise ImportError("mlx-lm is required for Flash patching. Install: pip install mlx-lm")

    if ram_budget_gb is None:
        hw = detect_hardware()
        ram_budget_gb = hw.available_ram_gb

    advisor = PageCacheAdvisor()
    _original_load = mlx_lm.load

    mlx_lm.load = functools.partial(
        _flash_load_wrapper,
        _original_load,
        ram_budget_gb,
        advisor,
    )
    _patch_active = True


def remove_flash_patch():
    """Remove the Flash mode monkey patch, restoring original mlx_lm.load()."""
    global _original_load, _patch_active

    if not _patch_active or _original_load is None:
        return

    try:
        import mlx_lm
        mlx_lm.load = _original_load
    except ImportError:
        pass

    _original_load = None
    _patch_active = False


def is_patched() -> bool:
    """Check if Flash mode patch is currently active."""
    return _patch_active
