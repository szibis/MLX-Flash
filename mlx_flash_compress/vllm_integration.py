"""vllm-mlx integration: drop-in expert caching for vllm-mlx's MoE layers.

vllm-mlx delegates all inference to mlx-lm, which uses `SwitchGLU` from
`mlx_lm/models/switch_layers.py` for MoE forward passes. Each `SwitchGLU`
has three `SwitchLinear` sub-modules (gate_proj, up_proj, down_proj) that
call `mx.gather_mm` / `mx.gather_qmm` with stacked expert weights.

This module provides a single function `enable_caching(model)` that swaps
those `SwitchLinear` instances with `CachedSwitchLinear` from our
expert_streaming module, enabling:
  - LCP eviction with layer-depth bias
  - Pre-stacked GPU tensors with lookup table
  - Skip-fallback with adaptive top-k
  - mx.clear_cache() after eviction
  - Profile-based warmup

Integration point in vllm-mlx:
  In `vllm_mlx/model_runner.py` → `MLXModelRunner.load_model()`:
  After `self.model, self.tokenizer = load(...)`, inject:
    from mlx_flash_compress.vllm_integration import enable_caching
    state = enable_caching(self.model, capacity_per_layer=200)

Or standalone with mlx-lm:
  from mlx_lm import load
  from mlx_flash_compress.vllm_integration import enable_caching
  model, tokenizer = load("mlx-community/Qwen3-30B-A3B-4bit")
  state = enable_caching(model)
"""

from pathlib import Path
from typing import Optional

from mlx_flash_compress.expert_streaming import (
    enable_expert_streaming,
    enable_skip_fallback,
    get_warmup_experts,
    StreamingState,
)


def enable_caching(
    model,
    capacity_per_layer: int = 200,
    model_path: Optional[str] = None,
    adaptive_skip_threshold: float = 0.0,
    warmup_task: Optional[str] = None,
) -> StreamingState:
    """Enable expert caching on any mlx-lm loaded model.

    This is the primary integration point for vllm-mlx and mlx-lm.
    It replaces SwitchLinear instances with CachedSwitchLinear for
    GPU-resident expert caching with LCP eviction.

    Args:
        model: A loaded mlx-lm model (from `mlx_lm.load()`).
        capacity_per_layer: Max experts to keep in GPU per layer.
            Set to total_experts for no streaming overhead.
            Set to 50% for memory-constrained systems.
        model_path: Path to model directory (for safetensors mmap).
            Auto-detected from HuggingFace cache if None.
        adaptive_skip_threshold: Skip secondary experts when top-1
            score is this many times larger (0 = disabled, 3.0 = aggressive).
        warmup_task: Task name for profile-based warmup
            ("coding", "writing", "math", "chat", "general", or None).

    Returns:
        StreamingState with cache references and control methods.

    Example:
        from mlx_lm import load
        from mlx_flash_compress.vllm_integration import enable_caching

        model, tokenizer = load("mlx-community/Qwen3-30B-A3B-4bit")
        state = enable_caching(model, capacity_per_layer=64)
        state.warmup()

        # Generate normally — caching is transparent
        from mlx_lm import generate
        output = generate(model, tokenizer, prompt="Hello", max_tokens=100)
        state.update()  # call between generation steps
    """
    # Enable expert streaming (replaces SwitchLinear with CachedSwitchLinear)
    state = enable_expert_streaming(
        model,
        capacity_per_layer=capacity_per_layer,
        model_path=model_path,
    )

    # Enable skip-fallback if threshold is set
    if adaptive_skip_threshold > 0 and state.caches:
        enable_skip_fallback(
            model, state.caches,
            adaptive_skip_threshold=adaptive_skip_threshold,
        )

    # Profile-based warmup
    if warmup_task and state.caches:
        cache = state.caches[0]
        expert_lists = get_warmup_experts(
            task=warmup_task,
            num_layers=len(state.caches),
            num_experts=cache.num_experts,
            top_n=cache.capacity,
        )
        for cache_obj, experts in zip(state.caches, expert_lists):
            cache_obj.initial_fill(experts)
    else:
        state.warmup()

    return state


def get_model_info(model) -> dict:
    """Extract MoE architecture info from a loaded model.

    Returns dict with expert count, layer count, and whether it's MoE.
    """
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is None:
        return {"is_moe": False, "num_layers": 0}

    num_moe_layers = 0
    num_experts = 0

    for layer in layers:
        if not hasattr(layer, "mlp"):
            continue
        mlp = layer.mlp
        switch = getattr(mlp, "switch_mlp", None)
        if switch is None:
            bsm = getattr(mlp, "block_sparse_moe", None)
            if bsm is not None:
                switch = getattr(bsm, "switch_mlp", None)
        if switch is None:
            continue

        num_moe_layers += 1
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(switch, proj_name, None)
            if proj is not None:
                w = getattr(proj, "weight", None)
                if w is not None and len(w.shape) == 3:
                    num_experts = max(num_experts, w.shape[0])
                    break

    return {
        "is_moe": num_moe_layers > 0,
        "num_layers": len(layers),
        "num_moe_layers": num_moe_layers,
        "num_experts": num_experts,
        "total_layers": len(layers),
    }


def auto_configure(model, ram_gb: float = None) -> dict:
    """Auto-configure caching parameters based on model and available RAM.

    Returns recommended kwargs for enable_caching().
    """
    info = get_model_info(model)

    if not info["is_moe"]:
        return {"capacity_per_layer": 0, "note": "Dense model — no expert caching needed"}

    if ram_gb is None:
        try:
            from mlx_flash_compress.memory_manager import get_memory_state
            ram_gb = get_memory_state().total_gb
        except Exception:
            ram_gb = 36  # default

    num_experts = info["num_experts"]
    num_layers = info["num_moe_layers"]

    # Estimate expert memory: ~50MB per expert per layer (rough)
    expert_memory_gb = num_experts * num_layers * 50 / 1024
    available_for_cache = ram_gb * 0.6  # 60% of RAM for expert cache

    if expert_memory_gb <= available_for_cache:
        capacity = num_experts  # all fit
        note = "All experts fit in RAM — full speed, no streaming"
    else:
        # How many experts fit?
        experts_that_fit = int(available_for_cache / (num_layers * 50 / 1024))
        capacity = max(min(experts_that_fit, num_experts), 4)
        note = f"Streaming mode: {capacity}/{num_experts} experts cached per layer"

    return {
        "capacity_per_layer": capacity,
        "adaptive_skip_threshold": 3.0 if capacity < num_experts else 0.0,
        "warmup_task": "general",
        "note": note,
        **info,
    }
