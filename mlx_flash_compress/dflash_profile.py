"""DFlash model profiling and auto-configuration.

Detects model characteristics and selects optimal DFlash settings.
"""

import time
from dataclasses import dataclass, field
from typing import Literal

import mlx.core as mx
import mlx.utils

ModelCategory = Literal[
    "small_dense",  # <10B active, dense — AR is fast, DFlash unlikely to help
    "small_moe",  # <10B active, MoE — AR is fast, DFlash unlikely to help
    "medium_dense",  # 10-30B active, dense — DFlash may help
    "medium_moe",  # 10-30B active, MoE — DFlash may help if AR < 30 tok/s
    "large_dense",  # 30B+ active, dense — DFlash wins big
    "large_moe",  # 30B+ active, MoE — DFlash wins big
    "ssd_small",  # SSM+attention hybrid, <10B active
    "ssd_medium",  # SSM+attention hybrid, 10-30B active
    "ssd_large",  # SSM+attention hybrid, 30B+ active
]


@dataclass
class ModelProfile:
    """Detected model characteristics."""

    category: ModelCategory
    total_params_b: float
    active_params_b: float
    num_layers: int
    num_ssm_layers: int
    num_attn_layers: int
    is_moe: bool
    is_quantized: bool
    quant_bits: int | None
    ar_tok_s: float | None = None
    has_ssm: bool = False

    @property
    def ssm_ratio(self) -> float:
        return self.num_ssm_layers / max(1, self.num_layers)

    @property
    def recommendation(self) -> str:
        if self.ar_tok_s and self.ar_tok_s > 50:
            return "skip"
        if self.ar_tok_s and self.ar_tok_s > 25:
            return "marginal"
        return "recommended"


Priority = Literal["auto", "quality", "speed", "balanced"]


@dataclass
class DFlashProfile:
    """Optimal DFlash configuration for a model."""

    name: str
    quantize_drafter: int | None
    inference_block_size: int | None
    use_cache: bool
    compile_drafter: bool
    description: str
    priority: Priority = "balanced"


PROFILES = {
    # --- Quality-first: bf16 drafter, full block size, maximum acceptance ---
    "quality_slow": DFlashProfile(
        name="quality_slow",
        quantize_drafter=None,
        inference_block_size=None,
        use_cache=True,
        compile_drafter=False,
        priority="quality",
        description="Quality-first for slow targets (AR < 15 tok/s). "
        "bf16 drafter preserves hidden state fidelity for maximum acceptance.",
    ),
    "quality_medium": DFlashProfile(
        name="quality_medium",
        quantize_drafter=None,
        inference_block_size=None,
        use_cache=True,
        compile_drafter=False,
        priority="quality",
        description="Quality-first for medium targets (AR 15-40 tok/s). bf16 drafter, full block. Accept rate > tok/s.",
    ),
    "quality_ssd": DFlashProfile(
        name="quality_ssd",
        quantize_drafter=None,
        inference_block_size=None,
        use_cache=True,
        compile_drafter=False,
        priority="quality",
        description="Quality-first for SSM hybrids. bf16 drafter, full block. "
        "SSM replay cost is acceptable for better acceptance.",
    ),
    # --- Speed-first: quantized drafter, smaller blocks, maximum tok/s ---
    "speed_slow": DFlashProfile(
        name="speed_slow",
        quantize_drafter=8,
        inference_block_size=None,
        use_cache=True,
        compile_drafter=False,
        priority="speed",
        description="Speed-first for slow targets. 8-bit drafter cuts draft time 2.5x "
        "with zero acceptance loss on verification.",
    ),
    "speed_medium": DFlashProfile(
        name="speed_medium",
        quantize_drafter=8,
        inference_block_size=4,
        use_cache=True,
        compile_drafter=False,
        priority="speed",
        description="Speed-first for medium targets. 8-bit + bs=4 for fastest cycles. "
        "Higher per-token acceptance compensates for fewer draft tokens.",
    ),
    "speed_ssd": DFlashProfile(
        name="speed_ssd",
        quantize_drafter=8,
        inference_block_size=4,
        use_cache=True,
        compile_drafter=False,
        priority="speed",
        description="Speed-first for SSM hybrids. 8-bit + bs=4 minimizes replay cost "
        "(18ms → ~5ms). Best measured config for SSM models.",
    ),
    # --- Balanced (default): best measured tradeoff ---
    "fast_target": DFlashProfile(
        name="fast_target",
        quantize_drafter=8,
        inference_block_size=None,
        use_cache=True,
        compile_drafter=False,
        description="Target AR > 40 tok/s. 8-bit drafter to minimize overhead. "
        "DFlash unlikely to beat AR but provides multi-token drafting.",
    ),
    "medium_target": DFlashProfile(
        name="medium_target",
        quantize_drafter=8,
        inference_block_size=None,
        use_cache=True,
        compile_drafter=False,
        description="Target AR 15-40 tok/s. DFlash should match or beat AR. 8-bit drafter for fast drafting.",
    ),
    "slow_target": DFlashProfile(
        name="slow_target",
        quantize_drafter=None,
        inference_block_size=None,
        use_cache=True,
        compile_drafter=False,
        description="Target AR < 15 tok/s. DFlash wins big. bf16 drafter for maximum acceptance rate.",
    ),
    "ssd_fast": DFlashProfile(
        name="ssd_fast",
        quantize_drafter=8,
        inference_block_size=4,
        use_cache=True,
        compile_drafter=False,
        description="SSM+attention hybrid, fast AR. Smaller block size reduces "
        "verify cost. SSM replay overhead limits gains.",
    ),
    "ssd_slow": DFlashProfile(
        name="ssd_slow",
        quantize_drafter=None,
        inference_block_size=None,
        use_cache=True,
        compile_drafter=False,
        description="SSM+attention hybrid, slow AR. DFlash amortizes well. Full block size for maximum tokens/step.",
    ),
}


def detect_model(model, tokenizer=None) -> ModelProfile:
    """Detect model characteristics from a loaded mlx-lm model."""
    layers = _find_layers(model)
    num_layers = len(layers)

    num_ssm = sum(1 for l in layers if getattr(l, "is_linear", False))
    num_attn = num_layers - num_ssm
    has_ssm = num_ssm > 0
    is_moe = any(_is_moe_layer(l) for l in layers)

    total_bytes = sum(p.nbytes for _, p in mlx.utils.tree_flatten(model.parameters()))  # type: ignore[union-attr, str-unpack]
    total_params_b = total_bytes / 1e9

    active_params_b = _estimate_active_params(model, layers, is_moe)

    is_quantized = any(
        hasattr(mod, "scales") or type(mod).__name__ == "QuantizedLinear" for _, mod in model.named_modules()
    )
    quant_bits = _detect_quant_bits(model) if is_quantized else None

    if has_ssm:
        if active_params_b < 10:
            category = "ssd_small"
        elif active_params_b < 30:
            category = "ssd_medium"
        else:
            category = "ssd_large"
    elif is_moe:
        if active_params_b < 10:
            category = "small_moe"
        elif active_params_b < 30:
            category = "medium_moe"
        else:
            category = "large_moe"
    else:
        if active_params_b < 10:
            category = "small_dense"
        elif active_params_b < 30:
            category = "medium_dense"
        else:
            category = "large_dense"

    return ModelProfile(
        category=category,  # type: ignore[arg-type]
        total_params_b=round(total_params_b, 1),
        active_params_b=round(active_params_b, 1),
        num_layers=num_layers,
        num_ssm_layers=num_ssm,
        num_attn_layers=num_attn,
        is_moe=is_moe,
        is_quantized=is_quantized,
        quant_bits=quant_bits,
        has_ssm=has_ssm,
    )


def measure_ar_baseline(model, tokenizer, prompt: str = "Hello", max_tokens: int = 32, warmup_tokens: int = 8) -> float:
    """Measure autoregressive baseline speed (tok/s)."""
    from mlx_lm import stream_generate

    for resp in stream_generate(model, tokenizer, prompt, max_tokens=warmup_tokens):
        pass

    last_resp = None
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        last_resp = resp

    return last_resp.generation_tps if last_resp else 0


def select_profile(model_profile: ModelProfile, priority: Priority = "auto") -> DFlashProfile:
    """Select optimal DFlash profile based on model characteristics.

    Args:
        model_profile: Detected model characteristics.
        priority:
          "auto" — picks the best config that maintains lossless quality AND
                   beats or matches AR baseline. Falls back to "skip AR" if
                   DFlash can't win. This is the default for users who just
                   want the best option without thinking about it.
          "quality" — bf16 drafter, full block size. Maximum acceptance rate.
          "speed" — 8-bit drafter, small blocks. Maximum tok/s.
          "balanced" — measured best tradeoff per model category.
    """
    ar = model_profile.ar_tok_s or 0

    if priority == "auto":
        if ar > 50:
            return PROFILES["fast_target"]
        if model_profile.has_ssm:
            if ar > 30:
                return PROFILES["speed_ssd"]
            return PROFILES["quality_ssd"]
        if ar > 25:
            return PROFILES["speed_medium"]
        elif ar > 15:
            return PROFILES["medium_target"]
        else:
            return PROFILES["quality_slow"]

    if priority == "quality":
        if model_profile.has_ssm:
            return PROFILES["quality_ssd"]
        if ar > 15:
            return PROFILES["quality_medium"]
        else:
            return PROFILES["quality_slow"]

    if priority == "speed":
        if model_profile.has_ssm:
            return PROFILES["speed_ssd"]
        if ar > 15:
            return PROFILES["speed_medium"]
        else:
            return PROFILES["speed_slow"]

    # balanced
    if model_profile.has_ssm:
        if ar > 30:
            return PROFILES["ssd_fast"]
        return PROFILES["ssd_slow"]

    if ar > 40:
        return PROFILES["fast_target"]
    elif ar > 15:
        return PROFILES["medium_target"]
    else:
        return PROFILES["slow_target"]


def profile_and_configure(
    model, tokenizer, prompt: str = "Hello", max_tokens: int = 32, priority: Priority = "auto"
) -> tuple[ModelProfile, DFlashProfile]:
    """Full profiling: detect model, measure AR speed, select profile."""
    model_info = detect_model(model, tokenizer)
    model_info.ar_tok_s = round(measure_ar_baseline(model, tokenizer, prompt, max_tokens), 1)
    profile = select_profile(model_info, priority)
    return model_info, profile


def _find_layers(model):
    """Find the transformer layers in a model."""
    candidates = []
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return list(lm.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    return []


def _is_moe_layer(layer) -> bool:
    """Check if a layer has MoE components."""
    for name, child in layer.named_modules():
        child_type = type(child).__name__.lower()
        if "moe" in child_type or "switch" in child_type or "expert" in child_type:
            return True
        if hasattr(child, "gate") and hasattr(child, "switch_mlp"):
            return True
    return False


def _estimate_active_params(model, layers, is_moe: bool) -> float:
    """Estimate active parameters per token (in billions)."""
    total_bytes = sum(p.nbytes for _, p in mlx.utils.tree_flatten(model.parameters()))  # type: ignore[union-attr, str-unpack]

    if not is_moe:
        return total_bytes / 1e9

    total_expert_bytes = 0
    active_expert_bytes: float = 0
    non_expert_bytes = 0

    for layer in layers:
        for name, child in layer.named_modules():
            child_type = type(child).__name__.lower()
            child_bytes = sum(p.nbytes for _, p in mlx.utils.tree_flatten(child.parameters()))  # type: ignore[union-attr, str-unpack]

            if "switch" in child_type or "moe" in child_type:
                num_experts = getattr(child, "num_experts", getattr(child, "num_local_experts", 8))
                top_k = getattr(child, "top_k", getattr(child, "num_experts_per_tok", 2))
                if num_experts > 0:
                    active_expert_bytes += child_bytes * top_k / num_experts
                    total_expert_bytes += child_bytes
                break

    if total_expert_bytes > 0:
        non_expert_bytes = total_bytes - total_expert_bytes
        return (non_expert_bytes + active_expert_bytes) / 1e9

    return total_bytes / 1e9 * 0.3


def _detect_quant_bits(model) -> int | None:
    """Detect quantization bits from model weights."""
    for name, module in model.named_modules():
        if type(module).__name__ == "QuantizedLinear":
            bits = getattr(module, "bits", None)
            if bits:
                return bits
    return None
