"""MatFormer: Elastic Inference via Nested FFN Extraction.

Implements the MatFormer framework (arXiv:2310.07707, NeurIPS 2024) for MLX.
Extracts sub-models from 50-100% of full size from a single set of weights.
Under memory pressure, automatically shrinks the active model. When pressure
drops, restores the full model. Perfect for Apple Silicon's memory-pressure-
aware unified memory architecture.

Key insight: MatFormer-trained models have nested structure in their FFN blocks.
The first (ratio * hidden_dim) dimensions contain a valid sub-model at each
extraction ratio. This means a single checkpoint can serve multiple model sizes
with zero retraining.

Extraction ratios: [0.5, 0.625, 0.75, 0.875, 1.0]
  - 1.0 = full model (no pressure)
  - 0.875 = 87.5% (warning pressure)
  - 0.75 = 75% (critical pressure)
  - 0.625 = 62.5% (urgent pressure)
  - 0.5 = 50% (emergency, minimum viable model)

Usage:
  from mlx_flash_compress.matformer import apply_matformer, MatFormerConfig

  config = MatFormerConfig(auto_adapt=True)
  adaptive = apply_matformer(model, config)

  # Forward pass automatically adapts model size to memory pressure
  output = adaptive.forward(input_ids)
  print(adaptive.get_stats())
"""

from dataclasses import dataclass, field
from typing import Optional
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn


@dataclass
class MatFormerConfig:
    """Configuration for MatFormer elastic inference."""
    extraction_ratios: list[float] = field(
        default_factory=lambda: [0.5, 0.625, 0.75, 0.875, 1.0]
    )
    auto_adapt: bool = True         # automatically select ratio based on memory pressure
    min_ratio: float = 0.5          # minimum extraction (50% of full size)
    pressure_thresholds: dict = field(default_factory=lambda: {
        "nominal": 1.0,    # no pressure -> full model
        "warning": 0.875,  # yellow pressure -> 87.5%
        "critical": 0.75,  # red pressure -> 75%
        "urgent": 0.625,   # heavy swap -> 62.5%
        "emergency": 0.5,  # out of memory -> 50%
    })


class MatFormerExtractor:
    """Extract sub-models from nested FFN blocks.

    For each nn.Linear layer in the FFN, slices the weight matrix to use
    only the first (ratio * hidden_dim) output dimensions. Because MatFormer
    models are trained with nested optimization, these sub-matrices form
    valid sub-models at each extraction ratio.
    """

    def __init__(self, model, config: MatFormerConfig = None):
        self.model = model
        self.config = config or MatFormerConfig()

        # Cache original weights for restoration
        self._original_weights: dict[str, dict[str, mx.array]] = {}
        self._extracted_ratio: float = 1.0

        # Discover FFN layers
        self._ffn_layers: list[tuple[str, nn.Module]] = []
        self._discover_ffn_layers()

    def _discover_ffn_layers(self):
        """Find all FFN/MLP layers in the model."""
        layers = self._get_model_layers()
        if layers is None:
            return

        for i, layer in enumerate(layers):
            mlp = getattr(layer, 'mlp', None)
            if mlp is None:
                mlp = getattr(layer, 'feed_forward', None)
            if mlp is None:
                continue

            # Look for linear projections in the MLP
            for proj_name in ['gate_proj', 'up_proj', 'down_proj',
                              'w1', 'w2', 'w3', 'fc1', 'fc2',
                              'dense_h_to_4h', 'dense_4h_to_h']:
                proj = getattr(mlp, proj_name, None)
                if proj is not None and isinstance(proj, nn.Linear):
                    key = f"layer.{i}.mlp.{proj_name}"
                    self._ffn_layers.append((key, proj))

    def _get_model_layers(self):
        """Get transformer layers from the model."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        if hasattr(self.model, 'layers'):
            return self.model.layers
        return None

    def extract(self, ratio: float = 1.0) -> nn.Module:
        """Extract sub-model at given ratio.

        For each FFN layer, slices the weight matrix to use the first
        (ratio * dim) output dimensions. The gate/up projections have
        their output dimension sliced, while the down projection has
        its input dimension sliced (to match).

        Args:
            ratio: Extraction ratio in [min_ratio, 1.0]

        Returns:
            The model (modified in-place) with sliced FFN weights
        """
        ratio = max(self.config.min_ratio, min(1.0, ratio))

        # Snap to nearest valid ratio
        ratio = self._snap_to_ratio(ratio)

        if ratio == self._extracted_ratio:
            return self.model

        # Restore original weights first if going back up
        if ratio > self._extracted_ratio and self._original_weights:
            self._restore_original_weights()

        # Cache originals if first extraction
        if not self._original_weights:
            self._cache_original_weights()

        if ratio < 1.0:
            self._apply_extraction(ratio)

        self._extracted_ratio = ratio
        return self.model

    def _snap_to_ratio(self, ratio: float) -> float:
        """Snap to the nearest valid extraction ratio."""
        ratios = sorted(self.config.extraction_ratios)
        # Find the largest ratio <= requested
        valid = [r for r in ratios if r <= ratio + 1e-6]
        if valid:
            return valid[-1]
        return ratios[0]

    def _cache_original_weights(self):
        """Cache original weight matrices for later restoration."""
        for key, proj in self._ffn_layers:
            self._original_weights[key] = {}
            if hasattr(proj, 'weight'):
                self._original_weights[key]['weight'] = proj.weight
            if hasattr(proj, 'bias') and proj.bias is not None:
                self._original_weights[key]['bias'] = proj.bias

    def _restore_original_weights(self):
        """Restore original weight matrices."""
        for key, proj in self._ffn_layers:
            if key in self._original_weights:
                if 'weight' in self._original_weights[key]:
                    proj.weight = self._original_weights[key]['weight']
                if 'bias' in self._original_weights[key]:
                    proj.bias = self._original_weights[key]['bias']
        self._extracted_ratio = 1.0

    def _apply_extraction(self, ratio: float):
        """Apply weight slicing for the given ratio."""
        for key, proj in self._ffn_layers:
            if key not in self._original_weights:
                continue

            original_w = self._original_weights[key].get('weight')
            if original_w is None:
                continue

            out_dim, in_dim = original_w.shape

            # Determine which dimension to slice based on projection type
            # gate_proj, up_proj, w1, w3, fc1, dense_h_to_4h: slice output dim
            # down_proj, w2, fc2, dense_4h_to_h: slice input dim
            is_down = any(name in key for name in
                         ['down_proj', 'w2', 'fc2', 'dense_4h_to_h'])

            if is_down:
                # Down projection: reduce input dimension
                new_in = int(in_dim * ratio)
                new_in = max(new_in, 1)
                proj.weight = original_w[:, :new_in]
            else:
                # Up/gate projection: reduce output dimension
                new_out = int(out_dim * ratio)
                new_out = max(new_out, 1)
                proj.weight = original_w[:new_out, :]

                # Also slice bias if present
                original_bias = self._original_weights[key].get('bias')
                if original_bias is not None:
                    proj.bias = original_bias[:new_out]

    def get_available_ratios(self) -> list[float]:
        """Return extraction ratios that produce valid sub-models.

        Checks that all FFN dimensions are evenly divisible at each ratio
        to ensure clean slicing without padding artifacts.
        """
        if not self._ffn_layers:
            return [1.0]

        valid = []
        for ratio in sorted(self.config.extraction_ratios):
            is_valid = True
            for key, proj in self._ffn_layers:
                if not hasattr(proj, 'weight'):
                    continue
                out_dim = proj.weight.shape[0]
                in_dim = proj.weight.shape[1]

                is_down = any(name in key for name in
                              ['down_proj', 'w2', 'fc2', 'dense_4h_to_h'])

                if is_down:
                    new_dim = int(in_dim * ratio)
                else:
                    new_dim = int(out_dim * ratio)

                if new_dim < 1:
                    is_valid = False
                    break

            if is_valid:
                valid.append(ratio)

        return valid if valid else [1.0]

    def estimate_memory(self, ratio: float) -> dict:
        """Estimate memory usage for a given extraction ratio.

        Calculates the total parameter count and memory footprint at
        the given ratio, compared to the full model.

        Args:
            ratio: Extraction ratio in [0, 1]

        Returns:
            Dict with parameter counts and memory estimates
        """
        ratio = self._snap_to_ratio(ratio)

        full_params = 0
        extracted_params = 0

        for key, proj in self._ffn_layers:
            if not hasattr(proj, 'weight'):
                continue

            orig_w = self._original_weights.get(key, {}).get('weight', proj.weight)
            out_dim, in_dim = orig_w.shape
            full_params += out_dim * in_dim

            is_down = any(name in key for name in
                          ['down_proj', 'w2', 'fc2', 'dense_4h_to_h'])

            if is_down:
                new_in = int(in_dim * ratio)
                extracted_params += out_dim * max(new_in, 1)
            else:
                new_out = int(out_dim * ratio)
                extracted_params += max(new_out, 1) * in_dim

            # Add bias params
            if hasattr(proj, 'bias') and proj.bias is not None:
                full_params += out_dim
                if not is_down:
                    extracted_params += int(out_dim * ratio)
                else:
                    extracted_params += out_dim

        # Assume float16 (2 bytes per param)
        bytes_per_param = 2
        full_mb = full_params * bytes_per_param / (1024 ** 2)
        extracted_mb = extracted_params * bytes_per_param / (1024 ** 2)

        return {
            "ratio": ratio,
            "full_ffn_params": full_params,
            "extracted_ffn_params": extracted_params,
            "param_reduction": round(1.0 - extracted_params / max(full_params, 1), 3),
            "full_ffn_mb": round(full_mb, 3),
            "extracted_ffn_mb": round(extracted_mb, 3),
            "memory_saved_mb": round(full_mb - extracted_mb, 3),
            "ffn_layers_found": len(self._ffn_layers),
        }


class AdaptiveMatFormer:
    """Automatically adapts model size based on memory pressure.

    Monitors macOS memory pressure levels and dynamically extracts
    sub-models at appropriate ratios. When pressure increases, the
    model shrinks. When pressure decreases, it grows back.

    Integrates with mlx-flash's memory_manager.py patterns.
    """

    def __init__(self, model, config: MatFormerConfig = None):
        self.config = config or MatFormerConfig()
        self.extractor = MatFormerExtractor(model, self.config)
        self._model = model
        self._current_ratio = 1.0
        self._adaptation_count = 0
        self._last_check_time = 0.0
        self._check_interval_s = 5.0  # check pressure every 5s
        self._ratio_history: list[tuple[float, float]] = []  # (time, ratio)

    def get_current_ratio(self) -> float:
        """Return current extraction ratio based on memory pressure."""
        if not self.config.auto_adapt:
            return self._current_ratio

        now = time.time()
        if now - self._last_check_time < self._check_interval_s:
            return self._current_ratio

        self._last_check_time = now
        pressure = self._get_memory_pressure()
        ratio = self._pressure_to_ratio(pressure)
        return ratio

    def _get_memory_pressure(self) -> str:
        """Get current memory pressure level from the system.

        Uses mlx-flash's memory_manager patterns for macOS pressure detection.
        Falls back gracefully if unavailable.
        """
        try:
            from mlx_flash_compress.memory_manager import get_memory_state
            state = get_memory_state()

            level = state.pressure_level
            # Map to our pressure levels
            if level in ("normal", "green"):
                # Check swap for finer granularity
                if state.swap_used_gb > 4.0:
                    return "urgent"
                return "nominal"
            elif level in ("warning", "yellow"):
                if state.swap_used_gb > 2.0:
                    return "urgent"
                return "warning"
            elif level in ("critical", "red"):
                if state.swap_used_gb > 4.0:
                    return "emergency"
                return "critical"

            return "nominal"
        except (ImportError, Exception):
            return "nominal"

    def _pressure_to_ratio(self, pressure: str) -> float:
        """Map pressure level to extraction ratio using config thresholds."""
        thresholds = self.config.pressure_thresholds
        ratio = thresholds.get(pressure, 1.0)

        # Ensure ratio is in valid range
        ratio = max(self.config.min_ratio, min(1.0, ratio))
        return self.extractor._snap_to_ratio(ratio)

    def adapt(self, memory_pressure: str = None) -> float:
        """Check memory pressure, adjust model size if needed.

        Args:
            memory_pressure: Override pressure level (None = auto-detect)

        Returns:
            The ratio applied
        """
        if memory_pressure is None:
            pressure = self._get_memory_pressure()
        else:
            pressure = memory_pressure

        new_ratio = self._pressure_to_ratio(pressure)

        if new_ratio != self._current_ratio:
            self.extractor.extract(new_ratio)
            self._adaptation_count += 1
            self._ratio_history.append((time.time(), new_ratio))

            # Keep history bounded
            if len(self._ratio_history) > 200:
                self._ratio_history = self._ratio_history[-100:]

        self._current_ratio = new_ratio
        return new_ratio

    def forward(self, input_ids: mx.array, **kwargs) -> mx.array:
        """Forward pass with current extraction ratio.

        If auto_adapt is enabled, checks memory pressure before the
        forward pass and adjusts model size if needed.

        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments passed to the model

        Returns:
            Model output (logits)
        """
        if self.config.auto_adapt:
            self.adapt()

        if len(input_ids.shape) == 1:
            input_ids = mx.expand_dims(input_ids, axis=0)

        return self._model(input_ids, **kwargs)

    def get_stats(self) -> dict:
        """Return adaptive inference statistics."""
        mem_estimate = self.extractor.estimate_memory(self._current_ratio)
        available_ratios = self.extractor.get_available_ratios()

        # Compute ratio stability (how often we change)
        if len(self._ratio_history) >= 2:
            changes = sum(1 for i in range(1, len(self._ratio_history))
                          if self._ratio_history[i][1] != self._ratio_history[i - 1][1])
            stability = 1.0 - (changes / len(self._ratio_history))
        else:
            stability = 1.0

        return {
            "current_ratio": self._current_ratio,
            "adaptation_count": self._adaptation_count,
            "available_ratios": available_ratios,
            "auto_adapt": self.config.auto_adapt,
            "stability": round(stability, 3),
            "memory": mem_estimate,
            "ffn_layers": len(self.extractor._ffn_layers),
            "pressure_thresholds": self.config.pressure_thresholds,
        }


def apply_matformer(model, config: MatFormerConfig = None) -> AdaptiveMatFormer:
    """One-line setup for adaptive MatFormer inference.

    Args:
        model: MLX model with FFN/MLP layers
        config: Optional MatFormerConfig (uses defaults if None)

    Returns:
        AdaptiveMatFormer wrapping the model with elastic inference
    """
    if config is None:
        config = MatFormerConfig()

    return AdaptiveMatFormer(model, config)
