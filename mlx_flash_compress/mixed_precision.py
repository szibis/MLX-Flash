"""Multi-precision expert quantization with configurable hot/warm/cold tiers.

Supports 6 precision levels for expert weights:

  Precision  | Bits | Size per 1K params | Quality     | Use Case
  -----------|------|--------------------|-----------  |----------
  FP16       | 16   | 2.0 KB             | Lossless    | Critical experts (top 5%)
  Q8         | 8    | 1.0 KB             | Near-perfect| Hot experts (top 20%)
  Q6         | 6    | 0.75 KB            | Very good   | Warm-hot experts
  Q5         | 5    | 0.625 KB           | Good        | Warm experts
  Q4 (default)| 4   | 0.5 KB             | Standard    | Base model format
  Q3         | 3    | 0.375 KB           | Acceptable  | Cold-warm experts
  Q2         | 2    | 0.25 KB            | Lossy       | Cold experts (bottom 30%)

The tier assignment is based on expert activation frequency:
  - frequency > 15%  → FP16 (keep full precision for most-used experts)
  - frequency > 8%   → Q8 (near-lossless for frequently-used experts)
  - frequency > 5%   → Q4 (standard, no requantization needed)
  - frequency > 2%   → Q3 (slight quality trade-off)
  - frequency < 2%   → Q2 (aggressive compression for rarely-used experts)

References:
  - DynaExq (arxiv.org/abs/2511.15015) — dynamic expert quantization
  - HOBBIT (arxiv.org/abs/2411.01433) — mixed precision per-expert
"""

import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

import numpy as np


@dataclass
class ExpertHotness:
    """Track expert activation frequency for mixed-precision decisions."""
    activation_counts: dict = field(default_factory=lambda: defaultdict(int))
    total_tokens: int = 0
    window_size: int = 1000  # rolling window for frequency estimation

    def record(self, layer_idx: int, expert_ids: list[int]):
        for eid in expert_ids:
            self.activation_counts[(layer_idx, eid)] += 1
        self.total_tokens += 1

    def get_frequency(self, layer_idx: int, expert_id: int) -> float:
        """Get activation frequency as fraction of total tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.activation_counts[(layer_idx, expert_id)] / self.total_tokens

    def classify(self, layer_idx: int, expert_id: int, threshold: float = 0.05) -> str:
        """Classify expert as 'hot' (>threshold) or 'cold' (<threshold)."""
        freq = self.get_frequency(layer_idx, expert_id)
        return "hot" if freq >= threshold else "cold"

    def classify_precision(self, layer_idx: int, expert_id: int) -> str:
        """Classify expert into a precision tier based on activation frequency.

        Returns one of: 'fp16', 'q8', 'q4', 'q3', 'q2'
        """
        freq = self.get_frequency(layer_idx, expert_id)
        if freq >= 0.15:
            return "fp16"
        elif freq >= 0.08:
            return "q8"
        elif freq >= 0.05:
            return "q4"
        elif freq >= 0.02:
            return "q3"
        else:
            return "q2"


# -- Precision tier definitions --

PRECISION_TIERS = {
    "fp16": {"bits": 16, "bytes_per_param": 2.0, "quality": "lossless"},
    "q8":   {"bits": 8,  "bytes_per_param": 1.0, "quality": "near-perfect"},
    "q6":   {"bits": 6,  "bytes_per_param": 0.75, "quality": "very good"},
    "q5":   {"bits": 5,  "bytes_per_param": 0.625, "quality": "good"},
    "q4":   {"bits": 4,  "bytes_per_param": 0.5, "quality": "standard"},
    "q3":   {"bits": 3,  "bytes_per_param": 0.375, "quality": "acceptable"},
    "q2":   {"bits": 2,  "bytes_per_param": 0.25, "quality": "lossy"},
}


def estimate_tier_savings(
    num_experts: int,
    expert_params: int,
    activation_frequencies: dict,
) -> dict:
    """Estimate memory savings from multi-tier precision.

    Args:
        num_experts: Total number of experts per layer
        expert_params: Parameters per expert
        activation_frequencies: {expert_id: frequency} dict

    Returns dict with per-tier counts and total savings.
    """
    hotness = ExpertHotness()
    hotness.total_tokens = 1000  # normalize
    for eid, freq in activation_frequencies.items():
        hotness.activation_counts[(0, eid)] = int(freq * 1000)

    tier_counts = {"fp16": 0, "q8": 0, "q4": 0, "q3": 0, "q2": 0}
    for eid in range(num_experts):
        tier = hotness.classify_precision(0, eid)
        tier_counts[tier] += 1

    # Calculate total bytes
    baseline_bytes = num_experts * expert_params * 0.5  # Q4 baseline
    tiered_bytes = sum(
        tier_counts[tier] * expert_params * PRECISION_TIERS[tier]["bytes_per_param"]
        for tier in tier_counts
    )

    return {
        "tier_counts": tier_counts,
        "baseline_bytes": baseline_bytes,
        "tiered_bytes": tiered_bytes,
        "savings_ratio": 1.0 - (tiered_bytes / baseline_bytes) if baseline_bytes > 0 else 0,
        "effective_bits": (tiered_bytes * 8) / (num_experts * expert_params) if expert_params > 0 else 0,
    }


def requantize_4bit_to_2bit(
    weight_uint32: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    group_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Requantize 4-bit packed uint32 weights to 2-bit packed uint8.

    The 4-bit format stores 8 nibbles per uint32.
    The 2-bit format stores 4 crumbs per uint8.

    Process per group:
    1. Unpack 4-bit nibbles (0-15)
    2. Map to 2-bit crumbs (0-3) via rounding: crumb = round(nibble / 5)
    3. Compute new scale/bias for the 2-bit range
    4. Pack 4 crumbs per uint8

    Returns: (packed_2bit, new_scales, new_biases, metadata)
    """
    rows, cols_u32 = weight_uint32.shape
    nibbles_per_row = cols_u32 * 8  # 8 nibbles per uint32

    # Unpack all 4-bit nibbles
    w_bytes = weight_uint32.view(np.uint8)  # (rows, cols_u32*4)
    low = (w_bytes & 0x0F).astype(np.float32)
    high = ((w_bytes >> 4) & 0x0F).astype(np.float32)
    # Interleave: low[0], high[0], low[1], high[1], ...
    nibbles = np.empty((rows, w_bytes.shape[1] * 2), dtype=np.float32)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high

    # Dequantize to approximate float values using original scales/biases
    # For MLX 4-bit: value = nibble * scale + bias (per group)
    s = np.array(scales, dtype=np.float32)  # (rows, n_groups)
    b = np.array(biases, dtype=np.float32)
    n_groups = s.shape[1]
    weights_per_group = nibbles.shape[1] // n_groups

    # Expand scales/biases to match nibble positions
    s_expanded = np.repeat(s, weights_per_group, axis=1)[:, :nibbles.shape[1]]
    b_expanded = np.repeat(b, weights_per_group, axis=1)[:, :nibbles.shape[1]]
    float_values = nibbles * s_expanded + b_expanded

    # Requantize to 2-bit (4 levels: 0, 1, 2, 3)
    # Per group: find min/max, compute new scale/bias
    float_groups = float_values.reshape(rows, n_groups, weights_per_group)

    new_scales = np.zeros((rows, n_groups), dtype=np.float16)
    new_biases = np.zeros((rows, n_groups), dtype=np.float16)
    crumbs_all = np.zeros_like(nibbles, dtype=np.uint8)

    for g in range(n_groups):
        group_vals = float_groups[:, g, :]  # (rows, weights_per_group)
        gmin = group_vals.min(axis=1, keepdims=True)
        gmax = group_vals.max(axis=1, keepdims=True)
        grange = gmax - gmin
        grange = np.where(grange < 1e-10, 1e-10, grange)

        # Scale to 0-3 range
        normalized = (group_vals - gmin) / grange * 3.0
        crumbs = np.clip(np.round(normalized), 0, 3).astype(np.uint8)

        start = g * weights_per_group
        end = start + weights_per_group
        crumbs_all[:, start:end] = crumbs

        new_scales[:, g] = np.float16((grange / 3.0).squeeze())
        new_biases[:, g] = np.float16(gmin.squeeze())

    # Pack 4 crumbs per uint8
    n_crumbs = crumbs_all.shape[1]
    n_packed = (n_crumbs + 3) // 4
    packed = np.zeros((rows, n_packed), dtype=np.uint8)
    for i in range(min(4, n_crumbs)):
        idx = i
        while idx < n_crumbs:
            byte_pos = idx // 4
            bit_pos = (idx % 4) * 2
            packed[:, byte_pos] |= crumbs_all[:, idx] << bit_pos
            idx += 4
            break
    # Proper packing
    packed = np.zeros((rows, n_packed), dtype=np.uint8)
    for j in range(n_crumbs):
        byte_idx = j // 4
        bit_shift = (j % 4) * 2
        packed[:, byte_idx] |= crumbs_all[:, j] << bit_shift

    metadata = {
        "original_shape": weight_uint32.shape,
        "packed_shape": packed.shape,
        "n_groups": n_groups,
        "weights_per_group": weights_per_group,
        "original_bytes": weight_uint32.nbytes + scales.nbytes + biases.nbytes,
        "packed_bytes": packed.nbytes + new_scales.nbytes + new_biases.nbytes,
    }
    metadata["ratio"] = metadata["original_bytes"] / metadata["packed_bytes"] if metadata["packed_bytes"] > 0 else 0

    return packed, new_scales, new_biases, metadata


def dequantize_2bit(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    n_values_per_row: int,
) -> np.ndarray:
    """Dequantize 2-bit packed values back to float32."""
    rows = packed.shape[0]
    n_groups = scales.shape[1]
    weights_per_group = n_values_per_row // n_groups

    # Unpack crumbs
    crumbs = np.zeros((rows, n_values_per_row), dtype=np.uint8)
    for j in range(n_values_per_row):
        byte_idx = j // 4
        bit_shift = (j % 4) * 2
        crumbs[:, j] = (packed[:, byte_idx] >> bit_shift) & 0x03

    # Dequantize
    s = np.array(scales, dtype=np.float32)
    b = np.array(biases, dtype=np.float32)
    s_expanded = np.repeat(s, weights_per_group, axis=1)[:, :n_values_per_row]
    b_expanded = np.repeat(b, weights_per_group, axis=1)[:, :n_values_per_row]

    return crumbs.astype(np.float32) * s_expanded + b_expanded


@dataclass
class MixedPrecisionResult:
    """Result of mixed-precision requantization benchmark."""
    expert_id: int
    original_bytes: int
    q4_bytes: int
    q2_bytes: int
    ratio_4to2: float
    requant_ms: float
    mse: float  # mean squared error vs 4-bit dequantized values
    max_error: float


def benchmark_mixed_precision(
    weight: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    expert_id: int = 0,
) -> MixedPrecisionResult:
    """Benchmark 4-bit to 2-bit requantization on a single expert."""
    t0 = time.monotonic()
    packed_2bit, new_scales, new_biases, meta = requantize_4bit_to_2bit(
        weight, scales, biases
    )
    requant_time = (time.monotonic() - t0) * 1000

    # Measure quality: dequantize both and compare
    # 4-bit dequantized values
    w_bytes = weight.view(np.uint8)
    low = (w_bytes & 0x0F).astype(np.float32)
    high = ((w_bytes >> 4) & 0x0F).astype(np.float32)
    nibbles = np.empty((weight.shape[0], w_bytes.shape[1] * 2), dtype=np.float32)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high

    s4 = np.array(scales, dtype=np.float32)
    b4 = np.array(biases, dtype=np.float32)
    n_groups = s4.shape[1]
    wpg = nibbles.shape[1] // n_groups
    s4_exp = np.repeat(s4, wpg, axis=1)[:, :nibbles.shape[1]]
    b4_exp = np.repeat(b4, wpg, axis=1)[:, :nibbles.shape[1]]
    values_4bit = nibbles * s4_exp + b4_exp

    # 2-bit dequantized values
    values_2bit = dequantize_2bit(packed_2bit, new_scales, new_biases, nibbles.shape[1])

    # Error metrics
    diff = values_4bit - values_2bit
    mse = float(np.mean(diff ** 2))
    max_err = float(np.max(np.abs(diff)))

    return MixedPrecisionResult(
        expert_id=expert_id,
        original_bytes=meta["original_bytes"],
        q4_bytes=meta["original_bytes"],
        q2_bytes=meta["packed_bytes"],
        ratio_4to2=meta["ratio"],
        requant_ms=requant_time,
        mse=mse,
        max_error=max_err,
    )
