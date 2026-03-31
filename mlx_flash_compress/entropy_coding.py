"""EntroLLM-inspired entropy coding for quantized weights.

Standard uint4 quantization uses exactly 4 bits per weight. But the
distribution of quantized values is highly non-uniform — some values
appear far more often than others. Huffman coding exploits this to
achieve effective bit-widths far below 4 bits.

Key results from EntroLLM (arXiv:2505.02380):
  uint4 nominal: 4 bits → effective 1.39 bits (65% compression)
  uint8 nominal: 8 bits → effective 5.58 bits (30% compression)
  Quality loss: <0.12 perplexity points on Mistral-7B

This module provides:
  1. HuffmanCodebook: build optimal prefix codes from weight distributions
  2. encode_weights: compress uint4 weights to variable-length bitstream
  3. decode_weights: restore original uint4 weights from bitstream
  4. analyze_distribution: measure entropy and potential compression ratio
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
import heapq
import struct


# -- Huffman tree --

class _HuffmanNode:
    """Node in a Huffman tree."""

    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


@dataclass
class HuffmanCodebook:
    """Huffman codebook mapping symbols to variable-length bit codes."""
    codes: dict = field(default_factory=dict)  # symbol -> bit string
    decode_table: dict = field(default_factory=dict)  # bit string -> symbol
    avg_bits: float = 0.0
    max_bits: int = 0
    num_symbols: int = 0

    @classmethod
    def from_distribution(cls, counts: dict) -> "HuffmanCodebook":
        """Build a Huffman codebook from symbol frequency counts."""
        if not counts:
            return cls()

        # Build priority queue of leaf nodes
        heap = []
        for symbol, count in counts.items():
            heapq.heappush(heap, _HuffmanNode(symbol=symbol, freq=count))

        # Handle single-symbol case
        if len(heap) == 1:
            node = heapq.heappop(heap)
            codes = {node.symbol: "0"}
            total = node.freq
            return cls(
                codes=codes,
                decode_table={"0": node.symbol},
                avg_bits=1.0,
                max_bits=1,
                num_symbols=1,
            )

        # Build tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = _HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, parent)

        root = heapq.heappop(heap)

        # Extract codes via DFS
        codes = {}
        _extract_codes(root, "", codes)

        # Build decode table
        decode_table = {v: k for k, v in codes.items()}

        # Compute average bits
        total = sum(counts.values())
        avg_bits = sum(len(codes[s]) * c for s, c in counts.items()) / max(total, 1)
        max_bits = max(len(v) for v in codes.values()) if codes else 0

        return cls(
            codes=codes,
            decode_table=decode_table,
            avg_bits=avg_bits,
            max_bits=max_bits,
            num_symbols=len(codes),
        )

    def encode_symbol(self, symbol) -> str:
        """Encode a single symbol to its bit string."""
        return self.codes.get(symbol, "")

    def compression_ratio(self, nominal_bits: int = 4) -> float:
        """Compression ratio vs nominal bit width."""
        if self.avg_bits == 0:
            return 1.0
        return nominal_bits / self.avg_bits


def _extract_codes(node, prefix, codes):
    """DFS to extract Huffman codes from tree."""
    if node.symbol is not None:
        codes[node.symbol] = prefix if prefix else "0"
        return
    if node.left:
        _extract_codes(node.left, prefix + "0", codes)
    if node.right:
        _extract_codes(node.right, prefix + "1", codes)


# -- Weight analysis --

@dataclass
class WeightDistribution:
    """Analysis of a quantized weight tensor's value distribution."""
    counts: dict = field(default_factory=dict)
    total_values: int = 0
    unique_values: int = 0
    entropy_bits: float = 0.0
    nominal_bits: int = 4
    theoretical_compression: float = 1.0


def analyze_distribution(weights: np.ndarray, nominal_bits: int = 4) -> WeightDistribution:
    """Analyze the distribution of quantized weight values.

    For uint4: values are in [0, 15]
    For uint8: values are in [0, 255]
    """
    flat = weights.flatten()
    counts = dict(Counter(flat.tolist()))
    total = len(flat)

    # Shannon entropy
    probs = np.array(list(counts.values()), dtype=np.float64) / total
    entropy = -np.sum(probs * np.log2(probs + 1e-15))

    return WeightDistribution(
        counts=counts,
        total_values=total,
        unique_values=len(counts),
        entropy_bits=round(entropy, 4),
        nominal_bits=nominal_bits,
        theoretical_compression=round(nominal_bits / max(entropy, 0.001), 4),
    )


# -- Encoding / Decoding --

def encode_weights(weights: np.ndarray, codebook: HuffmanCodebook) -> tuple[bytes, int]:
    """Encode quantized weights to a compressed bitstream.

    Returns: (compressed_bytes, original_count)
    """
    flat = weights.flatten().tolist()
    bits = []
    for val in flat:
        code = codebook.encode_symbol(val)
        bits.append(code)

    bitstring = "".join(bits)
    # Pad to byte boundary
    pad_bits = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * pad_bits

    # Convert to bytes
    result = bytearray()
    for i in range(0, len(bitstring), 8):
        byte = int(bitstring[i:i + 8], 2)
        result.append(byte)

    return bytes(result), len(flat)


def decode_weights(data: bytes, count: int, codebook: HuffmanCodebook,
                   shape: tuple = None) -> np.ndarray:
    """Decode compressed bitstream back to quantized weights."""
    # Convert bytes to bit string
    bitstring = "".join(f"{byte:08b}" for byte in data)

    values = []
    pos = 0
    while len(values) < count and pos < len(bitstring):
        # Try to match prefix codes (greedy — Huffman codes are prefix-free)
        found = False
        for length in range(1, codebook.max_bits + 1):
            if pos + length > len(bitstring):
                break
            candidate = bitstring[pos:pos + length]
            if candidate in codebook.decode_table:
                values.append(codebook.decode_table[candidate])
                pos += length
                found = True
                break
        if not found:
            break

    result = np.array(values[:count])
    if shape is not None:
        result = result.reshape(shape)
    return result


# -- High-level API --

def compress_tensor(weights: np.ndarray, nominal_bits: int = 4) -> dict:
    """Compress a quantized weight tensor using Huffman coding.

    Returns dict with compressed data, codebook, and statistics.
    """
    dist = analyze_distribution(weights, nominal_bits)
    codebook = HuffmanCodebook.from_distribution(dist.counts)
    compressed, count = encode_weights(weights, codebook)

    original_size = weights.nbytes
    compressed_size = len(compressed)

    return {
        "compressed": compressed,
        "count": count,
        "shape": weights.shape,
        "codebook": codebook,
        "distribution": dist,
        "original_bytes": original_size,
        "compressed_bytes": compressed_size,
        "compression_ratio": round(original_size / max(compressed_size, 1), 2),
        "effective_bits": round(codebook.avg_bits, 4),
        "entropy_bits": dist.entropy_bits,
    }


def decompress_tensor(compressed_data: dict) -> np.ndarray:
    """Decompress a tensor from its compressed representation."""
    return decode_weights(
        compressed_data["compressed"],
        compressed_data["count"],
        compressed_data["codebook"],
        compressed_data["shape"],
    )
