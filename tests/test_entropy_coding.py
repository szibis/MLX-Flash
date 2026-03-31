"""Tests for EntroLLM-inspired entropy coding."""
import numpy as np
import pytest
from mlx_flash_compress.entropy_coding import (
    HuffmanCodebook,
    analyze_distribution,
    encode_weights,
    decode_weights,
    compress_tensor,
    decompress_tensor,
)


class TestHuffmanCodebook:
    def test_from_simple_distribution(self):
        counts = {0: 100, 1: 50, 2: 25, 3: 12}
        cb = HuffmanCodebook.from_distribution(counts)
        assert cb.num_symbols == 4
        assert cb.avg_bits > 0
        # Most frequent symbol should have shortest code
        assert len(cb.codes[0]) <= len(cb.codes[3])

    def test_single_symbol(self):
        counts = {7: 1000}
        cb = HuffmanCodebook.from_distribution(counts)
        assert cb.num_symbols == 1
        assert cb.codes[7] == "0"
        assert cb.avg_bits == 1.0

    def test_empty_distribution(self):
        cb = HuffmanCodebook.from_distribution({})
        assert cb.num_symbols == 0
        assert cb.avg_bits == 0

    def test_two_symbols(self):
        counts = {0: 50, 1: 50}
        cb = HuffmanCodebook.from_distribution(counts)
        assert cb.num_symbols == 2
        assert cb.avg_bits == 1.0  # equal frequency = 1 bit each

    def test_prefix_free(self):
        """No code should be a prefix of another code."""
        counts = {i: max(100 - i * 10, 1) for i in range(10)}
        cb = HuffmanCodebook.from_distribution(counts)
        codes = list(cb.codes.values())
        for i, c1 in enumerate(codes):
            for j, c2 in enumerate(codes):
                if i != j:
                    assert not c2.startswith(c1), f"{c1} is prefix of {c2}"

    def test_compression_ratio(self):
        # Highly skewed distribution should compress well
        counts = {0: 900, 1: 50, 2: 30, 3: 10, 4: 5, 5: 3, 6: 1, 7: 1}
        cb = HuffmanCodebook.from_distribution(counts)
        ratio = cb.compression_ratio(nominal_bits=4)
        assert ratio > 1.0  # should achieve compression

    def test_decode_table_inverse_of_codes(self):
        counts = {0: 100, 1: 50, 2: 25}
        cb = HuffmanCodebook.from_distribution(counts)
        for symbol, code in cb.codes.items():
            assert cb.decode_table[code] == symbol


class TestAnalyzeDistribution:
    def test_uniform_distribution(self):
        weights = np.array([0, 1, 2, 3] * 100, dtype=np.uint8)
        dist = analyze_distribution(weights, nominal_bits=4)
        assert dist.total_values == 400
        assert dist.unique_values == 4
        assert abs(dist.entropy_bits - 2.0) < 0.01  # log2(4) = 2

    def test_skewed_distribution(self):
        # 90% zeros, 10% ones
        weights = np.array([0] * 900 + [1] * 100, dtype=np.uint8)
        dist = analyze_distribution(weights, nominal_bits=4)
        assert dist.entropy_bits < 1.0  # very low entropy
        assert dist.theoretical_compression > 4.0  # 4 bits / <1 bit

    def test_single_value(self):
        weights = np.zeros(1000, dtype=np.uint8)
        dist = analyze_distribution(weights, nominal_bits=4)
        assert dist.unique_values == 1
        assert dist.entropy_bits < 0.01  # near-zero entropy


class TestEncodeDecodeRoundtrip:
    def test_roundtrip_simple(self):
        weights = np.array([0, 1, 0, 1, 0, 0, 1, 0], dtype=np.uint8)
        dist = analyze_distribution(weights)
        cb = HuffmanCodebook.from_distribution(dist.counts)
        compressed, count = encode_weights(weights, cb)
        decoded = decode_weights(compressed, count, cb)
        np.testing.assert_array_equal(decoded, weights)

    def test_roundtrip_skewed(self):
        rng = np.random.default_rng(42)
        # Zipf-like: mostly low values
        weights = rng.choice(16, size=1000, p=_zipf_probs(16)).astype(np.uint8)
        dist = analyze_distribution(weights)
        cb = HuffmanCodebook.from_distribution(dist.counts)
        compressed, count = encode_weights(weights, cb)
        decoded = decode_weights(compressed, count, cb)
        np.testing.assert_array_equal(decoded, weights)

    def test_roundtrip_2d(self):
        weights = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.uint8)
        dist = analyze_distribution(weights)
        cb = HuffmanCodebook.from_distribution(dist.counts)
        compressed, count = encode_weights(weights, cb)
        decoded = decode_weights(compressed, count, cb, shape=weights.shape)
        np.testing.assert_array_equal(decoded, weights)

    def test_roundtrip_all_same(self):
        weights = np.full(500, 7, dtype=np.uint8)
        dist = analyze_distribution(weights)
        cb = HuffmanCodebook.from_distribution(dist.counts)
        compressed, count = encode_weights(weights, cb)
        decoded = decode_weights(compressed, count, cb)
        np.testing.assert_array_equal(decoded, weights)


class TestCompressDecompressTensor:
    def test_compress_basic(self):
        rng = np.random.default_rng(42)
        weights = rng.choice(16, size=(10, 20), p=_zipf_probs(16)).astype(np.uint8)
        result = compress_tensor(weights, nominal_bits=4)

        assert result["count"] == 200
        assert result["shape"] == (10, 20)
        assert result["effective_bits"] < 4.0  # should compress
        assert result["compression_ratio"] > 1.0

    def test_compress_decompress_roundtrip(self):
        rng = np.random.default_rng(42)
        weights = rng.choice(16, size=(5, 10), p=_zipf_probs(16)).astype(np.uint8)
        compressed = compress_tensor(weights)
        restored = decompress_tensor(compressed)
        np.testing.assert_array_equal(restored, weights)

    def test_compression_ratio_scales_with_skew(self):
        # More skewed = better compression
        uniform = np.arange(16, dtype=np.uint8).repeat(100)
        skewed = np.array([0] * 1200 + [1] * 200 + list(range(2, 16)) * 4, dtype=np.uint8)

        r_uniform = compress_tensor(uniform)
        r_skewed = compress_tensor(skewed)

        assert r_skewed["compression_ratio"] > r_uniform["compression_ratio"]

    def test_effective_bits_realistic(self):
        """Simulate realistic uint4 weight distribution (peaked around 7-8)."""
        rng = np.random.default_rng(42)
        # Normal-ish distribution centered at 7, clipped to [0,15]
        raw = rng.normal(7, 2, size=10000)
        weights = np.clip(np.round(raw), 0, 15).astype(np.uint8)
        result = compress_tensor(weights, nominal_bits=4)

        # Should achieve some compression (peaked distribution)
        assert result["effective_bits"] < 4.0
        assert result["entropy_bits"] < 4.0


def _zipf_probs(n):
    """Zipf distribution for n values."""
    p = np.array([(1.0 / (i + 1)) ** 1.0 for i in range(n)])
    return p / p.sum()
