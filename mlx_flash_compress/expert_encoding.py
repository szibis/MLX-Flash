"""Expert weight encoding strategies for compressible MoE weight storage.

Results on real Qwen1.5-MoE-A2.7B-Chat-4bit weights:

  4-bit quantized weights (uint32 packed, 89% of expert data):
    - Entropy: 7.52 bits/byte (out of 8.0 maximum)
    - LZ4:  1.00x (zero compression, every strategy tested)
    - ZSTD: 1.06x (6% — not worth the decompress cost)
    - Nibble stream separation + ZSTD: 1.91x on the 4-bit values,
      but expands the data 2x first (net ~1.0x)
    - XOR row delta: 1.00x (rows are uncorrelated)
    - Byte-plane separation: 1.00x (bytes are equally random)

  Scales + biases (float16, 11% of expert data):
    - Entropy: 4.5-5.0 bits/byte
    - Dictionary encoding: 1.9x (only 2,512 unique scale values
      across 2.7M entries)
    - ZSTD: 1.6-1.8x
    - Sorted + ZSTD: 734x (but impractical — destroys ordering)

  Combined best strategy:
    - Nibble separation + dict-encoded scales + ZSTD: 0.99x net
    - The weight portion dominates and resists all compression.

  Fundamental reason: A well-calibrated 4-bit quantizer maps weights
  to 16 bins with near-uniform utilization = maximum information density.
  The data is DESIGNED to be as compact as possible. Standard compression
  cannot improve on purpose-built quantization.

These encoders are provided for analysis and experimentation. For actual
MoE inference acceleration, the viable paths are:
  1. Lower-bit quantization (2-bit/3-bit) with quality trade-off
  2. Expert pruning (fewer experts per layer)
  3. Expert merging (combine similar experts, reduce K)
  4. Prediction-based prefetching (not compression, but latency hiding)
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from mlx_flash_compress.compression import LZ4Compressor, ZSTDCompressor


@dataclass
class EncodingResult:
    """Result of encoding an expert weight tensor."""
    name: str
    original_bytes: int
    encoded_bytes: int
    compressed_bytes: int  # after LZ4/ZSTD
    ratio_raw: float       # encoded/original
    ratio_compressed: float  # original/compressed
    encode_ms: float
    decode_ms: float
    compress_algo: str


def _time_ms(fn):
    t0 = time.monotonic()
    result = fn()
    return result, (time.monotonic() - t0) * 1000


class NibbleStreamEncoder:
    """Separate uint32 packed weights into 8 nibble streams.

    Each uint32 contains 8 x 4-bit values. Separating by nibble position
    creates streams with lower entropy (4 bits vs 8 bits per value).
    ZSTD can exploit this structure for ~1.9x on the nibble data alone.
    """

    def encode(self, weight_uint32: np.ndarray) -> bytes:
        """Encode (rows, cols) uint32 array into 8 nibble byte streams."""
        flat = weight_uint32.ravel()
        streams = []
        for shift in range(0, 32, 4):
            stream = ((flat >> shift) & 0xF).astype(np.uint8)
            streams.append(stream.tobytes())
        return b''.join(streams)

    def decode(self, data: bytes, shape: tuple, dtype=np.uint32) -> np.ndarray:
        """Reconstruct uint32 array from 8 nibble streams."""
        n_elements = 1
        for s in shape:
            n_elements *= s
        stream_len = n_elements

        result = np.zeros(n_elements, dtype=np.uint32)
        offset = 0
        for shift in range(0, 32, 4):
            stream = np.frombuffer(data[offset:offset + stream_len], dtype=np.uint8)
            result |= stream.astype(np.uint32) << shift
            offset += stream_len

        return result.reshape(shape)


class DictScaleEncoder:
    """Dictionary-encode scales and biases.

    Real quantized models have very few unique scale values (~2,500 out
    of millions). Mapping to uint16 indices saves ~47% on the scale data.
    """

    def encode(self, scales: np.ndarray, biases: np.ndarray) -> tuple[bytes, bytes]:
        """Returns (dictionary_bytes, indices_bytes)."""
        s_flat = scales.ravel().view(np.uint16)
        b_flat = biases.ravel().view(np.uint16)
        pairs = np.stack([s_flat, b_flat], axis=1)

        unique_pairs, indices = np.unique(pairs, axis=0, return_inverse=True)
        if len(unique_pairs) < 65536:
            idx_dtype = np.uint16
        else:
            idx_dtype = np.uint32

        dict_bytes = unique_pairs.tobytes()
        idx_bytes = indices.astype(idx_dtype).tobytes()

        # Header: num_unique (4 bytes) + idx_dtype_size (1 byte)
        header = len(unique_pairs).to_bytes(4, 'little')
        header += idx_dtype().itemsize.to_bytes(1, 'little')

        return header + dict_bytes, idx_bytes

    def decode(self, dict_data: bytes, idx_data: bytes, shape: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct scales and biases from dictionary encoding."""
        num_unique = int.from_bytes(dict_data[:4], 'little')
        idx_size = dict_data[4]
        dict_array = np.frombuffer(dict_data[5:], dtype=np.uint16).reshape(num_unique, 2)

        idx_dtype = np.uint16 if idx_size == 2 else np.uint32
        indices = np.frombuffer(idx_data, dtype=idx_dtype)

        pairs = dict_array[indices]
        scales = pairs[:, 0].view(np.float16).reshape(shape)
        biases = pairs[:, 1].view(np.float16).reshape(shape)
        return scales, biases


class XORDeltaEncoder:
    """XOR row-delta encoding for weight matrices.

    Adjacent rows in a weight matrix may share some bit patterns.
    XOR-delta creates runs of zeros where rows are identical.
    In practice, 4-bit quantized rows are mostly uncorrelated (gain: ~0%).
    """

    def encode(self, data: np.ndarray) -> np.ndarray:
        """XOR each row with the previous row."""
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, data.shape[0]):
            result[i] = np.bitwise_xor(data[i], data[i - 1])
        return result

    def decode(self, data: np.ndarray) -> np.ndarray:
        """Reconstruct from XOR deltas."""
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, data.shape[0]):
            result[i] = np.bitwise_xor(result[i - 1], data[i])
        return result


class CombinedExpertEncoder:
    """Best-effort combined encoder for a single expert.

    Applies all viable strategies:
    1. Separate weight/scales/biases components
    2. Nibble-stream separation on weights
    3. Dictionary encoding on scales+biases
    4. ZSTD compression on the encoded data

    Honest result: ~1.0-1.1x on real 4-bit quantized data.
    The weight data (89% of expert) is fundamentally incompressible.
    """

    def __init__(self, compress_algo: str = "zstd"):
        self._nibble = NibbleStreamEncoder()
        self._dict = DictScaleEncoder()
        self._xor = XORDeltaEncoder()
        if compress_algo == "zstd":
            self._compressor = ZSTDCompressor(level=3)
        else:
            self._compressor = LZ4Compressor()
        self._algo = compress_algo

    def encode_expert(
        self,
        weight: np.ndarray,
        scales: np.ndarray,
        biases: np.ndarray,
    ) -> tuple[bytes, dict]:
        """Encode a single expert's weight components.

        Returns (compressed_bytes, metadata for decoding).
        """
        t0 = time.monotonic()

        # 1. Nibble-stream separation on weights
        nibble_data = self._nibble.encode(weight)

        # 2. Dictionary encode scales+biases
        dict_data, idx_data = self._dict.encode(scales, biases)

        # 3. Combine all encoded streams
        # Header: component sizes for splitting on decode
        w_size = len(nibble_data)
        d_size = len(dict_data)
        i_size = len(idx_data)

        import struct
        header = struct.pack('<III', w_size, d_size, i_size)
        combined = header + nibble_data + dict_data + idx_data

        encode_time = (time.monotonic() - t0) * 1000

        # 4. Compress
        t0 = time.monotonic()
        compressed = self._compressor.compress(combined)
        compress_time = (time.monotonic() - t0) * 1000

        original_size = weight.nbytes + scales.nbytes + biases.nbytes

        metadata = {
            "weight_shape": weight.shape,
            "scales_shape": scales.shape,
            "weight_dtype": str(weight.dtype),
            "original_bytes": original_size,
            "encoded_bytes": len(combined),
            "compressed_bytes": compressed.compressed_size,
            "ratio": original_size / compressed.compressed_size if compressed.compressed_size > 0 else 0,
            "encode_ms": encode_time,
            "compress_ms": compress_time,
        }

        return compressed.data, metadata

    def decode_expert(
        self,
        compressed_data: bytes,
        metadata: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode a compressed expert back to weight/scales/biases."""
        import struct

        t0 = time.monotonic()

        # Decompress
        from mlx_flash_compress.compression import CompressedBuffer, CompressionAlgo
        buf = CompressedBuffer(
            data=compressed_data,
            original_size=metadata["encoded_bytes"],
            compressed_size=len(compressed_data),
            algo=CompressionAlgo.ZSTD if self._algo == "zstd" else CompressionAlgo.LZ4,
            compress_time_ms=0,
            layer_idx=0,
            expert_id=0,
        )
        combined = self._compressor.decompress(buf)

        # Split components
        w_size, d_size, i_size = struct.unpack('<III', combined[:12])
        offset = 12
        nibble_data = combined[offset:offset + w_size]
        offset += w_size
        dict_data = combined[offset:offset + d_size]
        offset += d_size
        idx_data = combined[offset:offset + i_size]

        # Decode weights
        weight = self._nibble.decode(nibble_data, metadata["weight_shape"])

        # Decode scales+biases
        scales, biases = self._dict.decode(dict_data, idx_data, metadata["scales_shape"])

        decode_time = (time.monotonic() - t0) * 1000
        return weight, scales, biases, decode_time


def benchmark_all_strategies(
    weight: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    expert_id: int = 0,
) -> list[EncodingResult]:
    """Run all encoding strategies on a single expert and return results."""
    lz4 = LZ4Compressor()
    zstd = ZSTDCompressor(3)
    original_size = weight.nbytes + scales.nbytes + biases.nbytes
    raw_bytes = weight.tobytes() + scales.tobytes() + biases.tobytes()

    results = []

    # Baseline: raw + LZ4
    buf = lz4.compress(raw_bytes)
    results.append(EncodingResult(
        name="Raw + LZ4",
        original_bytes=original_size,
        encoded_bytes=original_size,
        compressed_bytes=buf.compressed_size,
        ratio_raw=1.0,
        ratio_compressed=original_size / buf.compressed_size,
        encode_ms=0,
        decode_ms=0,
        compress_algo="LZ4",
    ))

    # Baseline: raw + ZSTD
    buf = zstd.compress(raw_bytes)
    results.append(EncodingResult(
        name="Raw + ZSTD",
        original_bytes=original_size,
        encoded_bytes=original_size,
        compressed_bytes=buf.compressed_size,
        ratio_raw=1.0,
        ratio_compressed=original_size / buf.compressed_size,
        encode_ms=0,
        decode_ms=0,
        compress_algo="ZSTD",
    ))

    # Nibble streams + ZSTD
    nibble_enc = NibbleStreamEncoder()
    encoded, enc_ms = _time_ms(lambda: nibble_enc.encode(weight))
    combined = encoded + scales.tobytes() + biases.tobytes()
    buf = zstd.compress(combined)
    _, dec_ms = _time_ms(lambda: zstd.decompress(buf))
    results.append(EncodingResult(
        name="Nibble streams + ZSTD",
        original_bytes=original_size,
        encoded_bytes=len(combined),
        compressed_bytes=buf.compressed_size,
        ratio_raw=original_size / len(combined),
        ratio_compressed=original_size / buf.compressed_size,
        encode_ms=enc_ms,
        decode_ms=dec_ms,
        compress_algo="ZSTD",
    ))

    # Dict scales + ZSTD
    dict_enc = DictScaleEncoder()
    (dict_data, idx_data), enc_ms = _time_ms(lambda: dict_enc.encode(scales, biases))
    combined_dict = weight.tobytes() + dict_data + idx_data
    buf = zstd.compress(combined_dict)
    _, dec_ms = _time_ms(lambda: zstd.decompress(buf))
    results.append(EncodingResult(
        name="Dict scales + ZSTD",
        original_bytes=original_size,
        encoded_bytes=len(combined_dict),
        compressed_bytes=buf.compressed_size,
        ratio_raw=original_size / len(combined_dict),
        ratio_compressed=original_size / buf.compressed_size,
        encode_ms=enc_ms,
        decode_ms=dec_ms,
        compress_algo="ZSTD",
    ))

    # XOR delta + ZSTD
    xor_enc = XORDeltaEncoder()
    w_xor, enc_ms = _time_ms(lambda: xor_enc.encode(weight))
    combined_xor = w_xor.tobytes() + scales.tobytes() + biases.tobytes()
    buf = zstd.compress(combined_xor)
    _, dec_ms = _time_ms(lambda: zstd.decompress(buf))
    results.append(EncodingResult(
        name="XOR row delta + ZSTD",
        original_bytes=original_size,
        encoded_bytes=len(combined_xor),
        compressed_bytes=buf.compressed_size,
        ratio_raw=1.0,
        ratio_compressed=original_size / buf.compressed_size,
        encode_ms=enc_ms,
        decode_ms=dec_ms,
        compress_algo="ZSTD",
    ))

    # Combined: nibble + dict + ZSTD
    combined_enc = CombinedExpertEncoder("zstd")
    (compressed, meta), enc_ms = _time_ms(
        lambda: combined_enc.encode_expert(weight, scales, biases)
    )
    _, _, _, dec_ms = combined_enc.decode_expert(compressed, meta)
    results.append(EncodingResult(
        name="COMBINED (nibble+dict+ZSTD)",
        original_bytes=original_size,
        encoded_bytes=meta["encoded_bytes"],
        compressed_bytes=meta["compressed_bytes"],
        ratio_raw=original_size / meta["encoded_bytes"],
        ratio_compressed=meta["ratio"],
        encode_ms=enc_ms,
        decode_ms=dec_ms,
        compress_algo="ZSTD",
    ))

    return results
