"""Tests for the Python<->Rust Unix socket bridge protocol."""
import json
import struct
import pytest
from mlx_flash_compress.rust_bridge import encode_message, decode_message


class TestProtocol:
    def test_encode_fetch_request(self):
        msg = {"FetchExperts": {"layer": 5, "experts": [1, 2, 3], "request_id": 42}}
        data = encode_message(msg)
        assert len(data) > 4
        length = struct.unpack(">I", data[:4])[0]
        assert length == len(data) - 4
        parsed = json.loads(data[4:])
        assert parsed["FetchExperts"]["layer"] == 5

    def test_decode_response(self):
        msg = {"ExpertData": {"request_id": 42, "expert_sizes": [256, 256]}}
        encoded = encode_message(msg)
        decoded, consumed = decode_message(encoded)
        assert decoded["ExpertData"]["request_id"] == 42
        assert consumed == len(encoded)

    def test_decode_partial_returns_none(self):
        result = decode_message(b"\x00\x00\x00\x10short")
        assert result is None

    def test_roundtrip(self):
        msg = {"RoutingReport": {"layer": 3, "activated": [1, 5, 9], "token_idx": 77}}
        encoded = encode_message(msg)
        decoded, _ = decode_message(encoded)
        assert decoded == msg
