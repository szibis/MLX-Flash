"""Python client for the Rust expert cache Unix socket."""
import json
import socket
import struct
from typing import Optional


def encode_message(msg: dict) -> bytes:
    """Encode a message as length-prefixed JSON (matches Rust protocol)."""
    data = json.dumps(msg).encode("utf-8")
    return struct.pack(">I", len(data)) + data


def decode_message(buf: bytes) -> Optional[tuple[dict, int]]:
    """Decode a length-prefixed JSON message. Returns (msg, bytes_consumed) or None."""
    if len(buf) < 4:
        return None
    length = struct.unpack(">I", buf[:4])[0]
    if len(buf) < 4 + length:
        return None
    msg = json.loads(buf[4:4 + length])
    return msg, 4 + length


class RustCacheClient:
    """Client for the Rust expert cache Unix socket server."""

    def __init__(self, socket_path: str = "/tmp/mlx-flash-cache.sock"):
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None

    def connect(self):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.socket_path)

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def _send_recv(self, msg: dict) -> dict:
        if self._sock is None:
            self.connect()
        data = encode_message(msg)
        self._sock.sendall(data)
        header = self._recv_exact(4)
        length = struct.unpack(">I", header)[0]
        body = self._recv_exact(length)
        return json.loads(body)

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed")
            buf += chunk
        return buf

    def fetch_experts(self, layer: int, experts: list[int], request_id: int = 0) -> dict:
        return self._send_recv({
            "FetchExperts": {"layer": layer, "experts": experts, "request_id": request_id}
        })

    def report_routing(self, layer: int, activated: list[int], token_idx: int) -> dict:
        return self._send_recv({
            "RoutingReport": {"layer": layer, "activated": activated, "token_idx": token_idx}
        })
