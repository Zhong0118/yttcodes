from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict


def send_json(sock: socket.socket, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj).encode("utf-8")
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


def recv_json(sock: socket.socket) -> Dict[str, Any]:
    header = recv_exact(sock, 4)
    length = struct.unpack("!I", header)[0]
    body = recv_exact(sock, length)
    return json.loads(body.decode("utf-8"))
