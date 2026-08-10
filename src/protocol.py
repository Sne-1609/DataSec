# protocol.py — Network protocol for message framing
# ============================================================
# Handles sending and receiving structured messages between
# the two devices over TCP sockets.
# ============================================================

import struct
import json


def send_message(sock, msg_type: bytes, payload: bytes = b""):
    """
    Send a framed message over TCP.
    
    Frame format:
    [4 bytes: total length] [16 bytes: msg_type (padded)] [payload]
    """
    msg_type_padded = msg_type.ljust(16, b"\x00")
    total = 4 + 16 + len(payload)
    frame = struct.pack(">I", total) + msg_type_padded + payload
    sock.sendall(frame)


def recv_message(sock) -> tuple:
    """
    Receive a framed message from TCP.
    
    Returns: (msg_type: bytes, payload: bytes)
    """
    # Read length header (4 bytes)
    header = _recv_exact(sock, 4)
    if not header:
        raise ConnectionError("Connection closed")
    
    total_length = struct.unpack(">I", header)[0]
    
    # Read the rest
    remaining = total_length - 4
    data = _recv_exact(sock, remaining)
    
    msg_type = data[:16].rstrip(b"\x00")
    payload = data[16:]
    
    return msg_type, payload


def _recv_exact(sock, num_bytes: int) -> bytes:
    """Receive exactly num_bytes from socket."""
    data = b""
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionError("Connection closed during receive")
        data += chunk
    return data


def send_json(sock, msg_type: bytes, data: dict):
    """Send a JSON-serializable dict as payload."""
    payload = json.dumps(data).encode("utf-8")
    send_message(sock, msg_type, payload)


def recv_json(sock) -> tuple:
    """Receive a message and parse the payload as JSON."""
    msg_type, payload = recv_message(sock)
    data = json.loads(payload.decode("utf-8")) if payload else {}
    return msg_type, data
