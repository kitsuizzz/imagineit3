"""Application-level authenticated encryption for the app <-> GPU channel.

The transport is a plain WebSocket.  Every application message after the
handshake is encrypted with AES-256-GCM using keys derived from a per-worker
pre-shared key.  This means the security of prompts, parameters, progress and
image bytes does not depend on TLS, although running over WSS is still
recommended as defence in depth.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from typing import Any, Literal

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PROTOCOL_VERSION = 1
NONCE_SIZE = 32
KEY_SIZE = 32
SESSION_NONCE_PREFIX_SIZE = 4
MAX_MESSAGE_BYTES = 64 * 1024 * 1024

AUTH_LABEL = b"imagineit-auth-v1\x00"
AUTH_OK_LABEL = b"imagineit-auth-ok-v1\x00"
HKDF_SALT_LABEL = b"imagineit-session-v1\x00"

Role = Literal["server", "client"]


class ProtocolError(Exception):
    """Raised when a handshake message or encrypted envelope is invalid."""


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64d(data: str) -> bytes:
    try:
        return base64.b64decode(data, validate=True)
    except Exception as exc:
        raise ProtocolError("invalid base64 payload") from exc


def new_nonce() -> bytes:
    return secrets.token_bytes(NONCE_SIZE)


def _session_salt(client_nonce: bytes, server_nonce: bytes, session_id: str) -> bytes:
    return hashlib.sha256(
        HKDF_SALT_LABEL + client_nonce + server_nonce + session_id.encode("utf-8")
    ).digest()


def _derive_half(
    psk: bytes,
    client_nonce: bytes,
    server_nonce: bytes,
    session_id: str,
    direction: str,
) -> tuple[bytes, bytes]:
    okm = HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE + SESSION_NONCE_PREFIX_SIZE,
        salt=_session_salt(client_nonce, server_nonce, session_id),
        info=f"imagineit-v1:{direction}".encode("ascii"),
    ).derive(psk)
    return okm[:KEY_SIZE], okm[KEY_SIZE:]


def derive_session_keys(
    psk: bytes,
    client_nonce: bytes,
    server_nonce: bytes,
    session_id: str,
) -> tuple[tuple[bytes, bytes], tuple[bytes, bytes]]:
    """Return ``((c2s_key, c2s_prefix), (s2c_key, s2c_prefix))``."""
    if len(psk) != KEY_SIZE:
        raise ProtocolError("pre-shared key must be 32 bytes")
    if len(client_nonce) != NONCE_SIZE or len(server_nonce) != NONCE_SIZE:
        raise ProtocolError("handshake nonces must be 32 bytes")
    return (
        _derive_half(psk, client_nonce, server_nonce, session_id, "c2s"),
        _derive_half(psk, client_nonce, server_nonce, session_id, "s2c"),
    )


def _transcript(
    worker_id: str,
    key_id: str,
    client_nonce: bytes,
    server_nonce: bytes,
    session_id: str,
) -> bytes:
    # Canonical JSON avoids separator ambiguity between identifiers and nonces.
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "worker_id": worker_id,
            "key_id": key_id,
            "client_nonce": b64e(client_nonce),
            "server_nonce": b64e(server_nonce),
            "session_id": session_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def make_client_proof(
    psk: bytes,
    worker_id: str,
    key_id: str,
    client_nonce: bytes,
    server_nonce: bytes,
    session_id: str,
) -> bytes:
    return hmac.new(
        psk,
        AUTH_LABEL
        + _transcript(worker_id, key_id, client_nonce, server_nonce, session_id),
        hashlib.sha256,
    ).digest()


def make_server_proof(
    psk: bytes,
    worker_id: str,
    key_id: str,
    client_nonce: bytes,
    server_nonce: bytes,
    session_id: str,
) -> bytes:
    return hmac.new(
        psk,
        AUTH_OK_LABEL
        + _transcript(worker_id, key_id, client_nonce, server_nonce, session_id),
        hashlib.sha256,
    ).digest()


def verify_proof(expected: bytes, provided: bytes) -> bool:
    return hmac.compare_digest(expected, provided)


def _aad(session_id: str, seq: int, msg_type: str) -> bytes:
    return f"{PROTOCOL_VERSION}|{session_id}|{seq}|{msg_type}".encode()


def _nonce(prefix: bytes, seq: int) -> bytes:
    return prefix + seq.to_bytes(8, "big")


@dataclass
class SecureChannel:
    """One direction-paired AES-GCM channel.

    Both peers construct the same object with ``role="server"`` or
    ``role="client"``.  Sending and receiving sequences are tracked separately,
    so both directions may start at zero.
    """

    session_id: str
    send_key: bytes
    send_prefix: bytes
    recv_key: bytes
    recv_prefix: bytes
    send_seq: int = 0
    recv_seq: int = 0
    _send_cipher: AESGCM = field(init=False, repr=False)
    _recv_cipher: AESGCM = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.send_key) != KEY_SIZE or len(self.recv_key) != KEY_SIZE:
            raise ProtocolError("session keys must be 32 bytes")
        if len(self.send_prefix) != SESSION_NONCE_PREFIX_SIZE:
            raise ProtocolError("invalid send nonce prefix")
        if len(self.recv_prefix) != SESSION_NONCE_PREFIX_SIZE:
            raise ProtocolError("invalid receive nonce prefix")
        self._send_cipher = AESGCM(self.send_key)
        self._recv_cipher = AESGCM(self.recv_key)

    @classmethod
    def from_psk(
        cls,
        psk: bytes,
        client_nonce: bytes,
        server_nonce: bytes,
        session_id: str,
        role: Role,
    ) -> SecureChannel:
        (c2s_key, c2s_prefix), (s2c_key, s2c_prefix) = derive_session_keys(
            psk, client_nonce, server_nonce, session_id
        )
        if role == "server":
            send_key, send_prefix = s2c_key, s2c_prefix
            recv_key, recv_prefix = c2s_key, c2s_prefix
        else:
            send_key, send_prefix = c2s_key, c2s_prefix
            recv_key, recv_prefix = s2c_key, s2c_prefix
        return cls(
            session_id=session_id,
            send_key=send_key,
            send_prefix=send_prefix,
            recv_key=recv_key,
            recv_prefix=recv_prefix,
        )

    def encrypt_json(self, msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.encrypt_bytes(
            msg_type,
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        )

    def encrypt_bytes(self, msg_type: str, payload: bytes) -> dict[str, Any]:
        if not msg_type or len(msg_type) > 64:
            raise ProtocolError("invalid message type")
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ProtocolError("message too large")
        seq = self.send_seq
        self.send_seq += 1
        nonce = _nonce(self.send_prefix, seq)
        ciphertext = self._send_cipher.encrypt(
            nonce, payload, _aad(self.session_id, seq, msg_type)
        )
        return {
            "v": PROTOCOL_VERSION,
            "sid": self.session_id,
            "seq": seq,
            "type": msg_type,
            "ct": b64e(ciphertext),
        }

    def decrypt_json(self, envelope: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        msg_type, payload = self.decrypt_bytes(envelope)
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise ProtocolError("invalid JSON payload") from exc
        if not isinstance(decoded, dict):
            raise ProtocolError("payload must be a JSON object")
        return msg_type, decoded

    def decrypt_bytes(self, envelope: dict[str, Any]) -> tuple[str, bytes]:
        if not isinstance(envelope, dict):
            raise ProtocolError("envelope must be an object")
        if envelope.get("v") != PROTOCOL_VERSION:
            raise ProtocolError("unsupported protocol version")
        if envelope.get("sid") != self.session_id:
            raise ProtocolError("session mismatch")
        msg_type = envelope.get("type")
        seq = envelope.get("seq")
        if not isinstance(msg_type, str) or not msg_type or len(msg_type) > 64:
            raise ProtocolError("invalid message type")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise ProtocolError("invalid sequence number")
        if seq != self.recv_seq:
            raise ProtocolError("unexpected sequence number")
        ciphertext = b64d(envelope.get("ct", ""))
        if len(ciphertext) < 16 or len(ciphertext) > MAX_MESSAGE_BYTES + 16:
            raise ProtocolError("invalid ciphertext size")
        try:
            plaintext = self._recv_cipher.decrypt(
                _nonce(self.recv_prefix, seq),
                ciphertext,
                _aad(self.session_id, seq, msg_type),
            )
        except Exception as exc:
            raise ProtocolError("message authentication failed") from exc
        self.recv_seq += 1
        return msg_type, plaintext
