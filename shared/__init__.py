"""Shared protocol and cryptography used by the app and GPU worker."""

from .protocol import (
    ProtocolError,
    SecureChannel,
    b64d,
    b64e,
    derive_session_keys,
    make_client_proof,
    make_server_proof,
    new_nonce,
    verify_proof,
)

__all__ = [
    "ProtocolError",
    "SecureChannel",
    "b64d",
    "b64e",
    "derive_session_keys",
    "make_client_proof",
    "make_server_proof",
    "new_nonce",
    "verify_proof",
]
