"""Client-held encryption for everything the app persists about a user.

The browser generates a P-256 keypair at signup and keeps the private half in
IndexedDB.  The server stores only the public key.  Job parameters and
generated images are sealed to that public key with an ephemeral ECDH key,
HKDF-SHA256 and AES-256-GCM, so the server can encrypt but never decrypt.

Envelope layout (all binary, then base64 where it has to be text):

    b"E2E1" | ephemeral public key (65 bytes, uncompressed P-256)
            | nonce (12 bytes) | AES-GCM ciphertext

Kaggle credentials cannot use this scheme: browserful signs in on the user's
behalf, so the server must be able to read them.  Those are encrypted with a
key derived from ``APP_SECRET`` instead (a different trust domain).
"""

from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

MAGIC = b"E2E1"
INFO = b"imagineit-e2e-v1"
NONCE_LEN = 12
PUBLIC_KEY_LEN = 65

_SERVER_SALT = b"imagineit-app-secret"
_SERVER_INFO = b"kaggle-credentials-v1"


class EnvelopeError(ValueError):
    """Raised when a sealed blob is malformed or cannot be opened."""


def _hkdf(salt: bytes, info: bytes) -> HKDF:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=info)


def load_public_key(encoded: str) -> ec.EllipticCurvePublicKey:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:  # noqa: BLE001 - surface as a validation error
        raise ValueError("public key is not valid base64") from exc
    if len(raw) != PUBLIC_KEY_LEN or raw[0] != 0x04:
        raise ValueError("public key must be a 65-byte uncompressed P-256 point")
    try:
        return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), raw)
    except ValueError as exc:
        raise ValueError("public key is not a valid P-256 point") from exc


def seal(public_key: str, plaintext: bytes) -> bytes:
    """Encrypt ``plaintext`` so only the holder of the private key can open it."""
    recipient = load_public_key(public_key)
    ephemeral = ec.generate_private_key(ec.SECP256R1())
    ephemeral_raw = ephemeral.public_key().public_bytes(
        Encoding.X962, PublicFormat.UncompressedPoint
    )
    shared = ephemeral.exchange(ec.ECDH(), recipient)
    key = _hkdf(ephemeral_raw, INFO).derive(shared)
    nonce = os.urandom(NONCE_LEN)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return MAGIC + ephemeral_raw + nonce + ciphertext


def seal_text(public_key: str, text: str) -> str:
    return base64.b64encode(seal(public_key, text.encode("utf-8"))).decode("ascii")


def open_with_private(
    private_key: ec.EllipticCurvePrivateKey, envelope: bytes
) -> bytes:
    """Open an envelope locally.  Only used by tests: the server has no key."""
    if len(envelope) < len(MAGIC) + PUBLIC_KEY_LEN + NONCE_LEN + 16:
        raise EnvelopeError("sealed blob is too short")
    if not envelope.startswith(MAGIC):
        raise EnvelopeError("unknown envelope version")
    offset = len(MAGIC)
    ephemeral_raw = envelope[offset : offset + PUBLIC_KEY_LEN]
    offset += PUBLIC_KEY_LEN
    nonce = envelope[offset : offset + NONCE_LEN]
    offset += NONCE_LEN
    ciphertext = envelope[offset:]

    ephemeral = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(), ephemeral_raw
    )
    shared = private_key.exchange(ec.ECDH(), ephemeral)
    key = _hkdf(ephemeral_raw, INFO).derive(shared)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, None)
    except InvalidTag as exc:
        raise EnvelopeError("sealed blob failed authentication") from exc


def _server_key() -> bytes:
    secret = os.getenv("APP_SECRET", "").strip()
    if len(secret) < 32:
        raise RuntimeError("APP_SECRET must be at least 32 characters")
    return _hkdf(_SERVER_SALT, _SERVER_INFO).derive(secret.encode("utf-8"))


def encrypt_server_secret(plaintext: bytes) -> str:
    nonce = os.urandom(NONCE_LEN)
    ciphertext = AESGCM(_server_key()).encrypt(nonce, plaintext, None)
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_server_secret(blob: str) -> bytes:
    try:
        raw = base64.b64decode(blob, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise EnvelopeError("server secret is not valid base64") from exc
    if len(raw) < NONCE_LEN + 16:
        raise EnvelopeError("server secret is too short")
    try:
        return AESGCM(_server_key()).decrypt(raw[:NONCE_LEN], raw[NONCE_LEN:], None)
    except InvalidTag as exc:
        raise EnvelopeError("server secret failed authentication") from exc
