import base64
import os

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from app.services import e2e, tokens


def _keypair() -> tuple[ec.EllipticCurvePrivateKey, str]:
    private = ec.generate_private_key(ec.SECP256R1())
    raw = private.public_key().public_bytes(
        Encoding.X962, PublicFormat.UncompressedPoint
    )
    return private, base64.b64encode(raw).decode("ascii")


def test_seal_roundtrip() -> None:
    private, public = _keypair()
    sealed = e2e.seal(public, b"a cat in a hat")
    assert sealed.startswith(e2e.MAGIC)
    assert e2e.open_with_private(private, sealed) == b"a cat in a hat"


def test_seal_text_roundtrip() -> None:
    private, public = _keypair()
    sealed = e2e.seal_text(public, '{"prompt": "neko"}')
    raw = base64.b64decode(sealed)
    assert e2e.open_with_private(private, raw).decode() == '{"prompt": "neko"}'


def test_wrong_key_rejected() -> None:
    _, public = _keypair()
    other, _ = _keypair()
    sealed = e2e.seal(public, b"secret")
    try:
        e2e.open_with_private(other, sealed)
    except e2e.EnvelopeError:
        return
    raise AssertionError("opened with the wrong private key")


def test_tamper_rejected() -> None:
    private, public = _keypair()
    sealed = bytearray(e2e.seal(public, b"secret"))
    sealed[-1] ^= 0x01
    try:
        e2e.open_with_private(private, bytes(sealed))
    except e2e.EnvelopeError:
        return
    raise AssertionError("tampered envelope was accepted")


def test_invalid_public_key_rejected() -> None:
    for value in ["", "not-base64", base64.b64encode(b"\x04" + b"\x00" * 63).decode()]:
        try:
            e2e.load_public_key(value)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid key {value!r}")


def test_server_secret_roundtrip() -> None:
    previous = os.environ.get("APP_SECRET")
    os.environ["APP_SECRET"] = "x" * 40
    try:
        blob = e2e.encrypt_server_secret(b"email+password")
        assert b"email" not in base64.b64decode(blob)
        assert e2e.decrypt_server_secret(blob) == b"email+password"
    finally:
        if previous is None:
            os.environ.pop("APP_SECRET", None)
        else:
            os.environ["APP_SECRET"] = previous


def test_server_secret_requires_app_secret() -> None:
    previous = os.environ.pop("APP_SECRET", None)
    try:
        try:
            e2e.encrypt_server_secret(b"x")
        except RuntimeError:
            return
        raise AssertionError("encrypted without APP_SECRET")
    finally:
        if previous is not None:
            os.environ["APP_SECRET"] = previous


def test_token_shape() -> None:
    token = tokens.new_token()
    assert token.startswith(tokens.TOKEN_PREFIX)
    assert tokens.hash_token(token) == tokens.hash_token(token)
    assert len(tokens.hash_token(token)) == 64
    assert tokens.hash_token(token) != tokens.hash_token(tokens.new_token())
