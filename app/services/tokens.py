"""Bearer tokens that exist only to sign a client in.

The raw token is returned once at signup and never stored: the database keeps
its SHA-256 hash.  It is deliberately not an encryption key — the client key
lives in the browser and never reaches this process.
"""

from __future__ import annotations

import hashlib
import secrets

TOKEN_PREFIX = "imi_"


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
