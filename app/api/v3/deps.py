"""Shared FastAPI dependencies for the public v3 API."""

from __future__ import annotations

from fastapi import Header, HTTPException

from app.database.models import Account
from app.services import accounts


def bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return value.strip()


async def require_account(
    authorization: str | None = Header(default=None),
) -> Account:
    token = bearer_token(authorization)
    account = await accounts.account_for_token(token) if token else None
    if account is None:
        raise HTTPException(
            status_code=401,
            detail="invalid or missing account token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return account
