"""Account creation, bearer-token lookup, and per-user Kaggle secrets."""

from __future__ import annotations

import json
import uuid

from sqlalchemy import delete, select

from app.database.db import db_session
from app.database.models import Account, KaggleCredential
from app.services import e2e
from app.services.tokens import hash_token, new_token


async def create_account(public_key: str) -> tuple[Account, str]:
    # Raises ValueError for anything that is not a usable P-256 point.
    e2e.load_public_key(public_key)

    token = new_token()
    account = Account(
        id=uuid.uuid4().hex,
        token_hash=hash_token(token),
        public_key=public_key,
    )
    async with db_session() as session:
        session.add(account)
        await session.commit()
    return account, token


async def account_for_token(token: str) -> Account | None:
    if not token:
        return None
    async with db_session() as session:
        return await session.scalar(
            select(Account).where(Account.token_hash == hash_token(token))
        )


async def get_account(account_id: str) -> Account | None:
    async with db_session() as session:
        return await session.scalar(select(Account).where(Account.id == account_id))


async def public_key_for(account_id: str | None) -> str | None:
    if not account_id:
        return None
    account = await get_account(account_id)
    return account.public_key if account else None


# --------------------------------------------------------------------------- #
# Kaggle credentials (server-encrypted, never returned to the client)
# --------------------------------------------------------------------------- #


async def save_kaggle_credentials(account_id: str, email: str, password: str) -> None:
    payload = json.dumps({"email": email, "password": password}).encode("utf-8")
    secret = e2e.encrypt_server_secret(payload)
    async with db_session() as session:
        row = await session.get(KaggleCredential, account_id)
        if row is None:
            session.add(KaggleCredential(account_id=account_id, secret=secret))
        else:
            row.secret = secret
        await session.commit()


async def load_kaggle_credentials(account_id: str) -> tuple[str, str] | None:
    async with db_session() as session:
        row = await session.get(KaggleCredential, account_id)
    if row is None:
        return None
    try:
        payload = json.loads(e2e.decrypt_server_secret(row.secret))
    except (e2e.EnvelopeError, json.JSONDecodeError):
        return None
    email = payload.get("email")
    password = payload.get("password")
    if not isinstance(email, str) or not isinstance(password, str):
        return None
    return email, password


async def delete_kaggle_credentials(account_id: str) -> None:
    async with db_session() as session:
        await session.execute(
            delete(KaggleCredential).where(KaggleCredential.account_id == account_id)
        )
        await session.commit()


async def has_kaggle_credentials(account_id: str) -> bool:
    async with db_session() as session:
        row = await session.get(KaggleCredential, account_id)
    return row is not None
