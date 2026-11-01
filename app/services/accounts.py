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


async def set_public_key(
    account_id: str, public_key: str, wrapped_key: str | None = None
) -> None:
    e2e.load_public_key(public_key)
    async with db_session() as session:
        account = await session.get(Account, account_id)
        if account is None:
            raise ValueError("unknown account")
        account.public_key = public_key
        if wrapped_key is not None:
            account.wrapped_key = wrapped_key
        await session.commit()


async def set_wrapped_key(account_id: str, wrapped_key: str) -> None:
    async with db_session() as session:
        account = await session.get(Account, account_id)
        if account is None:
            raise ValueError("unknown account")
        account.wrapped_key = wrapped_key
        await session.commit()


async def get_wrapped_key(account_id: str) -> str | None:
    async with db_session() as session:
        account = await session.get(Account, account_id)
    return account.wrapped_key if account else None


async def get_account(account_id: str) -> Account | None:
    async with db_session() as session:
        return await session.scalar(select(Account).where(Account.id == account_id))


async def public_key_for(account_id: str | None) -> str | None:
    if not account_id:
        return None
    account = await get_account(account_id)
    return account.public_key if account else None


# --------------------------------------------------------------------------- #
# Kaggle API key (server-encrypted, never returned to the client)
# --------------------------------------------------------------------------- #


async def save_kaggle_api_key(
    account_id: str,
    api_key: str,
    kernel_ref: str | None = None,
    username: str | None = None,
) -> None:
    payload = json.dumps(
        {
            "api_key": api_key.strip(),
            "kernel_ref": (kernel_ref or "").strip(),
            "username": (username or "").strip(),
        }
    ).encode("utf-8")
    secret = e2e.encrypt_server_secret(payload)
    async with db_session() as session:
        row = await session.get(KaggleCredential, account_id)
        if row is None:
            session.add(KaggleCredential(account_id=account_id, secret=secret))
        else:
            row.secret = secret
        await session.commit()


async def load_kaggle_api_key(account_id: str) -> str | None:
    async with db_session() as session:
        row = await session.get(KaggleCredential, account_id)
    if row is None:
        return None
    try:
        payload = json.loads(e2e.decrypt_server_secret(row.secret))
    except (e2e.EnvelopeError, json.JSONDecodeError):
        return None
    api_key = payload.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        return None
    return api_key.strip()


async def delete_kaggle_credentials(account_id: str) -> None:
    async with db_session() as session:
        await session.execute(
            delete(KaggleCredential).where(KaggleCredential.account_id == account_id)
        )
        await session.commit()


async def load_kaggle_kernel_ref(account_id: str) -> str | None:
    async with db_session() as session:
        row = await session.get(KaggleCredential, account_id)
    if row is None:
        return None
    try:
        payload = json.loads(e2e.decrypt_server_secret(row.secret))
    except (e2e.EnvelopeError, json.JSONDecodeError):
        return None
    ref = payload.get("kernel_ref")
    return ref.strip() if isinstance(ref, str) and ref.strip() else None


async def load_kaggle_username(account_id: str) -> str | None:
    async with db_session() as session:
        row = await session.get(KaggleCredential, account_id)
    if row is None:
        return None
    try:
        payload = json.loads(e2e.decrypt_server_secret(row.secret))
    except (e2e.EnvelopeError, json.JSONDecodeError):
        return None
    name = payload.get("username")
    return name.strip() if isinstance(name, str) and name.strip() else None


async def has_kaggle_credentials(account_id: str) -> bool:
    # A row from the old email/password flow decrypts without an api_key, so
    # treat only a loadable API key as "configured".
    return await load_kaggle_api_key(account_id) is not None
