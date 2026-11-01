"""Per-account GPU worker keys.

The server keeps the PSK encrypted (the handshake needs the raw bytes), and
the user receives it once as a ready-to-use JSON file.  A worker authenticated
with one of these keys is bound to that account and only receives its jobs.
"""

from __future__ import annotations

import base64
import json
import os
import secrets

from sqlalchemy import delete, select

from app.database.db import db_session
from app.database.models import WorkerKey
from app.services import e2e


class KeyRecord:
    __slots__ = ("account_id", "key_b64")

    def __init__(self, account_id: str, key_b64: str) -> None:
        self.account_id = account_id
        self.key_b64 = key_b64


def _new_key() -> str:
    return base64.b64encode(os.urandom(32)).decode("ascii")


async def create(account_id: str) -> dict[str, str]:
    worker_id = f"gpu-{account_id[:6]}-{secrets.token_hex(2)}"
    key_id = "k1"
    raw = _new_key()
    secret = e2e.encrypt_server_secret(json.dumps({"key": raw}).encode("utf-8"))
    async with db_session() as session:
        session.add(
            WorkerKey(
                account_id=account_id,
                worker_id=worker_id,
                key_id=key_id,
                secret=secret,
            )
        )
        await session.commit()
    return {"worker_id": worker_id, "key_id": key_id, "key": raw}


async def lookup(worker_id: str, key_id: str) -> KeyRecord | None:
    async with db_session() as session:
        row = await session.scalar(
            select(WorkerKey).where(
                WorkerKey.worker_id == worker_id, WorkerKey.key_id == key_id
            )
        )
        if row is None:
            # The worker spawns one child per GPU as "<base>-1", "<base>-2".
            base, _, suffix = worker_id.rpartition("-")
            if base and suffix.isdigit():
                row = await session.scalar(
                    select(WorkerKey).where(
                        WorkerKey.worker_id == base, WorkerKey.key_id == key_id
                    )
                )
    if row is None:
        return None
    try:
        payload = json.loads(e2e.decrypt_server_secret(row.secret))
    except (e2e.EnvelopeError, json.JSONDecodeError):
        return None
    key = payload.get("key")
    if not isinstance(key, str) or not key:
        return None
    return KeyRecord(row.account_id, key)


async def ensure_for(
    account_id: str, busy_worker_ids: set[str] | None = None
) -> dict[str, str]:
    """Return a key that is not already connected, creating one if needed.

    A worker id must be unique among live connections, so reusing the newest
    key for Kaggle would evict a local worker that is already using it.
    """
    busy = busy_worker_ids or set()
    async with db_session() as session:
        row = await session.scalar(
            select(WorkerKey)
            .where(WorkerKey.account_id == account_id)
            .order_by(WorkerKey.created_at.desc(), WorkerKey.id.desc())
        )
    if row is None:
        return await create(account_id)
    base = row.worker_id
    if any(wid == base or wid.startswith(base + "-") for wid in busy):
        return await create(account_id)
    try:
        payload = json.loads(e2e.decrypt_server_secret(row.secret))
    except (e2e.EnvelopeError, json.JSONDecodeError):
        return await create(account_id)
    key = payload.get("key")
    if not isinstance(key, str) or not key:
        return await create(account_id)
    return {"worker_id": row.worker_id, "key_id": row.key_id, "key": key}


async def list_for(account_id: str) -> list[dict[str, str]]:
    async with db_session() as session:
        rows = (
            await session.scalars(
                select(WorkerKey)
                .where(WorkerKey.account_id == account_id)
                .order_by(WorkerKey.created_at.desc())
            )
        ).all()
    return [
        {
            "worker_id": row.worker_id,
            "key_id": row.key_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]


async def revoke(account_id: str, worker_id: str) -> bool:
    async with db_session() as session:
        result = await session.execute(
            delete(WorkerKey).where(
                WorkerKey.account_id == account_id,
                WorkerKey.worker_id == worker_id,
            )
        )
        await session.commit()
    return bool(result.rowcount)
