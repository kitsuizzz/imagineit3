"""Database operations for the app-side inference job lifecycle.

MariaDB is the source of truth.  The dispatcher claims queued rows, the worker
gateway records progress and terminal state, and expired leases are returned to
the queue so a crashed GPU worker cannot strand a job forever.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update

from app.database.db import db_session
from app.database.models import Image

ACTIVE_STATUSES = ("dispatched", "running")
TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")


def utcnow() -> datetime:
    # MariaDB does not preserve timezone information, so store naive UTC.
    return datetime.now(UTC).replace(tzinfo=None)


def _params(row: Image) -> dict[str, Any]:
    if not row.params_json:
        return {}
    try:
        value = json.loads(row.params_json)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def job_to_dict(row: Image) -> dict[str, Any]:
    return {
        "promise_id": row.promise_id,
        "status": row.status,
        "steps_status": row.steps_status,
        "steps_completed": row.steps_completed,
        "steps_total": row.steps_total,
        "error": row.error,
        "worker_id": row.worker_id,
        "attempt": row.attempt,
        "params": _params(row),
    }


async def create_job(promise_id: str, params: dict[str, Any], steps: int) -> None:
    async with db_session() as session:
        session.add(
            Image(
                promise_id=promise_id,
                status="queued",
                steps_status=f"0/{steps}",
                steps_completed=0,
                steps_total=steps,
                params_json=json.dumps(params, separators=(",", ":")),
            )
        )
        await session.commit()


async def get_job(promise_id: str) -> dict[str, Any] | None:
    async with db_session() as session:
        row = await session.scalar(select(Image).where(Image.promise_id == promise_id))
        return job_to_dict(row) if row else None


async def claim_next_job(worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
    """Atomically move the oldest queued job to ``dispatched``."""
    async with db_session() as session:
        stmt = (
            select(Image)
            .where(Image.status == "queued")
            .order_by(Image.created_at, Image.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        row = (await session.scalars(stmt)).first()
        if not row:
            return None

        total = row.steps_total or 0
        row.status = "dispatched"
        row.worker_id = worker_id
        row.attempt = (row.attempt or 0) + 1
        row.lease_expires_at = utcnow() + timedelta(seconds=lease_seconds)
        row.steps_completed = 0
        row.steps_status = f"0/{total}"
        row.error = None
        await session.commit()
        return job_to_dict(row)


async def get_assignment(
    promise_id: str, attempt: int, worker_id: str
) -> dict[str, Any] | None:
    async with db_session() as session:
        row = await session.scalar(
            select(Image).where(
                Image.promise_id == promise_id,
                Image.attempt == attempt,
                Image.worker_id == worker_id,
                Image.status.in_(ACTIVE_STATUSES),
            )
        )
        return job_to_dict(row) if row else None


async def mark_running(
    promise_id: str, attempt: int, worker_id: str, lease_seconds: int
) -> bool:
    async with db_session() as session:
        result = await session.execute(
            update(Image)
            .where(
                Image.promise_id == promise_id,
                Image.attempt == attempt,
                Image.worker_id == worker_id,
                Image.status.in_(ACTIVE_STATUSES),
            )
            .values(
                status="running",
                lease_expires_at=utcnow() + timedelta(seconds=lease_seconds),
            )
        )
        await session.commit()
        return bool(result.rowcount)


async def update_progress(
    promise_id: str,
    attempt: int,
    worker_id: str,
    step: int,
    total: int,
    lease_seconds: int,
) -> bool:
    if step < 0 or total <= 0:
        return False
    step = min(step, total)
    async with db_session() as session:
        result = await session.execute(
            update(Image)
            .where(
                Image.promise_id == promise_id,
                Image.attempt == attempt,
                Image.worker_id == worker_id,
                Image.status.in_(ACTIVE_STATUSES),
            )
            .values(
                status="running",
                steps_completed=step,
                steps_total=total,
                steps_status=f"{step}/{total}",
                lease_expires_at=utcnow() + timedelta(seconds=lease_seconds),
            )
        )
        await session.commit()
        return bool(result.rowcount)


async def extend_lease(
    promise_id: str, attempt: int, worker_id: str, lease_seconds: int
) -> bool:
    async with db_session() as session:
        result = await session.execute(
            update(Image)
            .where(
                Image.promise_id == promise_id,
                Image.attempt == attempt,
                Image.worker_id == worker_id,
                Image.status.in_(ACTIVE_STATUSES),
            )
            .values(lease_expires_at=utcnow() + timedelta(seconds=lease_seconds))
        )
        await session.commit()
        return bool(result.rowcount)


async def complete_job(
    promise_id: str, attempt: int, worker_id: str, total: int
) -> bool:
    async with db_session() as session:
        result = await session.execute(
            update(Image)
            .where(
                Image.promise_id == promise_id,
                Image.attempt == attempt,
                Image.worker_id == worker_id,
                Image.status.in_(ACTIVE_STATUSES),
            )
            .values(
                status="succeeded",
                steps_completed=total,
                steps_total=total,
                steps_status=f"{total}/{total}",
                error=None,
                lease_expires_at=None,
            )
        )
        await session.commit()
        return bool(result.rowcount)


async def fail_job(promise_id: str, attempt: int, worker_id: str, error: str) -> bool:
    async with db_session() as session:
        result = await session.execute(
            update(Image)
            .where(
                Image.promise_id == promise_id,
                Image.attempt == attempt,
                Image.worker_id == worker_id,
                Image.status.in_(ACTIVE_STATUSES),
            )
            .values(
                status="failed",
                error=error[:4000],
                lease_expires_at=None,
            )
        )
        await session.commit()
        return bool(result.rowcount)


async def cancel_job(
    promise_id: str, attempt: int, worker_id: str, reason: str = "cancelled"
) -> bool:
    async with db_session() as session:
        result = await session.execute(
            update(Image)
            .where(
                Image.promise_id == promise_id,
                Image.attempt == attempt,
                Image.worker_id == worker_id,
                Image.status.in_(ACTIVE_STATUSES),
            )
            .values(status="cancelled", error=reason[:4000], lease_expires_at=None)
        )
        await session.commit()
        return bool(result.rowcount)


async def requeue_job(
    promise_id: str,
    attempt: int,
    worker_id: str,
    reason: str,
) -> bool:
    async with db_session() as session:
        result = await session.execute(
            update(Image)
            .where(
                Image.promise_id == promise_id,
                Image.attempt == attempt,
                Image.worker_id == worker_id,
                Image.status.in_(ACTIVE_STATUSES),
            )
            .values(
                status="queued",
                worker_id=None,
                lease_expires_at=None,
                error=reason[:4000],
            )
        )
        await session.commit()
        return bool(result.rowcount)


async def shorten_lease(
    promise_id: str, attempt: int, worker_id: str, grace_seconds: int
) -> bool:
    """Give a disconnected worker a short window to reconnect and resume."""
    async with db_session() as session:
        result = await session.execute(
            update(Image)
            .where(
                Image.promise_id == promise_id,
                Image.attempt == attempt,
                Image.worker_id == worker_id,
                Image.status.in_(ACTIVE_STATUSES),
            )
            .values(lease_expires_at=utcnow() + timedelta(seconds=grace_seconds))
        )
        await session.commit()
        return bool(result.rowcount)


async def resume_job(
    promise_id: str, attempt: int, worker_id: str, lease_seconds: int
) -> bool:
    """Keep a job on a worker that reconnected while still working on it."""
    async with db_session() as session:
        result = await session.execute(
            update(Image)
            .where(
                Image.promise_id == promise_id,
                Image.attempt == attempt,
                Image.worker_id == worker_id,
                Image.status.in_(ACTIVE_STATUSES),
            )
            .values(
                status="running",
                lease_expires_at=utcnow() + timedelta(seconds=lease_seconds),
            )
        )
        await session.commit()
        return bool(result.rowcount)


async def expire_leases() -> int:
    async with db_session() as session:
        result = await session.execute(
            update(Image)
            .where(
                Image.status.in_(ACTIVE_STATUSES),
                Image.lease_expires_at.is_not(None),
                Image.lease_expires_at < utcnow(),
            )
            .values(
                status="queued",
                worker_id=None,
                lease_expires_at=None,
                error="GPU worker lease expired",
            )
        )
        await session.commit()
        return max(result.rowcount or 0, 0)
