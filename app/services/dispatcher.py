"""Assigns queued database jobs to connected, idle GPU workers."""

from __future__ import annotations

import asyncio
import logging

from app.services import jobs
from app.services.worker_gateway import WorkerGateway

logger = logging.getLogger(__name__)


class JobDispatcher:
    def __init__(self, gateway: WorkerGateway):
        self.gateway = gateway
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="job-dispatcher")

    async def stop(self) -> None:
        self._stopping.set()
        self.gateway.wake.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Job dispatcher tick failed")
            try:
                await asyncio.wait_for(self.gateway.wake.wait(), timeout=1.0)
            except TimeoutError:
                pass
            self.gateway.wake.clear()

    async def _tick(self) -> None:
        expired = await jobs.expire_leases()
        if expired:
            logger.warning("Requeued %s job(s) with expired leases", expired)

        for conn in await self.gateway.idle_workers():
            if conn.closed:
                continue
            job = await jobs.claim_next_job(
                conn.worker_id, self.gateway.lease_seconds, conn.account_id
            )
            if job is None:
                break
            params = job.get("params")
            if not params:
                await jobs.fail_job(
                    job["promise_id"],
                    job["attempt"],
                    conn.worker_id,
                    "job parameters are missing",
                )
                continue

            conn.active_job_id = job["promise_id"]
            conn.active_attempt = job["attempt"]
            conn.status = "dispatching"
            try:
                await conn.send(
                    "job",
                    {
                        "job_id": job["promise_id"],
                        "attempt": job["attempt"],
                        "params": params,
                    },
                )
            except Exception:  # noqa: BLE001 - connection died while sending
                logger.warning(
                    "Failed to send job %s to worker %s",
                    job["promise_id"],
                    conn.worker_id,
                )
                conn.active_job_id = None
                conn.active_attempt = None
                conn.status = "ready" if not conn.closed else "offline"
                await jobs.requeue_job(
                    job["promise_id"],
                    job["attempt"],
                    conn.worker_id,
                    "failed to contact GPU worker",
                )
                continue

            logger.info(
                "Dispatched job %s to worker %s (attempt %s)",
                job["promise_id"],
                conn.worker_id,
                job["attempt"],
            )
