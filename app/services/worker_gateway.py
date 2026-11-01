"""WebSocket gateway for outbound-only GPU workers.

Each GPU worker dials the app, authenticates with an HMAC challenge, and then
exchanges AES-GCM encrypted messages.  The gateway keeps the live connection
registry, receives progress/results, and records state in MariaDB.  The GPU
never listens on a port.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.database.minio import upload_image
from app.services import accounts, e2e, worker_keys
from app.services import jobs
from app.worker_auth import WorkerKeyStore
from shared.protocol import (
    NONCE_SIZE,
    PROTOCOL_VERSION,
    ProtocolError,
    SecureChannel,
    b64d,
    b64e,
    make_client_proof,
    make_server_proof,
    new_nonce,
    verify_proof,
)

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _valid_id(value: object) -> bool:
    return isinstance(value, str) and bool(_ID_RE.match(value))


@dataclass
class WorkerConnection:
    websocket: WebSocket
    worker_id: str
    key_id: str
    session_id: str
    channel: SecureChannel
    connected_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    name: str = ""
    model: str = ""
    device: str = ""
    kind: str = ""  # "kaggle" when the worker reports a Kaggle host
    account_id: str | None = None  # set for keys issued by /workers/keys
    capabilities: dict[str, Any] = field(default_factory=dict)
    status: str = "loading"  # loading | ready | dispatching | busy
    active_job_id: str | None = None
    active_attempt: int | None = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False

    @property
    def idle(self) -> bool:
        return self.status == "ready" and self.active_job_id is None and not self.closed

    async def send(self, msg_type: str, payload: dict[str, Any]) -> None:
        if self.closed or self.websocket.client_state != WebSocketState.CONNECTED:
            raise ConnectionError("worker connection is closed")
        # Encryption increments the sequence number, so it must happen under
        # the same lock as the socket write.
        async with self.send_lock:
            envelope = self.channel.encrypt_json(msg_type, payload)
            await self.websocket.send_json(envelope)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.close(code=code, reason=reason)
        except Exception:  # noqa: BLE001 - best effort during cleanup
            pass


class WorkerGateway:
    def __init__(self, app: FastAPI, key_store: WorkerKeyStore):
        self.app = app
        self.key_store = key_store
        self.heartbeat_interval = _env_int("WORKER_HEARTBEAT_INTERVAL", 15)
        self.lease_seconds = _env_int("WORKER_LEASE_SECONDS", 300)
        self.disconnect_grace = _env_int("WORKER_DISCONNECT_GRACE_SECONDS", 60)
        self.max_result_bytes = _env_int("MAX_RESULT_BYTES", 8 * 1024 * 1024)
        self._workers: dict[str, WorkerConnection] = {}
        self._lock = asyncio.Lock()
        self.wake = asyncio.Event()

    async def register(self, conn: WorkerConnection) -> None:
        async with self._lock:
            old = self._workers.get(conn.worker_id)
            self._workers[conn.worker_id] = conn
        if old is not None and old is not conn:
            logger.info("Replacing existing connection for worker %s", conn.worker_id)
            await old.close(4001, "replaced by a newer connection")
        self.wake.set()

    async def unregister(self, conn: WorkerConnection) -> None:
        async with self._lock:
            if self._workers.get(conn.worker_id) is conn:
                del self._workers[conn.worker_id]
        self.wake.set()

    async def snapshot(self) -> list[WorkerConnection]:
        async with self._lock:
            return list(self._workers.values())

    async def idle_workers(self) -> list[WorkerConnection]:
        return [conn for conn in await self.snapshot() if conn.idle]

    async def public_workers(self) -> list[dict[str, Any]]:
        """Safe worker metadata for the frontend.

        Keys, session IDs, and active job IDs stay private: the promise ID is
        enough to fetch an image from the public API.
        """
        now = time.monotonic()
        workers = []
        for conn in await self.snapshot():
            workers.append(
                {
                    "worker_id": conn.worker_id,
                    "name": conn.name,
                    "model": conn.model,
                    "device": conn.device,
                    "kind": conn.kind,
                    "account_id": conn.account_id,
                    "status": conn.status,
                    "connected_seconds": max(0, int(now - conn.connected_at)),
                    "last_seen_seconds_ago": max(0, int(now - conn.last_seen)),
                    "has_active_job": conn.active_job_id is not None,
                }
            )
        return sorted(workers, key=lambda item: item["worker_id"])

    async def shutdown_kaggle_workers(
        self, reason: str, account_id: str | None = None
    ) -> int:
        """Ask this account's Kaggle workers to exit.

        The worker turns this into os._exit, which ends the notebook and frees
        the user's Kaggle session.  Kaggle exposes no stop endpoint, so this is
        the only way to disconnect without a browser.
        """
        sent = 0
        for conn in await self.snapshot():
            if conn.closed or conn.kind != "kaggle":
                continue
            if account_id is not None and conn.account_id != account_id:
                continue
            try:
                await conn.send("shutdown", {"reason": reason})
                sent += 1
            except Exception:  # noqa: BLE001 - worker may vanish mid-send
                logger.exception("Could not send shutdown to %s", conn.worker_id)
        return sent

    async def close_all(self) -> None:
        for conn in await self.snapshot():
            await conn.close(1001, "app shutting down")

    # ------------------------------------------------------------------ #
    # Connection handling
    # ------------------------------------------------------------------ #
    async def handle_socket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        conn: WorkerConnection | None = None
        try:
            hello = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            conn = await self._handshake(websocket, hello)
            if conn is None:
                return
            await self.register(conn)
            await conn.send(
                "welcome",
                {
                    "protocol": PROTOCOL_VERSION,
                    "heartbeat_interval": self.heartbeat_interval,
                    "lease_seconds": self.lease_seconds,
                    "max_result_bytes": self.max_result_bytes,
                },
            )
            logger.info(
                "GPU worker %s connected (key=%s, session=%s)",
                conn.worker_id,
                conn.key_id,
                conn.session_id,
            )

            while True:
                raw = await websocket.receive_json()
                msg_type, payload = conn.channel.decrypt_json(raw)
                conn.last_seen = time.monotonic()
                await self._handle_message(conn, msg_type, payload)

        except WebSocketDisconnect:
            pass
        except TimeoutError:
            logger.warning("GPU worker handshake timed out")
            await _safe_close(websocket, 4408, "handshake timeout")
        except ProtocolError as exc:
            logger.warning("Protocol error from GPU worker: %s", exc)
            await _safe_close(websocket, 4400, "protocol error")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unhandled GPU worker error")
            await _safe_close(websocket, 1011, "internal error")
        finally:
            if conn is not None:
                await self.unregister(conn)
                if conn.active_job_id is not None:
                    if conn.active_attempt is not None:
                        try:
                            await jobs.shorten_lease(
                                conn.active_job_id,
                                conn.active_attempt,
                                conn.worker_id,
                                self.disconnect_grace,
                            )
                        except Exception:
                            logger.exception(
                                "Could not shorten lease for job %s",
                                conn.active_job_id,
                            )
                    logger.info(
                        "Worker %s disconnected with active job %s; reconnect grace is %ss",
                        conn.worker_id,
                        conn.active_job_id,
                        self.disconnect_grace,
                    )
                conn.closed = True

    async def _handshake(
        self, websocket: WebSocket, hello: object
    ) -> WorkerConnection | None:
        if not isinstance(hello, dict):
            await _safe_close(websocket, 4400, "hello must be an object")
            return None
        if hello.get("type") != "hello" or hello.get("v") != PROTOCOL_VERSION:
            await _safe_close(websocket, 4400, "expected hello v1")
            return None

        worker_id = hello.get("worker_id")
        key_id = hello.get("key_id")
        if not _valid_id(worker_id) or not _valid_id(key_id):
            await _safe_close(websocket, 4400, "invalid worker identity")
            return None

        try:
            client_nonce = b64d(hello.get("client_nonce", ""))
        except ProtocolError:
            await _safe_close(websocket, 4400, "invalid client nonce")
            return None
        if len(client_nonce) != NONCE_SIZE:
            await _safe_close(websocket, 4400, "invalid client nonce")
            return None

        psk = self.key_store.get(worker_id, key_id)
        account_id: str | None = None
        if psk is None:
            record = await worker_keys.lookup(worker_id, key_id)
            if record is not None:
                try:
                    psk = b64d(record.key_b64)
                except ProtocolError:
                    psk = None
                else:
                    account_id = record.account_id
        if psk is None:
            logger.warning("Rejected unknown worker key %s/%s", worker_id, key_id)
            await _safe_close(websocket, 4401, "unknown worker or key")
            return None

        server_nonce = new_nonce()
        session_id = secrets.token_hex(16)
        await websocket.send_json(
            {
                "type": "challenge",
                "v": PROTOCOL_VERSION,
                "server_nonce": b64e(server_nonce),
                "session_id": session_id,
            }
        )

        auth = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        if not isinstance(auth, dict) or auth.get("type") != "auth":
            await _safe_close(websocket, 4400, "expected auth")
            return None
        try:
            provided = b64d(auth.get("proof", ""))
        except ProtocolError:
            await _safe_close(websocket, 4400, "invalid auth proof")
            return None

        expected = make_client_proof(
            psk, worker_id, key_id, client_nonce, server_nonce, session_id
        )
        if not verify_proof(expected, provided):
            logger.warning("Rejected bad auth proof from %s/%s", worker_id, key_id)
            await _safe_close(websocket, 4401, "authentication failed")
            return None

        await websocket.send_json(
            {
                "type": "auth_ok",
                "v": PROTOCOL_VERSION,
                "proof": b64e(
                    make_server_proof(
                        psk,
                        worker_id,
                        key_id,
                        client_nonce,
                        server_nonce,
                        session_id,
                    )
                ),
            }
        )

        channel = SecureChannel.from_psk(
            psk, client_nonce, server_nonce, session_id, role="server"
        )
        return WorkerConnection(
            websocket=websocket,
            worker_id=worker_id,
            key_id=key_id,
            session_id=session_id,
            channel=channel,
            account_id=account_id,
        )

    # ------------------------------------------------------------------ #
    # Message handling
    # ------------------------------------------------------------------ #
    async def _handle_message(
        self, conn: WorkerConnection, msg_type: str, payload: dict[str, Any]
    ) -> None:
        if msg_type == "ready":
            await self._on_ready(conn, payload)
        elif msg_type == "heartbeat":
            await self._on_heartbeat(conn, payload)
        elif msg_type == "job_accepted":
            await self._on_job_accepted(conn, payload)
        elif msg_type == "job_rejected":
            await self._on_job_rejected(conn, payload)
        elif msg_type == "progress":
            await self._on_progress(conn, payload)
        elif msg_type == "result":
            await self._on_result(conn, payload)
        elif msg_type == "error":
            await self._on_error(conn, payload)
        elif msg_type == "cancelled":
            await self._on_cancelled(conn, payload)
        else:
            logger.warning("Unknown message type %r from %s", msg_type, conn.worker_id)
            raise ProtocolError(f"unknown message type {msg_type!r}")

    def _active_from(self, payload: dict[str, Any]) -> tuple[str, int] | None:
        active = payload.get("active_job")
        if active is None:
            return None
        if not isinstance(active, dict):
            raise ProtocolError("active_job must be an object")
        job_id = active.get("job_id")
        attempt = active.get("attempt")
        if not _valid_id(job_id) or not isinstance(attempt, int) or attempt <= 0:
            raise ProtocolError("invalid active_job")
        return job_id, attempt

    async def _on_ready(self, conn: WorkerConnection, payload: dict[str, Any]) -> None:
        conn.name = str(payload.get("name", ""))[:200]
        conn.model = str(payload.get("model", ""))[:200]
        conn.device = str(payload.get("device", ""))[:64]
        conn.kind = str(payload.get("kind", ""))[:32]
        caps = payload.get("capabilities")
        conn.capabilities = caps if isinstance(caps, dict) else {}
        active = self._active_from(payload)

        if active is not None:
            job_id, attempt = active
            resumed = await jobs.resume_job(
                job_id, attempt, conn.worker_id, self.lease_seconds
            )
            if resumed:
                conn.active_job_id = job_id
                conn.active_attempt = attempt
                conn.status = "busy"
                logger.info(
                    "Worker %s resumed job %s (attempt %s)",
                    conn.worker_id,
                    job_id,
                    attempt,
                )
            else:
                # Keep the worker reserved until it confirms cancellation.  The
                # old job may still be using the GPU, so dispatching a new one
                # immediately would make the worker reject it and churn.
                conn.active_job_id = job_id
                conn.active_attempt = attempt
                conn.status = "busy"
                await conn.send(
                    "cancel",
                    {
                        "job_id": job_id,
                        "attempt": attempt,
                        "reason": "job is no longer assigned to this worker",
                    },
                )
        else:
            conn.active_job_id = None
            conn.active_attempt = None
            conn.status = "ready"
        self.wake.set()

    async def _on_heartbeat(
        self, conn: WorkerConnection, payload: dict[str, Any]
    ) -> None:
        active = self._active_from(payload)
        if active is not None:
            job_id, attempt = active
            if conn.active_job_id != job_id or conn.active_attempt != attempt:
                conn.active_job_id = job_id
                conn.active_attempt = attempt
            conn.status = "busy"
            still_active = await jobs.extend_lease(
                job_id, attempt, conn.worker_id, self.lease_seconds
            )
            if not still_active:
                await conn.send(
                    "cancel",
                    {
                        "job_id": job_id,
                        "attempt": attempt,
                        "reason": "job is no longer assigned to this worker",
                    },
                )
        elif conn.status == "busy":
            conn.active_job_id = None
            conn.active_attempt = None
            conn.status = "ready"
        await conn.send("heartbeat_ack", {})
        self.wake.set()

    async def _on_job_accepted(
        self, conn: WorkerConnection, payload: dict[str, Any]
    ) -> None:
        job_id, attempt = self._job_ref(payload)
        if not await jobs.mark_running(
            job_id, attempt, conn.worker_id, self.lease_seconds
        ):
            await conn.send(
                "cancel",
                {
                    "job_id": job_id,
                    "attempt": attempt,
                    "reason": "assignment is stale",
                },
            )
            # Wait for the worker's cancelled/error message before making it
            # available for another job.
            conn.active_job_id = job_id
            conn.active_attempt = attempt
            conn.status = "busy"
            self.wake.set()
            return
        conn.active_job_id = job_id
        conn.active_attempt = attempt
        conn.status = "busy"

    async def _on_job_rejected(
        self, conn: WorkerConnection, payload: dict[str, Any]
    ) -> None:
        job_id, attempt = self._job_ref(payload)
        reason = str(payload.get("reason", "worker rejected job"))[:500]
        await jobs.requeue_job(job_id, attempt, conn.worker_id, reason)
        conn.active_job_id = None
        conn.active_attempt = None
        conn.status = "ready"
        self.wake.set()

    async def _on_progress(
        self, conn: WorkerConnection, payload: dict[str, Any]
    ) -> None:
        job_id, attempt = self._job_ref(payload)
        step = payload.get("step")
        total = payload.get("total")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or not isinstance(total, int)
            or isinstance(total, bool)
        ):
            raise ProtocolError("invalid progress counters")
        updated = await jobs.update_progress(
            job_id, attempt, conn.worker_id, step, total, self.lease_seconds
        )
        if not updated:
            await conn.send(
                "cancel",
                {
                    "job_id": job_id,
                    "attempt": attempt,
                    "reason": "job assignment was lost",
                },
            )
            conn.active_job_id = job_id
            conn.active_attempt = attempt
            conn.status = "busy"
            self.wake.set()

    async def _reject_result(
        self,
        conn: WorkerConnection,
        job_id: str,
        attempt: int,
        error: str,
    ) -> None:
        await conn.send(
            "result_nack",
            {
                "job_id": job_id,
                "attempt": attempt,
                "error": error,
                "retry": False,
            },
        )
        await jobs.fail_job(job_id, attempt, conn.worker_id, error)
        conn.active_job_id = None
        conn.active_attempt = None
        conn.status = "ready"
        self.wake.set()

    async def _on_result(self, conn: WorkerConnection, payload: dict[str, Any]) -> None:
        job_id, attempt = self._job_ref(payload)
        encoded = payload.get("image_b64")
        if not isinstance(encoded, str) or not encoded:
            await self._reject_result(
                conn, job_id, attempt, "result is missing image data"
            )
            return
        if len(encoded) > ((self.max_result_bytes + 2) // 3) * 4 + 8:
            await self._reject_result(conn, job_id, attempt, "result too large")
            return
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except Exception:
            await self._reject_result(conn, job_id, attempt, "invalid base64 image")
            return
        if not image_bytes or len(image_bytes) > self.max_result_bytes:
            await self._reject_result(conn, job_id, attempt, "invalid image size")
            return
        if not image_bytes.startswith(b"\xff\xd8"):
            await self._reject_result(conn, job_id, attempt, "expected JPEG data")
            return

        assignment = await jobs.get_assignment(job_id, attempt, conn.worker_id)
        if assignment is None:
            await conn.send(
                "result_nack",
                {
                    "job_id": job_id,
                    "attempt": attempt,
                    "error": "stale assignment",
                    "retry": False,
                },
            )
            conn.active_job_id = None
            conn.active_attempt = None
            conn.status = "ready"
            self.wake.set()
            return

        public_key = await accounts.public_key_for(assignment.get("account_id"))
        if not public_key:
            await self._reject_result(conn, job_id, attempt, "unknown account")
            return

        try:
            sealed = e2e.seal(public_key, image_bytes)
        except Exception:
            logger.exception("Failed to seal result for %s", job_id)
            await self._reject_result(conn, job_id, attempt, "encryption failed")
            return

        try:
            await upload_image(self.app, io.BytesIO(sealed), job_id)
        except Exception:
            logger.exception("Failed to store result for %s", job_id)
            await conn.send(
                "result_nack",
                {
                    "job_id": job_id,
                    "attempt": attempt,
                    "error": "storage unavailable",
                    "retry": True,
                },
            )
            return

        total = assignment.get("steps_total") or assignment.get("steps_completed") or 0
        total = int(total) if isinstance(total, int) else 0
        completed = await jobs.complete_job(job_id, attempt, conn.worker_id, total)
        if not completed:
            await conn.send(
                "result_nack",
                {
                    "job_id": job_id,
                    "attempt": attempt,
                    "error": "stale assignment",
                    "retry": False,
                },
            )
        else:
            await conn.send("result_ack", {"job_id": job_id, "attempt": attempt})
            logger.info(
                "Job %s completed by worker %s (attempt %s)",
                job_id,
                conn.worker_id,
                attempt,
            )
        conn.active_job_id = None
        conn.active_attempt = None
        conn.status = "ready"
        self.wake.set()

    async def _on_error(self, conn: WorkerConnection, payload: dict[str, Any]) -> None:
        job_id, attempt = self._job_ref(payload)
        error = str(payload.get("error", "inference failed"))[:4000]
        await jobs.fail_job(job_id, attempt, conn.worker_id, error)
        conn.active_job_id = None
        conn.active_attempt = None
        conn.status = "ready"
        await conn.send("error_ack", {"job_id": job_id, "attempt": attempt})
        self.wake.set()

    async def _on_cancelled(
        self, conn: WorkerConnection, payload: dict[str, Any]
    ) -> None:
        job_id, attempt = self._job_ref(payload)
        await jobs.cancel_job(
            job_id, attempt, conn.worker_id, str(payload.get("reason", "cancelled"))
        )
        conn.active_job_id = None
        conn.active_attempt = None
        conn.status = "ready"
        self.wake.set()

    @staticmethod
    def _job_ref(payload: dict[str, Any]) -> tuple[str, int]:
        job_id = payload.get("job_id")
        attempt = payload.get("attempt")
        if not _valid_id(job_id) or not isinstance(attempt, int) or attempt <= 0:
            raise ProtocolError("invalid job reference")
        return job_id, attempt


async def _safe_close(websocket: WebSocket, code: int, reason: str) -> None:
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=code, reason=reason)
    except Exception:  # noqa: BLE001
        pass
