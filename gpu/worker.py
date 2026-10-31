"""Outbound-only GPU worker.

Run this on the GPU machine:

    python gpu_worker.py --key-file ~/.imagineit/worker.json

Use ``--gpus auto`` on a multi-GPU host to start one child process per GPU; each
child owns its own model weights.  A single worker opens one WebSocket to the
app backend, authenticates with a pre-shared key, loads a model, and then
receives encrypted inference jobs.  It never listens on a port.  If the
connection drops the current job keeps running and its result is retried after
reconnecting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import signal
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets

from gpu.errors import InferenceCancelled
from shared.protocol import (
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

logger = logging.getLogger("gpu_worker")

DEFAULT_APP_URL = "ws://localhost:4600/internal/v1/workers/ws"
DEFAULT_MODEL = "wssdaas/nova-anime-xl-ilv-19.0"
WS_MAX_SIZE = 96 * 1024 * 1024


@dataclass
class WorkerConfig:
    app_url: str
    worker_id: str
    key_id: str
    key: bytes
    model: str
    name: str
    ca_file: str | None = None
    reconnect_min: float = 1.0
    reconnect_max: float = 60.0


def _read_key_file(
    path: Path, worker_id: str | None, key_id: str | None
) -> tuple[str, str, bytes]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("key file must contain a JSON object")
    file_worker_id = raw.get("worker_id")
    file_key_id = raw.get("key_id", "default")
    file_key = raw.get("key")
    worker = worker_id or file_worker_id
    key = key_id or file_key_id
    if not isinstance(worker, str) or not worker:
        raise ValueError("worker_id is required in the key file or --worker-id")
    if not isinstance(key, str) or not key:
        raise ValueError("key_id must be a string")
    if not isinstance(file_key, str):
        raise ValueError("key file must contain a base64 'key'")
    decoded = b64d(file_key)
    if len(decoded) != 32:
        raise ValueError("worker key must be 32 bytes")
    return worker, key, decoded


def _build_ssl_context(url: str, ca_file: str | None) -> ssl.SSLContext | None:
    if not url.startswith("wss://"):
        return None
    context = ssl.create_default_context()
    if ca_file:
        context.load_verify_locations(ca_file)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


def _detect_gpu_count() -> int:
    """Return the number of usable accelerators, or one CPU worker."""
    if sys.platform == "darwin":
        # MLX exposes the Apple Silicon GPU as a single unified device.
        return 1
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return max(1, int(torch.cuda.device_count()))
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return max(1, int(torch.xpu.device_count()))
    except Exception:  # noqa: BLE001 - detection is best effort
        pass
    return 1


def _resolve_gpu_count(requested: str) -> int:
    value = str(requested).strip().lower()
    detected = _detect_gpu_count()
    if value in {"auto", "all"}:
        return detected
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError("--gpus must be auto, all, or a positive integer") from exc
    if count < 1:
        raise ValueError("--gpus must be at least 1")
    if count > detected:
        logger.warning(
            "Requested %s GPUs but detected %s; using %s",
            count,
            detected,
            detected,
        )
        return detected
    return count


def _visible_device_ids() -> list[str]:
    raw = os.getenv("CUDA_VISIBLE_DEVICES", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _run_children(args: argparse.Namespace, gpu_count: int) -> int:
    """Run one worker process per GPU so each process owns its model weights."""
    key_path = Path(args.key_file).expanduser()
    try:
        _check_key_file_permissions(key_path)
        base_worker_id, key_id, _ = _read_key_file(
            key_path, args.worker_id, args.key_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot read key file %s: %s", key_path, exc)
        return 2

    repo_root = str(Path(__file__).resolve().parents[1])
    visible_ids = _visible_device_ids()
    children: list[subprocess.Popen] = []

    def device_id_for(index: int) -> str:
        return visible_ids[index] if index < len(visible_ids) else str(index)

    def child_command(index: int) -> list[str]:
        child_worker_id = base_worker_id if index == 0 else f"{base_worker_id}-{index}"
        child_name = args.name if index == 0 else f"{args.name} (GPU {index})"
        command = [
            sys.executable,
            "-m",
            "gpu.worker",
            "--app-url",
            args.app_url,
            "--key-file",
            str(key_path),
            "--key-id",
            key_id,
            "--worker-id",
            child_worker_id,
            "--model",
            args.model,
            "--name",
            child_name,
            "--gpus",
            "1",
            "--child",
            "--log-level",
            args.log_level,
        ]
        if args.ca_file:
            command.extend(["--ca-file", args.ca_file])
        return command

    logger.info(
        "Detected %s GPU(s); starting %s worker process(es)", gpu_count, gpu_count
    )
    try:
        for index in range(gpu_count):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device_id_for(index)
            env["ZE_AFFINITY_MASK"] = device_id_for(index)
            env["PYTHONPATH"] = repo_root + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            process = subprocess.Popen(child_command(index), env=env)
            children.append(process)
            logger.info(
                "Started GPU worker %s for device %s (pid %s)",
                base_worker_id if index == 0 else f"{base_worker_id}-{index}",
                device_id_for(index),
                process.pid,
            )
    except Exception:
        logger.exception("Failed to start GPU workers")
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            process.wait(timeout=10)
        return 1

    stop = False

    def request_stop(signum, frame):  # noqa: ANN001
        nonlocal stop
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, request_stop)
        except (ValueError, OSError):
            pass

    exit_code = 0
    reported: set[int] = set()
    try:
        while not stop:
            for process in children:
                code = process.poll()
                if code is not None and code != 0 and process.pid not in reported:
                    reported.add(process.pid)
                    exit_code = 1
                    logger.error(
                        "GPU worker pid %s exited with code %s", process.pid, code
                    )
            if all(process.poll() is not None for process in children):
                break
            time.sleep(1)
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
        for process in children:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return 0 if stop else exit_code


def _check_key_file_permissions(path: Path) -> None:
    if os.name != "posix":
        return
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        logger.warning(
            "Key file %s is accessible by group/others (mode %o); chmod 600 it",
            path,
            mode,
        )


class GpuWorker:
    def __init__(self, config: WorkerConfig, backend: Any):
        self.config = config
        self.backend = backend
        self.heartbeat_interval = 15
        self.lease_seconds = 300
        self.max_result_bytes = 8 * 1024 * 1024
        self._stop = asyncio.Event()
        self._ws: Any = None
        self._channel: SecureChannel | None = None
        self._send_lock = asyncio.Lock()
        self._connected = asyncio.Event()
        self._heartbeat_task: asyncio.Task | None = None
        self._active_job: dict[str, Any] | None = None
        self._active_task: asyncio.Task | None = None
        self._cancel_event: asyncio.Event | None = None
        self._pending_result: dict[str, Any] | None = None
        self._result_event = asyncio.Event()
        self._result_retry = True

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        stop_watcher = asyncio.create_task(self._stop_watcher(), name="gpu-stop")
        try:
            backoff = self.config.reconnect_min
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    await self._connect_and_serve()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - reconnect forever
                    logger.warning("Connection ended: %s", exc)
                finally:
                    await self._close_connection()
                if self._stop.is_set():
                    break

                # A long-lived connection resets the backoff.
                if time.monotonic() - started > 10:
                    backoff = self.config.reconnect_min
                logger.info("Reconnecting in %.1fs", backoff)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except TimeoutError:
                    pass
                backoff = min(backoff * 2, self.config.reconnect_max)
        finally:
            stop_watcher.cancel()
            await self._shutdown_job()

    async def _stop_watcher(self) -> None:
        await self._stop.wait()
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - shutting down anyway
                pass

    async def _connect_and_serve(self) -> None:
        ssl_context = _build_ssl_context(self.config.app_url, self.config.ca_file)
        logger.info(
            "Connecting to %s as %s (%s/%s)",
            self.config.app_url,
            self.config.worker_id,
            self.config.key_id,
            os.uname().nodename if hasattr(os, "uname") else self.config.name,
        )
        async with websockets.connect(
            self.config.app_url,
            ssl=ssl_context,
            max_size=WS_MAX_SIZE,
            open_timeout=20,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._send_lock = asyncio.Lock()
            self._channel = await self._handshake(ws)
            self._connected.set()
            self._result_event.set()
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="gpu-heartbeat"
            )
            logger.info("Authenticated with app backend")

            active = self._active_ref()
            await self._send(
                "ready",
                {
                    "name": self.config.name,
                    "model": self.config.model,
                    "device": self._device_name(),
                    "kind": self._worker_kind(),
                    "capabilities": {
                        "schedulers": [
                            "euler",
                            "euler_a",
                            "unipc",
                            "dpm",
                            "dpm_singlestep",
                            "heun",
                            "lms",
                            "kdpm2",
                            "kdpm2_a",
                        ],
                        "max_width": 2048,
                        "max_height": 2048,
                        "max_steps": 150,
                    },
                    "active_job": active,
                },
            )
            if self._pending_result is not None:
                asyncio.create_task(self._retry_pending_result())

            while not self._stop.is_set():
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    raise ProtocolError("expected a JSON text frame")
                try:
                    envelope = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ProtocolError("invalid envelope JSON") from exc
                msg_type, payload = self._channel.decrypt_json(envelope)
                await self._handle_message(msg_type, payload)

    async def _close_connection(self) -> None:
        self._connected.clear()
        # Wake a result sender that is waiting for an acknowledgement so it can
        # retry on the next connection instead of waiting out its timeout.
        self._result_event.set()
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
            self._heartbeat_task = None
        self._ws = None
        self._channel = None

    async def _shutdown_job(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._active_task is not None:
            self._active_task.cancel()
            try:
                await self._active_task
            except (asyncio.CancelledError, Exception):
                pass
            self._active_task = None
        self._active_job = None
        self._pending_result = None

    async def _handshake(self, ws: Any) -> SecureChannel:
        client_nonce = new_nonce()
        await ws.send(
            json.dumps(
                {
                    "type": "hello",
                    "v": PROTOCOL_VERSION,
                    "worker_id": self.config.worker_id,
                    "key_id": self.config.key_id,
                    "client_nonce": b64e(client_nonce),
                }
            )
        )
        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        challenge = _loads_message(raw, "challenge")
        if challenge.get("v") != PROTOCOL_VERSION:
            raise ProtocolError("protocol version mismatch")
        session_id = challenge.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ProtocolError("missing session id")
        server_nonce = b64d(challenge.get("server_nonce", ""))
        if len(server_nonce) != 32:
            raise ProtocolError("invalid server nonce")

        proof = make_client_proof(
            self.config.key,
            self.config.worker_id,
            self.config.key_id,
            client_nonce,
            server_nonce,
            session_id,
        )
        await ws.send(
            json.dumps({"type": "auth", "v": PROTOCOL_VERSION, "proof": b64e(proof)})
        )

        raw = await asyncio.wait_for(ws.recv(), timeout=10)
        auth_ok = _loads_message(raw, "auth_ok")
        expected = make_server_proof(
            self.config.key,
            self.config.worker_id,
            self.config.key_id,
            client_nonce,
            server_nonce,
            session_id,
        )
        provided = b64d(auth_ok.get("proof", ""))
        if not verify_proof(expected, provided):
            raise ProtocolError("server authentication failed")

        channel = SecureChannel.from_psk(
            self.config.key, client_nonce, server_nonce, session_id, role="client"
        )
        msg_type, welcome = channel.decrypt_json(
            json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        )
        if msg_type != "welcome" or welcome.get("protocol") != PROTOCOL_VERSION:
            raise ProtocolError("expected welcome")
        self.heartbeat_interval = max(
            1, min(int(welcome.get("heartbeat_interval", 15)), 300)
        )
        self.lease_seconds = max(1, int(welcome.get("lease_seconds", 300)))
        self.max_result_bytes = max(
            1,
            min(
                int(welcome.get("max_result_bytes", 8 * 1024 * 1024)),
                64 * 1024 * 1024,
            ),
        )
        return channel

    # ------------------------------------------------------------------ #
    # Sending and receiving
    # ------------------------------------------------------------------ #
    async def _send(self, msg_type: str, payload: dict[str, Any]) -> None:
        ws = self._ws
        channel = self._channel
        lock = self._send_lock
        if ws is None or channel is None or not self._connected.is_set():
            raise ConnectionError("app backend is not connected")
        async with lock:
            if self._ws is not ws or self._channel is not channel:
                raise ConnectionError("connection changed while sending")
            envelope = channel.encrypt_json(msg_type, payload)
            await ws.send(json.dumps(envelope))

    async def _heartbeat_loop(self) -> None:
        while self._connected.is_set() and not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.heartbeat_interval
                )
                return
            except TimeoutError:
                pass
            try:
                await self._send(
                    "heartbeat",
                    {
                        "status": "busy" if self._active_job else "ready",
                        "active_job": self._active_ref(),
                    },
                )
            except Exception:  # noqa: BLE001 - reconnect loop owns cleanup
                return

    def _active_ref(self) -> dict[str, Any] | None:
        if self._active_job is None:
            return None
        return {
            "job_id": self._active_job["job_id"],
            "attempt": self._active_job["attempt"],
        }

    async def _handle_message(self, msg_type: str, payload: dict[str, Any]) -> None:
        if msg_type == "job":
            await self._on_job(payload)
        elif msg_type == "cancel":
            await self._on_cancel(payload)
        elif msg_type == "result_ack":
            await self._on_result_ack(payload, success=True)
        elif msg_type == "result_nack":
            await self._on_result_ack(payload, success=False)
        elif msg_type == "heartbeat_ack" or msg_type == "error_ack":
            pass
        elif msg_type == "shutdown":
            await self._on_shutdown(payload)
        else:
            logger.warning("Unknown message from app: %s", msg_type)

    async def _on_shutdown(self, payload: dict[str, Any]) -> None:
        """Exit so the host session ends.

        Kaggle exposes no stop endpoint, so the app stops a run by telling the
        worker inside it to quit. The supervising parent exits once every child
        is gone, which ends the notebook and the GPU session with it.
        """
        reason = str(payload.get("reason") or "requested by the app")
        logger.info("Shutdown requested (%s); exiting", reason)
        try:
            await self._safe_send("shutdown_ack", {"reason": reason})
        except Exception:  # noqa: BLE001 - best effort before exiting
            pass
        await asyncio.sleep(0.5)
        os._exit(0)

    async def _on_job(self, payload: dict[str, Any]) -> None:
        job_id = payload.get("job_id")
        attempt = payload.get("attempt")
        params = payload.get("params")
        if (
            not isinstance(job_id, str)
            or not job_id
            or not isinstance(attempt, int)
            or attempt <= 0
            or not isinstance(params, dict)
        ):
            logger.warning("Ignoring malformed job")
            return
        if self._active_job is not None:
            await self._send(
                "job_rejected",
                {
                    "job_id": job_id,
                    "attempt": attempt,
                    "reason": "worker is already busy",
                },
            )
            return

        self._active_job = {
            "job_id": job_id,
            "attempt": attempt,
            "params": params,
        }
        self._pending_result = None
        self._result_event.clear()
        self._cancel_event = asyncio.Event()
        try:
            await self._send("job_accepted", {"job_id": job_id, "attempt": attempt})
        except Exception:  # noqa: BLE001 - keep working while offline
            logger.warning("Could not acknowledge job %s; continuing offline", job_id)
        self._active_task = asyncio.create_task(
            self._run_job(job_id, attempt, params), name=f"job-{job_id}"
        )

    async def _on_cancel(self, payload: dict[str, Any]) -> None:
        job_id = payload.get("job_id")
        attempt = payload.get("attempt")
        if self._active_job is not None and (
            self._active_job["job_id"] != job_id
            or self._active_job["attempt"] != attempt
        ):
            return
        if self._active_job is None:
            return
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._pending_result is not None:
            self._pending_result = None
            self._result_event.set()
            await self._safe_send("cancelled", {"job_id": job_id, "attempt": attempt})
        logger.info("Cancellation requested for job %s", job_id)

    async def _on_result_ack(self, payload: dict[str, Any], success: bool) -> None:
        if self._active_job is None:
            return
        if (
            payload.get("job_id") != self._active_job["job_id"]
            or payload.get("attempt") != self._active_job["attempt"]
        ):
            return
        if success:
            logger.info("Result accepted for job %s", self._active_job["job_id"])
            self._result_retry = False
        else:
            retry = payload.get("retry", True)
            self._result_retry = bool(retry)
            logger.warning(
                "Result rejected for job %s: %s",
                self._active_job["job_id"],
                payload.get("error"),
            )
        self._result_event.set()

    # ------------------------------------------------------------------ #
    # Job execution
    # ------------------------------------------------------------------ #
    async def _run_job(self, job_id: str, attempt: int, params: dict[str, Any]) -> None:
        try:
            image_bytes = await self.backend.infer(
                job_id,
                prompt=str(params.get("prompt", "")),
                negative_prompt=str(params.get("negative_prompt", "")),
                guidance_scale=float(params.get("guidance_scale", 1.0)),
                steps=int(params.get("steps", 28)),
                seed=int(params.get("seed", 42)),
                width=int(params.get("width", 1024)),
                height=int(params.get("height", 1024)),
                clip_skip=int(params.get("clip_skip", 2)),
                scheduler=str(params.get("scheduler", "euler")),
                on_step=self._on_step,
            )
        except InferenceCancelled:
            logger.info("Job %s cancelled", job_id)
            self._clear_active()
            await self._safe_send("cancelled", {"job_id": job_id, "attempt": attempt})
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self._clear_active()
            await self._safe_send(
                "error",
                {"job_id": job_id, "attempt": attempt, "error": str(exc)[:4000]},
            )
            return

        if self._cancel_event is not None and self._cancel_event.is_set():
            logger.info("Job %s cancelled before result upload", job_id)
            self._clear_active()
            await self._safe_send("cancelled", {"job_id": job_id, "attempt": attempt})
            return
        if image_bytes is None:
            self._clear_active()
            await self._safe_send(
                "error",
                {
                    "job_id": job_id,
                    "attempt": attempt,
                    "error": "backend returned no image",
                },
            )
            return
        if len(image_bytes) > self.max_result_bytes:
            self._clear_active()
            await self._safe_send(
                "error",
                {
                    "job_id": job_id,
                    "attempt": attempt,
                    "error": "result is larger than the app limit",
                },
            )
            return

        self._pending_result = {
            "job_id": job_id,
            "attempt": attempt,
            "image": image_bytes,
        }
        self._result_event.clear()
        self._result_retry = True
        await self._send_pending_result_loop()

    async def _on_step(self, step: int, total: int) -> bool:
        if self._cancel_event is not None and self._cancel_event.is_set():
            return False
        if self._active_job is None:
            return False
        try:
            await self._send(
                "progress",
                {
                    "job_id": self._active_job["job_id"],
                    "attempt": self._active_job["attempt"],
                    "step": step,
                    "total": total,
                },
            )
        except Exception:  # noqa: BLE001 - keep generating while offline
            pass
        return self._cancel_event is None or not self._cancel_event.is_set()

    async def _send_pending_result_loop(self) -> None:
        while self._pending_result is not None and not self._stop.is_set():
            if not self._connected.is_set():
                await asyncio.sleep(1)
                continue
            pending = self._pending_result
            # Clear before sending so an acknowledgement cannot race with the
            # clear and leave the worker waiting for a duplicate ack.
            self._result_event.clear()
            try:
                await self._send(
                    "result",
                    {
                        "job_id": pending["job_id"],
                        "attempt": pending["attempt"],
                        "image_b64": b64e(pending["image"]),
                        "mime": "image/jpeg",
                    },
                )
            except Exception:  # noqa: BLE001 - reconnect and retry
                await asyncio.sleep(2)
                continue

            try:
                await asyncio.wait_for(self._result_event.wait(), timeout=60)
            except TimeoutError:
                logger.warning("No result acknowledgement; retrying")
                continue

            if self._pending_result is None:
                # Cancelled while waiting for the ack.
                break
            if not self._result_retry:
                logger.info("Result for job %s acknowledged", pending["job_id"])
                self._clear_active()
                break
            # result_nack asked for a retry
            await asyncio.sleep(2)
        self._clear_active()

    async def _retry_pending_result(self) -> None:
        if self._pending_result is not None and not self._task_is_running():
            await self._send_pending_result_loop()

    def _task_is_running(self) -> bool:
        return self._active_task is not None and not self._active_task.done()

    def _clear_active(self) -> None:
        self._active_job = None
        self._active_task = None
        self._cancel_event = None
        self._pending_result = None

    async def _safe_send(self, msg_type: str, payload: dict[str, Any]) -> None:
        try:
            await self._send(msg_type, payload)
        except Exception:  # noqa: BLE001 - message is lost if offline
            pass

    def _worker_kind(self) -> str:
        """Report where this worker runs.

        Kaggle kernels always carry KAGGLE_KERNEL_RUN_TYPE and a
        /kaggle/working directory; reporting that explicitly lets the UI offer
        "Disconnect" without guessing from the worker id.
        """
        if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle/working"):
            return "kaggle"
        return ""

    def _device_name(self) -> str:
        if sys.platform == "darwin" and platform.machine().lower() == "arm64":
            return "mlx"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
            return "cpu"
        except Exception:  # noqa: BLE001
            return platform.machine()


def _loads_message(raw: Any, expected_type: str) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raise ProtocolError("expected a JSON text frame")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid JSON") from exc
    if not isinstance(obj, dict) or obj.get("type") != expected_type:
        raise ProtocolError(f"expected {expected_type}")
    return obj


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Outbound-only ImagineIt GPU worker")
    parser.add_argument(
        "--app-url",
        default=os.getenv("APP_WS_URL", DEFAULT_APP_URL),
        help="App backend WebSocket URL (ws:// or wss://)",
    )
    parser.add_argument(
        "--key-file",
        default=os.getenv(
            "GPU_WORKER_KEY_FILE", str(Path.home() / ".imagineit" / "worker.json")
        ),
        help="JSON file containing worker_id, key_id and base64 key",
    )
    parser.add_argument("--worker-id", default=os.getenv("GPU_WORKER_ID"))
    parser.add_argument("--key-id", default=os.getenv("GPU_WORKER_KEY_ID"))
    parser.add_argument(
        "--model",
        default=os.getenv("GPU_MODEL", DEFAULT_MODEL),
        help="Hugging Face model repository",
    )
    parser.add_argument(
        "--name",
        default=os.getenv("GPU_WORKER_NAME", platform.node() or "gpu-worker"),
    )
    parser.add_argument(
        "--ca-file",
        default=os.getenv("GPU_CA_FILE"),
        help="CA bundle for a self-signed/private WSS certificate",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Only one concurrent job is currently supported",
    )
    parser.add_argument(
        "--gpus",
        default=os.getenv("GPU_WORKER_GPUS", "1"),
        help="Number of GPU worker processes: auto, all, or a positive integer",
    )
    parser.add_argument(
        "--child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.max_concurrency != 1:
        logger.error("max-concurrency > 1 is not supported yet")
        return 2

    try:
        gpu_count = _resolve_gpu_count(args.gpus)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    if gpu_count > 1 and not args.child:
        return _run_children(args, gpu_count)

    key_path = Path(args.key_file).expanduser()
    try:
        _check_key_file_permissions(key_path)
        worker_id, key_id, key = _read_key_file(key_path, args.worker_id, args.key_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Cannot read key file %s: %s", key_path, exc)
        return 2

    config = WorkerConfig(
        app_url=args.app_url,
        worker_id=worker_id,
        key_id=key_id,
        key=key,
        model=args.model,
        name=args.name,
        ca_file=args.ca_file,
    )

    try:
        logger.info("Loading model %s", config.model)
        from gpu.backends import LoadedSDXL

        backend = LoadedSDXL(config.model)
    except ModuleNotFoundError:
        logger.error("GPU dependencies are missing. Run: pip install -e '.[gpu]'")
        return 1
    except Exception:
        logger.exception("Failed to load model")
        return 1

    worker = GpuWorker(config, backend)

    async def runner() -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, worker._stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        await worker.run()

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
