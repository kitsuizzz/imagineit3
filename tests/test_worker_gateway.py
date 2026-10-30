"""Integration coverage for the worker gateway handshake.

These tests do not touch MariaDB or MinIO; they only exercise the authenticated
WebSocket handshake and the first encrypted worker message.
"""

import asyncio
import os
import time

os.environ.setdefault("DB_ENGINE", "mysql")
os.environ.setdefault("DB_DRIVER", "aiomysql")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "3306")
os.environ.setdefault("DB_NAME", "test")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.internal.workers import router
from app.api.v3.endpoints import workers as workers_endpoint
from app.services.worker_gateway import WorkerConnection, WorkerGateway
from app.worker_auth import WorkerKey, WorkerKeyStore
from shared.protocol import (
    SecureChannel,
    b64d,
    b64e,
    make_client_proof,
    make_server_proof,
    new_nonce,
    verify_proof,
)


def test_handshake_and_encrypted_ready():
    psk = bytes(range(32))
    store = WorkerKeyStore([WorkerKey("gpu-test", "k1", psk)])
    app = FastAPI()
    gateway = WorkerGateway(app, store)
    app.state.gateway = gateway
    app.include_router(router, prefix="/internal/v1")

    client = TestClient(app)
    with client.websocket_connect("/internal/v1/workers/ws") as ws:
        client_nonce = new_nonce()
        ws.send_json(
            {
                "type": "hello",
                "v": 1,
                "worker_id": "gpu-test",
                "key_id": "k1",
                "client_nonce": b64e(client_nonce),
            }
        )
        challenge = ws.receive_json()
        assert challenge["type"] == "challenge"
        server_nonce = b64d(challenge["server_nonce"])
        session_id = challenge["session_id"]

        ws.send_json(
            {
                "type": "auth",
                "v": 1,
                "proof": b64e(
                    make_client_proof(
                        psk,
                        "gpu-test",
                        "k1",
                        client_nonce,
                        server_nonce,
                        session_id,
                    )
                ),
            }
        )
        auth_ok = ws.receive_json()
        assert auth_ok["type"] == "auth_ok"
        assert verify_proof(
            make_server_proof(
                psk,
                "gpu-test",
                "k1",
                client_nonce,
                server_nonce,
                session_id,
            ),
            b64d(auth_ok["proof"]),
        )

        channel = SecureChannel.from_psk(
            psk, client_nonce, server_nonce, session_id, role="client"
        )
        msg_type, welcome = channel.decrypt_json(ws.receive_json())
        assert msg_type == "welcome"
        assert welcome["protocol"] == 1

        ws.send_json(
            channel.encrypt_json(
                "ready",
                {
                    "name": "test-gpu-host",
                    "model": "test-model",
                    "device": "test-device",
                    "capabilities": {},
                    "active_job": None,
                },
            )
        )


def test_public_worker_snapshot_redacts_private_fields():
    async def run():
        gateway = WorkerGateway(
            FastAPI(), WorkerKeyStore([WorkerKey("gpu-test", "k1", bytes(32))])
        )
        conn = WorkerConnection(
            websocket=None,
            worker_id="gpu-test",
            key_id="k1",
            session_id="secret-session",
            channel=None,
        )
        conn.name = "test-gpu-host"
        conn.model = "test-model"
        conn.device = "mlx"
        conn.status = "busy"
        conn.active_job_id = "secret-promise-id"
        conn.connected_at = time.monotonic() - 65
        conn.last_seen = time.monotonic() - 3
        await gateway.register(conn)
        return await gateway.public_workers()

    workers = asyncio.run(run())
    assert len(workers) == 1
    worker = workers[0]
    assert worker["worker_id"] == "gpu-test"
    assert worker["name"] == "test-gpu-host"
    assert worker["model"] == "test-model"
    assert worker["device"] == "mlx"
    assert worker["status"] == "busy"
    assert worker["has_active_job"] is True
    assert 64 <= worker["connected_seconds"] <= 67
    assert 2 <= worker["last_seen_seconds_ago"] <= 4
    assert "active_job_id" not in worker
    assert "key_id" not in worker
    assert "session_id" not in worker


def test_workers_endpoint_shape():
    class FakeGateway:
        async def public_workers(self):
            return [
                {
                    "worker_id": "gpu-test",
                    "name": "test-gpu-host",
                    "model": "test-model",
                    "device": "mlx",
                    "status": "ready",
                    "connected_seconds": 12,
                    "last_seen_seconds_ago": 1,
                    "has_active_job": False,
                }
            ]

    app = FastAPI()
    app.state.gateway = FakeGateway()
    app.include_router(workers_endpoint.router)
    response = TestClient(app).get("/workers")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["workers"][0]["worker_id"] == "gpu-test"
