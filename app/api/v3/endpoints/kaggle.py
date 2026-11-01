"""Per-user Kaggle control.

The browser never talks to Kaggle.  It stores a ``KGAT_`` API token with the
account, and the app uses that token to push the notebook, which starts a run.
Kaggle has no stop endpoint, so disconnect asks the worker inside the notebook
to exit instead (the worker's shutdown handler ends the session).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.v3.deps import require_account
from app.database.models import Account
from app.services import accounts, kaggle_api, worker_keys

router = APIRouter()


class KaggleApiKeyIn(BaseModel):
    api_key: str = Field(min_length=8, max_length=500)
    kernel_ref: str | None = Field(default=None, max_length=200)
    username: str | None = Field(default=None, max_length=100)


@router.get("/kaggle/credentials")
async def kaggle_credentials_status(account: Account = Depends(require_account)):
    return {
        "configured": await accounts.has_kaggle_credentials(account.id),
        "kernel_ref": await accounts.load_kaggle_kernel_ref(account.id),
        "username": await accounts.load_kaggle_username(account.id),
        "worker_keys": len(await worker_keys.list_for(account.id)),
    }


@router.put("/kaggle/credentials")
async def save_kaggle_credentials(
    payload: KaggleApiKeyIn,
    account: Account = Depends(require_account),
):
    await accounts.save_kaggle_api_key(
        account.id, payload.api_key, payload.kernel_ref, payload.username
    )
    return {"configured": True}


@router.delete("/kaggle/credentials")
async def remove_kaggle_credentials(account: Account = Depends(require_account)):
    await accounts.delete_kaggle_credentials(account.id)
    return {"configured": False}


@router.post("/kaggle/connect")
async def connect_kaggle(
    request: Request,
    account: Account = Depends(require_account),
):
    api_key = await accounts.load_kaggle_api_key(account.id)
    if not api_key:
        raise HTTPException(
            status_code=400, detail={"error": "kaggle_api_key_required"}
        )

    kernel_ref = await accounts.load_kaggle_kernel_ref(account.id)
    username = await accounts.load_kaggle_username(account.id)
    if not username and not kernel_ref:
        username = await asyncio.to_thread(kaggle_api.detect_username, api_key)
        if username:
            await accounts.save_kaggle_api_key(account.id, api_key, None, username)
    if not kernel_ref:
        if not username:
            raise HTTPException(
                status_code=400, detail={"error": "kaggle_username_unknown"}
            )
        kernel_ref = f"{username}/imagineitv3"
    gateway = getattr(request.app.state, "gateway", None)
    busy = {conn.worker_id for conn in await gateway.snapshot()} if gateway else set()
    worker = await worker_keys.ensure_for(account.id, busy)
    key_payload = {
        "worker_id": worker["worker_id"],
        "key_id": worker["key_id"],
        "key": worker["key"],
    }

    try:
        user, slug = kaggle_api.split_ref(kernel_ref)
        status = await asyncio.to_thread(kaggle_api.kernel_status, api_key, user, slug)
        if kaggle_api.is_running(status):
            return {"report": f"{user}/{slug} is already {status}."}
        result = await asyncio.to_thread(
            kaggle_api.start_kernel, api_key, kernel_ref, worker_key=key_payload
        )
    except kaggle_api.KaggleApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if result.get("updated"):
        return {
            "report": f"Updated {result['ref']} to the app template and started it."
        }
    return {"report": f"Started {result['ref']} (notebook already up to date)."}


@router.post("/kaggle/disconnect")
async def disconnect_kaggle(
    request: Request,
    account: Account = Depends(require_account),
):
    gateway = getattr(request.app.state, "gateway", None)
    if gateway is None:
        return {"report": "No worker gateway is running."}

    sent = await gateway.shutdown_kaggle_workers(
        "requested from the web UI", account.id
    )
    if sent == 0:
        return {"report": "No Kaggle worker is connected."}
    return {"report": f"Asked {sent} Kaggle worker(s) to stop."}
