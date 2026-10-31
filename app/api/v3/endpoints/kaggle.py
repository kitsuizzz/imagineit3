"""Start or stop the Kaggle GPU worker through the browserful agent.

The browserful key stays on the server: the browser posts credentials to this
API, and this module forwards the instruction to browserful with the bearer
token from ``BROWSERFUL_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v3.deps import require_account
from app.database.models import Account
from app.services import accounts

router = APIRouter()

DEFAULT_BROWSERFUL_URL = "https://browserful.kitsui.co"
BROWSERFUL_TIMEOUT_SECONDS = 660
BROWSERFUL_SESSION = "imagineit-kaggle"

CONNECT_PROMPT = (
    "Open Kaggle then open notebook called imagineitv3. "
    "Select GPU 2x T4s (Make sure you hover on them first, then click) "
    "then run the notebook. If not logged in, signin with email: {email}, pass: {password}"
)

DISCONNECT_PROMPT = (
    "Open Kaggle then select notebook imagineitv3. "
    "If it is running (Draft Session Starting means its running), Stop this instance. "
    "If not logged in, signin with email: {email}, pass: {password}"
)


class KaggleCredentials(BaseModel):
    email: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=1, max_length=200)


class KaggleCredentialsInput(BaseModel):
    """Body for connect/disconnect, where both fields are optional."""

    email: str | None = Field(default=None, max_length=200)
    password: str | None = Field(default=None, max_length=200)


def _run_browserful(prompt: str, password: str, session: str) -> dict:
    api_key = os.getenv("BROWSERFUL_KEY", "").strip()
    if not api_key:
        raise RuntimeError("BROWSERFUL_KEY is not configured on the app backend")

    base_url = os.getenv("BROWSERFUL_URL", DEFAULT_BROWSERFUL_URL).rstrip("/")
    payload = json.dumps(
        {
            "prompt": prompt,
            "sessionId": session,
            # Keep the password out of browserful's event stream and reports.
            "redact": [password],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url}/api/run",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request, timeout=BROWSERFUL_TIMEOUT_SECONDS
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        detail = detail.replace(password, "«redacted»")
        raise RuntimeError(f"browserful returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach browserful: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("browserful did not answer before the timeout") from exc


async def _dispatch(
    template: str, credentials: KaggleCredentials, account_id: str
) -> dict:
    prompt = template.format(email=credentials.email, password=credentials.password)
    # One browserful session per account: two users must never share a Kaggle
    # browser, or their notebook actions would interleave.
    session = f"{BROWSERFUL_SESSION}-{account_id}"
    try:
        return await asyncio.to_thread(
            _run_browserful, prompt, credentials.password, session
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


async def _resolve_credentials(
    account: Account, payload: KaggleCredentialsInput | None
) -> KaggleCredentials:
    """Use the supplied login and persist it, or fall back to the stored one.

    Browserful needs the password in the clear to sign in, so this is the one
    piece of data the server encrypts with APP_SECRET instead of sealing to the
    client key.  It is never returned by the API.
    """
    if payload is not None and payload.email and payload.password:
        await accounts.save_kaggle_credentials(
            account.id, payload.email, payload.password
        )
        return KaggleCredentials(email=payload.email, password=payload.password)
    if payload is not None and (payload.email or payload.password):
        raise HTTPException(
            status_code=400,
            detail={"error": "kaggle_credentials_incomplete"},
        )
    stored = await accounts.load_kaggle_credentials(account.id)
    if stored is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "kaggle_credentials_required"},
        )
    email, password = stored
    return KaggleCredentials(email=email, password=password)


@router.post("/kaggle/connect")
async def connect_kaggle(
    payload: KaggleCredentialsInput | None = None,
    account: Account = Depends(require_account),
):
    credentials = await _resolve_credentials(account, payload)
    return await _dispatch(CONNECT_PROMPT, credentials, account.id)


@router.post("/kaggle/disconnect")
async def disconnect_kaggle(
    payload: KaggleCredentialsInput | None = None,
    account: Account = Depends(require_account),
):
    credentials = await _resolve_credentials(account, payload)
    return await _dispatch(DISCONNECT_PROMPT, credentials, account.id)


@router.get("/kaggle/credentials")
async def kaggle_credentials_status(account: Account = Depends(require_account)):
    return {"configured": await accounts.has_kaggle_credentials(account.id)}


@router.put("/kaggle/credentials")
async def save_kaggle_credentials(
    payload: KaggleCredentials,
    account: Account = Depends(require_account),
):
    await accounts.save_kaggle_credentials(account.id, payload.email, payload.password)
    return {"configured": True}


@router.delete("/kaggle/credentials")
async def remove_kaggle_credentials(account: Account = Depends(require_account)):
    await accounts.delete_kaggle_credentials(account.id)
    return {"configured": False}
