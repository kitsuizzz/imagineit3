"""Account creation and identity.

Signup is the only unauthenticated call: the browser generates a P-256
keypair, sends the public half, and receives a bearer token exactly once.
There is no password, no email, and no recovery — the token is the credential.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v3.deps import require_account
from app.database.models import Account
from app.services import accounts

router = APIRouter()


class AccountCreate(BaseModel):
    public_key: str = Field(min_length=80, max_length=200)


class AccountCreated(BaseModel):
    account_id: str
    token: str


class PublicKeyIn(BaseModel):
    public_key: str = Field(min_length=80, max_length=200)
    wrapped_key: str | None = Field(default=None, max_length=4000)


class WrappedKeyIn(BaseModel):
    wrapped_key: str = Field(min_length=40, max_length=4000)


@router.post("/accounts", response_model=AccountCreated, status_code=201)
async def create_account(payload: AccountCreate):
    try:
        account, token = await accounts.create_account(payload.public_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AccountCreated(account_id=account.id, token=token)


@router.put("/account/public-key")
async def replace_public_key(
    payload: PublicKeyIn, account: Account = Depends(require_account)
):
    try:
        await accounts.set_public_key(
            account.id, payload.public_key, payload.wrapped_key
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"account_id": account.id}


@router.put("/account/wrapped-key")
async def store_wrapped_key(
    payload: WrappedKeyIn, account: Account = Depends(require_account)
):
    await accounts.set_wrapped_key(account.id, payload.wrapped_key)
    return {"stored": True}


@router.get("/account/wrapped-key")
async def read_wrapped_key(account: Account = Depends(require_account)):
    wrapped = await accounts.get_wrapped_key(account.id)
    if not wrapped:
        raise HTTPException(status_code=404, detail="no_wrapped_key")
    return {"wrapped_key": wrapped}


@router.get("/account")
async def current_account(account: Account = Depends(require_account)):
    return {
        "account_id": account.id,
        "created_at": account.created_at.isoformat() if account.created_at else None,
    }
