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


@router.post("/accounts", response_model=AccountCreated, status_code=201)
async def create_account(payload: AccountCreate):
    try:
        account, token = await accounts.create_account(payload.public_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AccountCreated(account_id=account.id, token=token)


@router.get("/account")
async def current_account(account: Account = Depends(require_account)):
    return {
        "account_id": account.id,
        "created_at": account.created_at.isoformat() if account.created_at else None,
    }
