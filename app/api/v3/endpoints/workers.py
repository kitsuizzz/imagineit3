from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.v3.deps import require_account
from app.api.v3.schemas.workers import WorkerListResponse
from app.database.models import Account
from app.services import worker_keys
from app.services.worker_gateway import WorkerGateway

router = APIRouter()


@router.get("/workers", response_model=WorkerListResponse)
async def list_workers(req: Request, account: Account = Depends(require_account)):
    gateway: WorkerGateway | None = getattr(req.app.state, "gateway", None)
    if gateway is None:
        return {"count": 0, "workers": []}
    workers = [
        worker
        for worker in await gateway.public_workers()
        if worker.get("account_id") == account.id
    ]
    return {"count": len(workers), "workers": workers}


@router.post("/workers/keys", status_code=201)
async def create_worker_key(account: Account = Depends(require_account)):
    return await worker_keys.create(account.id)


@router.get("/workers/keys")
async def list_worker_keys(account: Account = Depends(require_account)):
    return {"keys": await worker_keys.list_for(account.id)}


@router.get("/workers/keys/current")
async def current_worker_key(account: Account = Depends(require_account)):
    """Return the newest key's material, creating one on first use.

    The raw key is only ever returned to the authenticated account, and the
    row stores it encrypted.
    """
    return await worker_keys.ensure_for(account.id)


@router.delete("/workers/keys/{worker_id}")
async def revoke_worker_key(
    worker_id: str, account: Account = Depends(require_account)
):
    if not await worker_keys.revoke(account.id, worker_id):
        raise HTTPException(status_code=404, detail="unknown worker key")
    return {"revoked": True}
