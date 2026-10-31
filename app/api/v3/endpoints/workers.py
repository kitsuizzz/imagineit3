from fastapi import APIRouter, Depends, Request

from app.api.v3.deps import require_account
from app.database.models import Account

from app.api.v3.schemas.workers import WorkerListResponse
from app.services.worker_gateway import WorkerGateway

router = APIRouter()


@router.get("/workers", response_model=WorkerListResponse)
async def list_workers(
    req: Request, account: Account = Depends(require_account)
):
    gateway: WorkerGateway | None = getattr(req.app.state, "gateway", None)
    if gateway is None:
        return {"count": 0, "workers": []}
    workers = await gateway.public_workers()
    return {"count": len(workers), "workers": workers}
