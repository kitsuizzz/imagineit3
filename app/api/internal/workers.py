from fastapi import APIRouter, WebSocket

from app.services.worker_gateway import WorkerGateway

router = APIRouter()


@router.websocket("/workers/ws")
async def worker_socket(websocket: WebSocket):
    gateway: WorkerGateway | None = getattr(websocket.app.state, "gateway", None)
    if gateway is None:
        await websocket.close(code=1011, reason="worker gateway is not running")
        return
    await gateway.handle_socket(websocket)
