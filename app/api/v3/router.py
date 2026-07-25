from fastapi import APIRouter
from .endpoints import images

v3_router = APIRouter()


@v3_router.get("/status")
async def get_v3_status():
    return {"status": "ok"}


v3_router.include_router(images.router, prefix="/images")
