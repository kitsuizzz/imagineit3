from fastapi import APIRouter

from .workers import router as workers_router

router = APIRouter()
router.include_router(workers_router)
