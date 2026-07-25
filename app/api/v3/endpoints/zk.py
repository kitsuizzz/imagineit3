from fastapi import APIRouter

router = APIRouter()


@router.get("/identity")
async def generate_client_secret():
    pass
