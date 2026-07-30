import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from database.minio import BUCKET_NAME, connect_bucket, init_minio
from api.v3 import v3_router
from database.valkey import init_valkey
from imagen.sdxl import init_sdxl

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # hold ref to event loop
    app.state.event_loop = asyncio.get_running_loop()
    # App pre-inits
    await init_minio(app)
    init_valkey(app)
    init_sdxl(app, "wssdaas/nova-anime-xl-ilv-19.0")
    await connect_bucket(app, BUCKET_NAME)

    yield  # App execution
    # App exits


origins = [
    "http://localhost",
]
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(v3_router, prefix="/api/v3")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=4600, reload=True)
