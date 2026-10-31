import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.internal.router import router as internal_router
from app.api.v3 import v3_router
from app.database.minio import init_minio
from app.database.valkey import init_valkey
from app.services import jobs
from app.services.dispatcher import JobDispatcher
from app.services.worker_gateway import WorkerGateway
from app.worker_auth import WorkerKeyStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep a reference to the loop for any thread-pool integrations.
    app.state.event_loop = asyncio.get_running_loop()

    if len(os.getenv("APP_SECRET", "").strip()) < 32:
        raise RuntimeError("APP_SECRET must be set to at least 32 characters")

    await init_minio(app)
    init_valkey(app)

    # Plaintext prompts only ever live in process memory, so anything that was
    # still unfinished when the process died can no longer be rendered.
    orphaned = await jobs.fail_unfinished(
        "server restarted before the job finished; submit the prompt again"
    )
    if orphaned:
        logger.warning("Failed %s unfinished job(s) after restart", orphaned)

    app.state.worker_keys = WorkerKeyStore.from_env_or_file()
    app.state.gateway = WorkerGateway(app, app.state.worker_keys)
    app.state.dispatcher = JobDispatcher(app.state.gateway)
    await app.state.dispatcher.start()
    logger.info(
        "Worker gateway ready with %s configured key(s)",
        len(app.state.worker_keys),
    )

    yield  # App execution

    await app.state.dispatcher.stop()
    await app.state.gateway.close_all()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(v3_router, prefix="/api/v3")
app.include_router(internal_router, prefix="/internal/v1")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=4600,
        reload=True,
        ws_max_size=32 * 1024 * 1024,
    )
