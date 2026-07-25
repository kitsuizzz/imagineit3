from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

from database import init_db
from database.minio import BUCKET_NAME, connect_bucket, init_minio
from api.v3 import v3_router
from database.valkey import init_valkey

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # App pre-inits
    init_db()
    init_minio(app)
    init_valkey(app)
    await connect_bucket(app, BUCKET_NAME)

    yield  # App execution
    # App exits


app = FastAPI(lifespan=lifespan)
app.include_router(v3_router, prefix="/api/v3")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=4600, reload=True)
