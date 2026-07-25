from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from database.minio import connect_bucket, BUCKET_NAME  # noqa: E402
from api.v3 import v3_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # App pre-inits
    # Minio
    await connect_bucket(BUCKET_NAME)

    yield  # App execution
    # App exits


app = FastAPI(lifespan=lifespan)
app.include_router(v3_router, prefix="/api/v3")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=4600, reload=True)
