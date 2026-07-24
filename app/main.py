from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # App pre-inits
    yield  # App execution
    # App exits


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Hello from your FastAPI application!"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=4600, reload=True)
