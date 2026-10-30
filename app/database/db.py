from contextlib import asynccontextmanager
import os

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm.decl_api import DeclarativeMeta

_DATABASE_URL = None
_AsyncSessionLocal = None
_Base = None


def init_db():
    global _DATABASE_URL
    global _AsyncSessionLocal
    global _Base
    _DATABASE_URL = f"{os.getenv('DB_ENGINE')}+{os.getenv('DB_DRIVER')}://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

    echo = os.getenv("DB_ECHO", "false").lower() in {"1", "true", "yes"}
    engine = create_async_engine(
        _DATABASE_URL, pool_recycle=3600, pool_pre_ping=True, echo=echo
    )
    _AsyncSessionLocal = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    _Base = declarative_base()


async def get_db():
    if not _AsyncSessionLocal:
        raise ConnectionError("DB not connected")
    async with _AsyncSessionLocal() as session:
        yield session


@asynccontextmanager
async def db_session():
    if not _AsyncSessionLocal:
        raise ConnectionError("DB not connected")
    async with _AsyncSessionLocal() as session:
        yield session


def get_db_url() -> str:
    global _DATABASE_URL
    if not _DATABASE_URL:
        raise ValueError("DB not initialized")
    return _DATABASE_URL


def get_base() -> DeclarativeMeta:
    global _Base
    if not _Base:
        raise ValueError("DB not initialized")
    return _Base
