"""Delete pre-account inferences and every object in the image bucket.

Run once, after `alembic upgrade head`, when switching to the account model:
rows created before accounts have no owner, and their JPEGs were stored in the
clear.  Both are unreachable from the authenticated API, so this simply clears
the leftovers.

    .venv/bin/python scripts/wipe_legacy_data.py
"""

from __future__ import annotations

import asyncio
import os


async def _wipe() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    from minio import Minio
    from sqlalchemy import delete

    from app.database.db import db_session
    from app.database.minio import BUCKET_NAME
    from app.database.models import Image

    async with db_session() as session:
        result = await session.execute(delete(Image))
        await session.commit()
    rows = int(result.rowcount or 0)

    client = Minio(
        f"{os.getenv('MINIO_HOST', 'localhost')}:{os.getenv('MINIO_PORT')}",
        access_key=os.getenv("MINIO_ROOT_USER"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
        secure=False,
    )
    objects = 0
    if client.bucket_exists(BUCKET_NAME):
        for item in client.list_objects(BUCKET_NAME, recursive=True):
            client.remove_object(BUCKET_NAME, item.object_name)
            objects += 1

    print(f"deleted {rows} inference row(s) and {objects} stored object(s)")


if __name__ == "__main__":
    asyncio.run(_wipe())
