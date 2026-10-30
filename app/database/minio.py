import os
import io

from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI

from minio import Minio
from minio.error import S3Error

BUCKET_NAME = "imagineit-images"


async def init_minio(app: FastAPI):
    minio_client = Minio(
        f"{os.getenv('MINIO_HOST', 'localhost')}:{os.getenv('MINIO_PORT')}",
        access_key=os.getenv("MINIO_ROOT_USER"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
        secure=False,
    )
    app.state.minio = minio_client
    await connect_bucket(app, BUCKET_NAME)


def _connect_bucket(app: FastAPI, bucket_name: str):
    minio_client: Minio = app.state.minio
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)


async def connect_bucket(app: FastAPI, bucket_name: str):
    await run_in_threadpool(_connect_bucket, app, bucket_name)


def _upload_image(app: FastAPI, image_data: io.BytesIO, object_name: str):
    minio_client: Minio = app.state.minio
    minio_client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=image_data,
        length=len(image_data.getvalue()),
        content_type="image/jpeg",
    )


async def upload_image(app: FastAPI, image_data: io.BytesIO, object_name: str):
    await run_in_threadpool(_upload_image, app, image_data, object_name)


def _retrieve_image(app: FastAPI, object_name: str):
    minio_client: Minio = app.state.minio
    data_stream = None
    try:
        data_stream = minio_client.get_object(
            bucket_name=BUCKET_NAME, object_name=object_name
        )
        image_bytes = data_stream.read()
        return image_bytes
    except S3Error:
        pass
    finally:
        if data_stream:
            data_stream.close()
            data_stream.release_conn()
    return None


async def retrieve_image(app: FastAPI, object_name: str):
    return await run_in_threadpool(_retrieve_image, app, object_name)
