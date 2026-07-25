import os
import io

from fastapi.concurrency import run_in_threadpool

from minio import Minio

BUCKET_NAME = "imagineit-images"

minio_client = Minio(
    f"localhost:{os.getenv('MINIO_PORT')}",
    access_key=os.getenv("MINIO_ROOT_USER"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
    secure=False,
)


def _connect_bucket(bucket_name: str):
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)


async def connect_bucket(bucket_name: str):
    await run_in_threadpool(_connect_bucket, bucket_name)


def _upload_image(image_data: io.BytesIO, object_name: str):
    minio_client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=object_name,
        data=image_data,
        length=len(image_data.getvalue()),
        content_type="image/png",
    )


async def upload_image(image_data: io.BytesIO, object_name: str):
    await run_in_threadpool(_upload_image, image_data, object_name)


def _retrieve_image(object_name: str):
    data_stream = None
    try:
        data_stream = minio_client.get_object(
            bucket_name=BUCKET_NAME, object_name=object_name
        )
        image_bytes = data_stream.read()
        return image_bytes
    finally:
        if data_stream:
            data_stream.close()
            data_stream.release_conn()
    return None


async def retrieve_image(object_name: str):
    await run_in_threadpool(_retrieve_image, object_name)
