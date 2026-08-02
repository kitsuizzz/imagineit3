import io
from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, Request, BackgroundTasks, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v3.schemas.images import ImagenRequest, InferenceResponse
from app.database.db import get_db
from app.database.minio import retrieve_image
from app.database.models import Image
from imagen.sdxl import DedicatedSDXLWorker


router = APIRouter()


@router.post("/inferences")
async def imagen(
    req: Request, inf_in: ImagenRequest, db: Annotated[AsyncSession, Depends(get_db)]
):
    sdxl: DedicatedSDXLWorker = req.app.state.sdxl
    inf_param = {
        "prompt": inf_in.prompt,
        "negative_prompt": "bad quality"
        if not inf_in.negative_prompt
        else inf_in.negative_prompt,
        "steps": 28 if not inf_in.steps else inf_in.steps,
        "guidance_scale": 1.0 if not inf_in.cfg else inf_in.cfg,
        "width": 1024 if not inf_in.width else inf_in.width,
        "height": 1024 if not inf_in.height else inf_in.height,
        "seed": 42 if not inf_in.seed else inf_in.seed,
    }
    promise_id = str(uuid.uuid4())
    new_img = Image(promise_id=promise_id)
    db.add(new_img)
    await db.commit()
    sdxl.queue_inference(promise_id, inf_param)
    return {"promise_id": promise_id}


@router.get("/inferences/{promise_id}", response_model=InferenceResponse)
async def imagen_status(
    req: Request, promise_id: str, db: Annotated[AsyncSession, Depends(get_db)]
):
    stmt = select(Image).where(Image.promise_id == promise_id)
    result = await db.scalars(stmt)
    image_metadata: Image | None = result.first()
    if not image_metadata:
        return Response(status_code=404)
    sdxl: DedicatedSDXLWorker = req.app.state.sdxl
    live_inf_status = sdxl.get_inference_status(promise_id)
    if live_inf_status:
        image_metadata.steps_status = live_inf_status
    if not image_metadata.steps_status:
        image_metadata.steps_status = "{}"
    return image_metadata


@router.get("/{promise_id}")
async def get_image(
    req: Request, promise_id: str, db: Annotated[AsyncSession, Depends(get_db)]
):
    stmt = select(Image).where(Image.promise_id == promise_id)
    result = await db.scalars(stmt)
    image_metadata = result.first()
    if not image_metadata:
        return Response("No images found.", status_code=404)
    img_data = await retrieve_image(req.app, promise_id)
    if not img_data:
        return Response("Reason: Missing minio resource.", status_code=404)
    return Response(img_data, media_type="image/jpeg")
