import uuid

from fastapi import APIRouter, Request, BackgroundTasks

from app.api.v3.schemas.images import ImagenRequest
from imagen.sdxl import LoadedSDXL

router = APIRouter()


@router.post("/inference")
async def imagen(
    req: Request, inf_in: ImagenRequest, background_tasks: BackgroundTasks
):
    sdxl: LoadedSDXL = req.app.state.sdxl
    task_id = str(uuid.uuid4())
    background_tasks.add_task(
        sdxl.infer_and_wait,
        inf_in.prompt,
        "" if not inf_in.negative_prompt else inf_in.negative_prompt,
        1.0 if not inf_in.cfg else inf_in.cfg,
        28 if not inf_in.steps else inf_in.steps,
        42 if not inf_in.seed else inf_in.seed,
        1024 if not inf_in.width else inf_in.width,
        1024 if not inf_in.height else inf_in.height,
    )
    return {"success": True}
