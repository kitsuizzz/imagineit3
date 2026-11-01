import json
import uuid

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import select

from app.api.v3.deps import require_account
from app.api.v3.schemas.images import ImagenRequest, InferenceResponse
from app.database.db import db_session
from app.database.minio import retrieve_image
from app.database.models import Account, Image
from app.services import e2e, jobs
from app.services.worker_gateway import WorkerGateway

router = APIRouter()


@router.post("/inferences")
async def imagen(
    req: Request,
    inf_in: ImagenRequest,
    account: Account = Depends(require_account),
):
    inf_param = {
        "prompt": inf_in.prompt,
        "negative_prompt": inf_in.negative_prompt or "bad quality",
        "steps": 28 if inf_in.steps is None else inf_in.steps,
        "guidance_scale": 1.0 if inf_in.cfg is None else inf_in.cfg,
        "width": 1024 if inf_in.width is None else inf_in.width,
        "height": 1024 if inf_in.height is None else inf_in.height,
        # seed=0 is a valid seed and must not fall back to the default.
        "seed": 42 if inf_in.seed is None else inf_in.seed,
        "clip_skip": 2 if inf_in.clip_skip is None else inf_in.clip_skip,
        "scheduler": inf_in.scheduler or "euler",
    }
    promise_id = str(uuid.uuid4())
    sealed = e2e.seal_text(
        account.public_key, json.dumps(inf_param, separators=(",", ":"))
    )
    await jobs.create_job(promise_id, account.id, inf_param, sealed, inf_param["steps"])

    gateway: WorkerGateway | None = getattr(req.app.state, "gateway", None)
    if gateway is not None:
        gateway.wake.set()
    return {"promise_id": promise_id}


@router.get("/inferences/{promise_id}", response_model=InferenceResponse)
async def imagen_status(promise_id: str, account: Account = Depends(require_account)):
    job = await jobs.get_job(promise_id, account.id)
    if not job:
        return Response(status_code=404)
    if not job["steps_status"]:
        job["steps_status"] = f"0/{job['steps_total'] or 0}"
    return job


@router.get("")
async def list_images(account: Account = Depends(require_account), limit: int = 60):
    limit = max(1, min(int(limit), 200))
    async with db_session() as session:
        rows = (
            await session.scalars(
                select(Image)
                .where(Image.account_id == account.id, Image.status == "succeeded")
                .order_by(Image.created_at.desc(), Image.id.desc())
                .limit(limit)
            )
        ).all()
    return {
        "images": [
            {
                "promise_id": row.promise_id,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    }


@router.get("/{promise_id}")
async def get_image(
    req: Request,
    promise_id: str,
    account: Account = Depends(require_account),
):
    async with db_session() as session:
        stmt = select(Image).where(
            Image.promise_id == promise_id, Image.account_id == account.id
        )
        image_metadata = (await session.scalars(stmt)).first()
    if not image_metadata:
        return Response("No images found.", status_code=404)
    img_data = await retrieve_image(req.app, promise_id)
    if not img_data:
        return Response("Reason: Missing minio resource.", status_code=404)
    # Sealed envelope: the browser opens it with the account's private key.
    return Response(
        img_data,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
