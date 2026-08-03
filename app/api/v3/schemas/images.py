from pydantic import ConfigDict
from pydantic.main import BaseModel


class ImagenRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    cfg: float | None = None
    steps: int | None = None
    seed: int | None = None
    clip_skip: int | None = None


class InferenceResponse(BaseModel):
    steps_status: str

    model_config = ConfigDict(from_attributes=True)
