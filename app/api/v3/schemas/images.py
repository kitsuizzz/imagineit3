from pydantic import ConfigDict, Field
from pydantic.main import BaseModel


class ImagenRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    width: int | None = Field(default=None, ge=64, le=2048, multiple_of=8)
    height: int | None = Field(default=None, ge=64, le=2048, multiple_of=8)
    cfg: float | None = Field(default=None, ge=0.0, le=20.0)
    steps: int | None = Field(default=None, ge=1, le=150)
    seed: int | None = None
    clip_skip: int | None = Field(default=None, ge=1, le=12)
    scheduler: str | None = None


class InferenceResponse(BaseModel):
    promise_id: str
    status: str
    steps_status: str | None = None
    steps_completed: int = 0
    steps_total: int | None = None
    error: str | None = None

    model_config = ConfigDict(from_attributes=True)
