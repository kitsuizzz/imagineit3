from pydantic.main import BaseModel


class ImagenRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    width: int | None = None
    height: int | None = None
    cfg: float | None = None
    steps: int | None = None
    seed: int | None = None
