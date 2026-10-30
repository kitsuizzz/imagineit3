from pydantic import BaseModel


class WorkerInfo(BaseModel):
    worker_id: str
    name: str
    model: str
    device: str
    status: str
    connected_seconds: int
    last_seen_seconds_ago: int
    has_active_job: bool


class WorkerListResponse(BaseModel):
    count: int
    workers: list[WorkerInfo]
