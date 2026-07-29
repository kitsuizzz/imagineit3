from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from .db import _Base as Base


class Image(Base):
    __tablename__ = "inferences"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    promise_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    steps_status: Mapped[str] = mapped_column(String(8), nullable=True)
