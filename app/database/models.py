from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .db import _Base as Base


class Account(Base):
    """A user account.

    Only a hash of the bearer token is stored; the token itself is shown once
    at creation.  ``public_key`` is the P-256 point the browser generated, so
    the server can seal content to it but never open it.
    """

    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    public_key: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class KaggleCredential(Base):
    """Per-account Kaggle login, encrypted with the server's APP_SECRET.

    Browserful signs in on the user's behalf, so this cannot be sealed to the
    client key: the server has to read it to drive the session.
    """

    __tablename__ = "kaggle_credentials"

    account_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("accounts.id", ondelete="CASCADE"), primary_key=True
    )
    secret: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Image(Base):
    __tablename__ = "inferences"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    account_id: Mapped[str | None] = mapped_column(
        String(32),
        ForeignKey("accounts.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    promise_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="queued",
        server_default="queued",
        index=True,
    )
    steps_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    steps_completed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    steps_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    params_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attempt: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
