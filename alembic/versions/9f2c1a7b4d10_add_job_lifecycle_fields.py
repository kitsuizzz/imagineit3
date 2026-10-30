"""Add job lifecycle fields for the app/GPU split

Revision ID: 9f2c1a7b4d10
Revises: 73690b2c941c
Create Date: 2026-10-30 16:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9f2c1a7b4d10"
down_revision: Union[str, Sequence[str], None] = "73690b2c941c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "inferences",
        sa.Column(
            "status", sa.String(length=16), nullable=False, server_default="queued"
        ),
    )
    op.add_column(
        "inferences",
        sa.Column("steps_completed", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("inferences", sa.Column("steps_total", sa.Integer(), nullable=True))
    op.add_column("inferences", sa.Column("params_json", sa.Text(), nullable=True))
    op.add_column("inferences", sa.Column("error", sa.Text(), nullable=True))
    op.add_column(
        "inferences", sa.Column("worker_id", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "inferences",
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "inferences",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "inferences",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.add_column(
        "inferences",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.alter_column(
        "inferences",
        "steps_status",
        existing_type=sa.String(length=8),
        type_=sa.String(length=32),
        existing_nullable=True,
    )
    op.create_index(
        op.f("ix_inferences_status"), "inferences", ["status"], unique=False
    )
    op.execute(
        "UPDATE inferences SET status='failed', "
        "error='Legacy job created before the app/GPU split has no stored parameters' "
        "WHERE params_json IS NULL"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_inferences_status"), table_name="inferences")
    op.alter_column(
        "inferences",
        "steps_status",
        existing_type=sa.String(length=32),
        type_=sa.String(length=8),
        existing_nullable=True,
    )
    op.drop_column("inferences", "updated_at")
    op.drop_column("inferences", "created_at")
    op.drop_column("inferences", "lease_expires_at")
    op.drop_column("inferences", "attempt")
    op.drop_column("inferences", "worker_id")
    op.drop_column("inferences", "error")
    op.drop_column("inferences", "params_json")
    op.drop_column("inferences", "steps_total")
    op.drop_column("inferences", "steps_completed")
    op.drop_column("inferences", "status")
