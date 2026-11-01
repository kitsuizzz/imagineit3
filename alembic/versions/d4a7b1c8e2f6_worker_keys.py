"""Add per-account GPU worker keys

Revision ID: d4a7b1c8e2f6
Revises: c3f1a9d2b7e4
Create Date: 2026-10-31 19:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4a7b1c8e2f6"
down_revision: Union[str, Sequence[str], None] = "c3f1a9d2b7e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "worker_keys",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("account_id", sa.String(length=32), nullable=False),
        sa.Column("worker_id", sa.String(length=64), nullable=False),
        sa.Column("key_id", sa.String(length=32), nullable=False),
        sa.Column("secret", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("worker_id"),
    )
    op.create_index(
        op.f("ix_worker_keys_account_id"), "worker_keys", ["account_id"], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_worker_keys_account_id"), table_name="worker_keys")
    op.drop_table("worker_keys")
