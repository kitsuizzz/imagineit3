"""Add accounts, per-user Kaggle credentials, and inference ownership

Revision ID: c3f1a9d2b7e4
Revises: 9f2c1a7b4d10
Create Date: 2026-10-31 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3f1a9d2b7e4"
down_revision: Union[str, Sequence[str], None] = "9f2c1a7b4d10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "accounts",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("public_key", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_table(
        "kaggle_credentials",
        sa.Column("account_id", sa.String(length=32), nullable=False),
        sa.Column("secret", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("account_id"),
    )
    op.add_column(
        "inferences", sa.Column("account_id", sa.String(length=32), nullable=True)
    )
    op.create_index(
        op.f("ix_inferences_account_id"), "inferences", ["account_id"], unique=False
    )
    op.create_foreign_key(
        "fk_inferences_account_id",
        "inferences",
        "accounts",
        ["account_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("fk_inferences_account_id", "inferences", type_="foreignkey")
    op.drop_index(op.f("ix_inferences_account_id"), table_name="inferences")
    op.drop_column("inferences", "account_id")
    op.drop_table("kaggle_credentials")
    op.drop_table("accounts")
