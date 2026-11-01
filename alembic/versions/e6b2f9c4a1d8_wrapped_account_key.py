"""Store the token-wrapped private key on the account

Revision ID: e6b2f9c4a1d8
Revises: d4a7b1c8e2f6
Create Date: 2026-10-31 20:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e6b2f9c4a1d8"
down_revision: Union[str, Sequence[str], None] = "d4a7b1c8e2f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("accounts", sa.Column("wrapped_key", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("accounts", "wrapped_key")
