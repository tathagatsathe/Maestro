"""Add run_type and source_filename to runs.

Revision ID: 002
Revises: 001
Create Date: 2026-06-10

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column(
            "run_type",
            sa.String(length=32),
            nullable=False,
            server_default="research",
        ),
    )
    op.add_column(
        "runs",
        sa.Column("source_filename", sa.String(length=512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("runs", "source_filename")
    op.drop_column("runs", "run_type")
