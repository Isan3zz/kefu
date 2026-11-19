"""Add authorized image assets and durable multimodal indexing jobs."""
from alembic import op

from app.db import MULTIMODAL_TABLES

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade():
    for table in MULTIMODAL_TABLES:
        table.create(op.get_bind(), checkfirst=not op.get_context().as_sql)


def downgrade():
    for table in reversed(MULTIMODAL_TABLES):
        table.drop(op.get_bind())
