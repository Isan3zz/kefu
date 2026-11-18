"""Add the independent memory store, consent and durable extraction batches."""
from alembic import op

from app.db import MEMORY_TABLES

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade():
    for table in MEMORY_TABLES:
        table.create(op.get_bind(), checkfirst=not op.get_context().as_sql)


def downgrade():
    for table in reversed(MEMORY_TABLES):
        table.drop(op.get_bind())
