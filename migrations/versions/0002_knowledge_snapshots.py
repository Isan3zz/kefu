"""Make the active knowledge snapshot part of the same SQLite transaction as governance."""
from alembic import op
from sqlalchemy import Column,String,Text,inspect
revision='0002'
down_revision='0001'
branch_labels=None
depends_on=None
def upgrade():
    if not inspect(op.get_bind()).has_table('knowledge_snapshots'):
        op.create_table('knowledge_snapshots',Column('id',String,primary_key=True),Column('payload',Text,nullable=False))
def downgrade(): op.drop_table('knowledge_snapshots')
