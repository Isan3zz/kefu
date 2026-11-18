"""Initial schema, frozen at release 0.1."""
from alembic import op
from pathlib import Path
revision='0001'
down_revision=None
branch_labels=None
depends_on=None
def upgrade():
    from sqlalchemy import inspect
    # This revision must not change when runtime SQLAlchemy metadata evolves.
    statements=Path(__file__).with_name('0001_schema.sql').read_text(encoding='utf-8').split(';')
    import re
    for statement in statements:
        if not statement.strip(): continue
        name=re.search(r'CREATE TABLE (\w+)',statement).group(1)
        if not inspect(op.get_bind()).has_table(name): op.get_bind().exec_driver_sql(statement)
def downgrade():
    import re
    text=Path(__file__).with_name('0001_schema.sql').read_text(encoding='utf-8')
    for name in reversed(re.findall(r'CREATE TABLE (\w+)',text)):
        op.drop_table(name)
