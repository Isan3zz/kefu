from alembic import context
from app.db import metadata, DB
from app.config import settings
target_metadata=metadata
if context.is_offline_mode():
    context.configure(url=settings.sqlite_url,target_metadata=metadata,literal_binds=True)
    with context.begin_transaction(): context.run_migrations()
else:
    db=DB(settings.sqlite_url)
    with db.engine.connect() as c:
        context.configure(connection=c,target_metadata=metadata)
        with context.begin_transaction(): context.run_migrations()
