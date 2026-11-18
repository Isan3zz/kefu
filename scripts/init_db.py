from app.config import settings
from app.db import DB
if __name__=='__main__':
    db=DB(settings.sqlite_url); db.seed(); print('Database initialized and demo data seeded (idempotent).')
