import argparse

from app.business import Business
from app.config import settings
from app.db import DB
from app.memory.service import MemoryService
from app.rag import RAG


def main():
    parser = argparse.ArgumentParser(description='Rebuild only the long-term memory collection from SQLite')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    db = DB(settings.sqlite_url)
    rag = RAG(db, config=settings)
    try:
        service = MemoryService(Business(db), rag, settings)
        count = service.rebuild_index(args.dry_run)
        print(f'{"Eligible" if args.dry_run else "Rebuilt"} memories: {count}')
    finally:
        rag.close()
        db.engine.dispose()


if __name__ == '__main__':
    main()
