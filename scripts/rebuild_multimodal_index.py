"""Rebuild or retry the local catalog index. Stop API writers before running."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.business import Business, DomainError
from app.config import settings
from app.db import DB
from app.multimodal.images import ImageRepository
from app.multimodal.index import MultimodalIndex
from app.rag import RAG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['all', 'product_image', 'visual_text'], default='all')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--retry', action='store_true')
    args = parser.parse_args()
    db = DB(settings.sqlite_url)
    if not args.dry_run:
        db.initialize()
    repo = ImageRepository(Business(db), settings)
    rag = RAG(db, config=settings)
    index = MultimodalIndex(repo, rag.client)
    try:
        report = index.sync(args.target) if args.retry else index.rebuild(args.target, args.dry_run)
        if not args.dry_run:
            report['expired_uploads'] = repo.expire()
        print(json.dumps(report, ensure_ascii=False))
        if report.get('failed'):
            parser.exit(1, 'Index work remains pending. Check error codes and run --retry.\n')
    except DomainError as error:
        parser.exit(1, f'{error.code}: {error.message}\n')
    finally:
        rag.close()


if __name__ == '__main__':
    main()
