"""Import local image files for an existing catalog product."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.business import Actor, Business, DomainError
from app.config import settings
from app.db import DB, products
from app.rag import RAG
from app.multimodal.index import MultimodalIndex
from app.multimodal.images import ImageRepository
from sqlalchemy import select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--product-id', required=True)
    parser.add_argument('--file', type=Path, required=True, action='append')
    parser.add_argument('--replace-id')
    parser.add_argument('--caption', help='Chinese short text describing the product image')
    parser.add_argument('--doc-id', help='Link the short text to a visible RAG document')
    parser.add_argument('--queue-only', action='store_true', help='Commit SQL and defer encoding to the retry command')
    args = parser.parse_args()
    if args.replace_id and len(args.file) != 1:
        parser.error('--replace-id requires exactly one --file')
    db = DB(settings.sqlite_url)
    db.initialize()
    repo = ImageRepository(Business(db), settings)
    with db.tx() as connection:
        product = connection.execute(select(products).where(products.c.id == args.product_id)).mappings().first()
    if product is None:
        parser.exit(1, 'Product not found\n')
    actor = Actor('local-image-import', product['tenant_id'], 'admin')
    rag = RAG(db, config=settings)
    index = MultimodalIndex(repo, rag.client)
    try:
        for path in args.file:
            with path.open('rb') as source:
                content = source.read(settings.multimodal_max_image_bytes + 1)
            print(repo.save(content, path.name, actor, product_id=args.product_id, replace_id=args.replace_id))
        index.put_text(actor, args.caption or product['name'], product_id=args.product_id, doc_id=args.doc_id)
        if not args.queue_only:
            report = index.sync()
            print(report)
            if report['failed']:
                parser.exit(1, 'Images saved; index jobs remain. Run rebuild_multimodal_index.py --retry.\n')
    except (OSError, DomainError) as error:
        parser.exit(1, f'{error}\n')
    finally:
        rag.close()


if __name__ == '__main__':
    main()
