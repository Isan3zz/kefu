"""Chroma projections of the authoritative SQLite image and short-text records."""
import hashlib
from io import BytesIO
from pathlib import Path

from filelock import FileLock
from PIL import Image
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from app.business import Actor, DomainError
from app.db import documents, image_assets, multimodal_index_jobs, now, product_images, products, visual_text_records
from app.multimodal.encoder import DIMENSION, PREPROCESSING, local_encoder, normalized
from app.multimodal.images import queue_job
from app.observability import emit
from app.rag import atomic_json, visible


IMAGES = 'product_image'
TEXTS = 'visual_text'
JOBS = multimodal_index_jobs.c


class MultimodalIndex:
    def __init__(self, repo, client, encoder=None):
        self.repo, self.db, self.config = repo, repo.db, repo.config
        self.client = client
        self.encoder = encoder or local_encoder(str(Path(self.config.chinese_clip_model_path).resolve()),
                                                self.config.chinese_clip_device, self.config.chinese_clip_batch_size)
        self.names = {IMAGES: self.config.multimodal_chroma_images_collection,
                      TEXTS: self.config.multimodal_chroma_text_collection}
        self.path = Path(self.config.chroma_path)
        self.manifest = self.path/'multimodal_index_manifest.json'
        self.lock = FileLock(self.path/'multimodal-index.lock', timeout=15)

    def collection(self, kind, *, create=False):
        from chromadb.errors import NotFoundError
        client = self.client()
        try:
            return client.get_collection(self.names[kind], embedding_function=None)
        except NotFoundError:
            if not create:
                return None
            return client.get_or_create_collection(
                self.names[kind], embedding_function=None,
                configuration={'hnsw': {'space': 'cosine'}},
                metadata=self.model_metadata(),
            )

    def model_metadata(self):
        return dict(model_revision=self.encoder.revision, dimension=DIMENSION, preprocessing=PREPROCESSING)

    def check_model(self, collection):
        metadata = self.model_metadata()
        if any((collection.metadata or {}).get(key) != value for key, value in metadata.items()):
            raise DomainError('INDEX_NOT_READY', '模型或预处理版本已改变，请重建两个多模态集合')

    def write_manifest(self):
        counts = {}
        for kind, name in self.names.items():
            collection = self.collection(kind)
            counts[name] = collection.count() if collection is not None else 0
        metadata = {'model_revision': None, 'dimension': DIMENSION, 'preprocessing': PREPROCESSING}
        revisions = set()
        for kind in self.names:
            collection = self.collection(kind)
            if collection is not None:
                revisions.add(collection.metadata['model_revision'])
        if len(revisions) == 1:
            metadata['model_revision'] = revisions.pop()
        atomic_json(self.manifest, {**metadata, 'collections': counts, 'updated_at': now()})

    def put_text(self, principal, content, *, product_id=None, doc_id=None, key='main', source_version='1'):
        """Update one stable catalog label. No user upload enters either public index."""
        if principal.role != 'admin':
            raise DomainError('FORBIDDEN')
        if not content.strip() or len(content) > 240 or not (product_id or doc_id):
            raise DomainError('INVALID_ARGUMENT', '短文本须关联商品或知识文档，且最多 240 字')
        ident = hashlib.sha256(repr((principal.tenant_id, product_id, doc_id, key)).encode()).hexdigest()[:24]
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        row = dict(visual_text_id=ident, tenant_id=principal.tenant_id, product_id=product_id,
                   doc_id=doc_id, text_content=content, text_hash=content_hash,
                   source_version=source_version, status='ACTIVE')
        with self.db.tx() as connection:
            if product_id and not connection.execute(select(products.c.id).where(
                    products.c.id == product_id, products.c.tenant_id == principal.tenant_id)).first():
                raise DomainError('NOT_FOUND', '商品不存在')
            if doc_id:
                doc = connection.execute(select(documents).where(documents.c.id == doc_id)).mappings().first()
                if not doc or not visible(doc, principal.tenant_id):
                    raise DomainError('NOT_FOUND', '知识文档不可访问')
                row['source_version'] = doc['version']
            connection.execute(insert(visual_text_records).values(**row).on_conflict_do_update(
                index_elements=['visual_text_id'], set_={key: val for key, val in row.items() if key != 'visual_text_id'}))
            queue_job(connection, TEXTS, ident, 'UPSERT', self.version(TEXTS, row))
        return ident

    def delete_text(self, principal, ident):
        if principal.role != 'admin':
            raise DomainError('FORBIDDEN')
        table = visual_text_records
        with self.db.tx() as connection:
            row = connection.execute(select(table).where(table.c.visual_text_id == ident,
                                     table.c.tenant_id == principal.tenant_id)).mappings().first()
            if not row:
                raise DomainError('NOT_FOUND')
            connection.execute(table.update().where(table.c.visual_text_id == ident).values(status='DELETED'))
            queue_job(connection, TEXTS, ident, 'DELETE', self.version(TEXTS, row))

    @staticmethod
    def version(kind, row):
        return row['sha256_hex'] if kind == IMAGES else f"{row['source_version']}:{row['text_hash']}"

    def row(self, connection, kind, ident):
        if kind == IMAGES:
            statement = select(image_assets, product_images.c.product_id).join(product_images).where(
                image_assets.c.image_id == ident)
        else:
            statement = select(visual_text_records).where(visual_text_records.c.visual_text_id == ident)
        return connection.execute(statement).mappings().first()

    def project(self, job):
        kind, ident = job['target_kind'], job['target_id']
        with self.db.tx() as connection:
            row = self.row(connection, kind, ident)
        if not row or row['status'] != 'ACTIVE':
            collection = self.collection(kind)
            if collection is not None:
                collection.delete(ids=[ident])
            if kind == IMAGES and row and not self.repo.remove_file(row):
                raise OSError('Unable to remove stored image')
            emit('multimodal.index.deleted', target_id=ident, collection=self.names[kind])
            return
        if job['content_version'] != self.version(kind, row):
            return
        collection = self.collection(kind, create=True)
        self.check_model(collection)
        metadata = {key: row[key] for key in ('tenant_id', 'product_id') if row[key] is not None}
        metadata.update(self.model_metadata(), content_version=self.version(kind, row))
        if kind == IMAGES:
            metadata['image_id'] = ident
            principal = Actor('local-indexer', row['tenant_id'], 'admin')
            _, content = self.repo.read(ident, principal, purpose='index')
            with Image.open(BytesIO(content)) as image:
                vector = self.encoder.encode_images([image])
        else:
            metadata['visual_text_id'] = ident
            if row['doc_id'] is not None:
                metadata['doc_id'] = row['doc_id']
            vector = self.encoder.encode_texts([row['text_content']])
        collection.upsert(ids=[ident], embeddings=normalized(vector, 1).tolist(), metadatas=[metadata])
        emit('multimodal.index.upserted', target_id=ident, collection=self.names[kind],
             model_revision=metadata['model_revision'])

    def sync(self, target='all', *, target_id=None):
        self.path.mkdir(parents=True, exist_ok=True)
        with self.lock:
            return self.sync_locked(target, target_id=target_id)

    def sync_locked(self, target, *, target_id=None):
        kinds = tuple(self.names) if target == 'all' else (target,)
        statement = select(multimodal_index_jobs).where(JOBS.target_kind.in_(kinds), JOBS.status != 'DONE')
        if target_id is not None:
            statement = statement.where(JOBS.target_id == target_id)
        with self.db.tx() as connection:
            jobs = list(connection.execute(statement.order_by(JOBS.updated_at, JOBS.job_id)).mappings())
        report = {'done': 0, 'failed': 0, 'errors': {}}
        for job in jobs:
            where = JOBS.job_id == job['job_id']
            with self.db.tx() as connection:
                connection.execute(multimodal_index_jobs.update().where(where).values(
                    status='RUNNING', attempt_count=JOBS.attempt_count+1, updated_at=now()))
            code = None
            try:
                self.project(job)
            except Exception as error:
                code = error.code if isinstance(error, DomainError) else type(error).__name__
            with self.db.tx() as connection:
                # A concurrent identical request can requeue this job while it is running.
                connection.execute(multimodal_index_jobs.update().where(where, JOBS.status == 'RUNNING')
                                   .values(status='FAILED' if code else 'DONE', last_error=code, updated_at=now()))
            if code:
                emit('multimodal.index.failed', target_id=job['target_id'], collection=self.names[job['target_kind']],
                     error_code=code, status='error')
                report['failed'] += 1
                report['errors'][code] = report['errors'].get(code, 0) + 1
            else:
                report['done'] += 1
        if report['done'] and not report['failed']:
            self.write_manifest()
        return report

    def rebuild(self, target='all', dry_run=False):
        kinds = tuple(self.names) if target == 'all' else (target,)
        tables = {IMAGES: image_assets, TEXTS: visual_text_records}
        plans = {}
        with self.db.tx() as connection:
            for kind in kinds:
                table = tables[kind]
                query = select(table).where(table.c.status == 'ACTIVE')
                if kind == IMAGES:
                    query = query.join(product_images).where(table.c.source_type == 'product')
                plans[kind] = list(connection.execute(query).mappings())
        report = {'planned': {kind: len(rows) for kind, rows in plans.items()}, 'dry_run': dry_run}
        if dry_run:
            return report
        self.path.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if any(plans.values()):
                # Load before replacing any usable index. The application should be stopped for rebuilds.
                if hasattr(self.encoder, 'load'):
                    with self.encoder.lock:
                        self.encoder.load()
                if target != 'all':
                    for kind in self.names.keys() - set(kinds):
                        collection = self.collection(kind)
                        if collection is not None:
                            self.check_model(collection)
            with self.db.tx() as connection:
                for kind, rows in plans.items():
                    for row in rows:
                        ident = row['image_id' if kind == IMAGES else 'visual_text_id']
                        queue_job(connection, kind, ident, 'UPSERT', self.version(kind, row))
            for kind in kinds:
                if self.collection(kind) is not None:
                    self.client().delete_collection(self.names[kind])
            report.update(self.sync_locked(target))
            if not report['failed']:
                self.write_manifest()
                emit('multimodal.index.rebuilt', **report)
        return report

    def nearest(self, kind, tenant, vector, limit):
        collection = self.collection(kind)
        if collection is None or not collection.count():
            return []
        self.check_model(collection)
        scope = {'tenant_id': tenant}
        available = collection.get(where=scope, limit=limit, include=[])['ids']
        if not available:
            return []
        data = collection.query(query_embeddings=normalized([vector], 1).tolist(),
                                n_results=len(available), where=scope, include=['metadatas', 'distances'])
        return list(zip(data['ids'][0], data['metadatas'][0], data['distances'][0]))
