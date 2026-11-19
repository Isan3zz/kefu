"""Search by CLIP vectors, then hydrate and authorize every candidate in SQLite."""
import asyncio
from hashlib import sha256
from io import BytesIO
from time import perf_counter

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.business import DomainError, PriceCents
from app.db import documents, image_assets, product_images, products, visual_text_records
from app.multimodal.encoder import normalized
from app.multimodal.index import IMAGES, TEXTS
from app.observability import emit
from app.rag import visible


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)
    top_k: int = Field(default=5, ge=1, le=20)


class Filters(BaseModel):
    model_config = SearchArgs.model_config
    max_price_cents: PriceCents | None = None
    category: str | None = Field(default=None, min_length=1, max_length=100)
    in_stock: bool | None = None


class ByImage(SearchArgs):
    image_id: str = Field(min_length=1, max_length=64)


class ByText(SearchArgs):
    query: str = Field(min_length=1, max_length=300)
    filters: Filters | None = None


class ByImageAndText(ByImage, ByText):
    pass


TOOL_ARGS = {
    'search_products_by_image': ByImage,
    'search_text_by_image': ByImage,
    'search_images_by_text': ByText,
    'search_products_by_image_and_text': ByImageAndText,
}


class MultimodalRetrievalService:
    def __init__(self, index):
        self.index, self.repo, self.db = index, index.repo, index.db

    def has_catalog(self, tenant, kind=IMAGES):
        table = image_assets if kind == IMAGES else visual_text_records
        query = select(table.c.status).where(table.c.tenant_id == tenant, table.c.status == 'ACTIVE')
        if kind == IMAGES:
            query = query.join(product_images)
        with self.db.tx() as connection:
            return connection.execute(query.limit(1)).first() is not None

    async def search_products_by_image(self, image_id, top_k=5, *, principal, conversation_id=None):
        return await self.run('search_products_by_image', principal, conversation_id, image_id=image_id, top_k=top_k)

    async def search_text_by_image(self, image_id, top_k=5, *, principal, conversation_id=None):
        return await self.run('search_text_by_image', principal, conversation_id, image_id=image_id, top_k=top_k)

    async def search_images_by_text(self, query, top_k=5, *, principal, filters=None, conversation_id=None):
        return await self.run('search_images_by_text', principal, conversation_id,
                              query=query, top_k=top_k, filters=filters)

    async def search_products_by_image_and_text(self, image_id, query, top_k=5, *,
                                               principal, filters=None, conversation_id=None):
        return await self.run('search_products_by_image_and_text', principal, conversation_id,
                              image_id=image_id, query=query, top_k=top_k, filters=filters)

    async def run(self, name, principal, conversation_id=None, *, required_max_price=None, **arguments):
        return await asyncio.to_thread(self.search, name, principal, conversation_id,
                                       required_max_price=required_max_price, **arguments)

    def search(self, name, principal, conversation_id=None, *, required_max_price=None, **arguments):
        if not self.repo.config.multimodal_enabled:
            raise DomainError('MULTIMODAL_DISABLED')
        try:
            args = TOOL_ARGS[name].model_validate(arguments)
        except ValueError:
            raise DomainError('INVALID_ARGUMENT', '检索参数不符合定义') from None
        kind = TEXTS if name == 'search_text_by_image' else IMAGES
        attrs = dict(query_type=name, top_k=args.top_k, collection=self.index.names[kind],
                     tenant_hash=sha256(principal.tenant_id.encode()).hexdigest()[:12])
        emit('multimodal.search.started', **attrs)
        start = perf_counter()
        try:
            image_bytes = None
            if isinstance(args, ByImage):
                _, image_bytes = self.repo.read(args.image_id, principal, 'retrieval',
                                                conversation_id=conversation_id)
            filters = self.filters(getattr(args, 'filters', None), required_max_price)
            hits, count = self.candidates(kind, principal, args, image_bytes, filters)
        except Exception as error:
            code = error.code if isinstance(error, DomainError) else 'INDEX_NOT_READY'
            emit('multimodal.search.failed', **attrs, status='error', error_code=code,
                 error_type=type(error).__name__, duration_ms=(perf_counter()-start)*1000)
            if isinstance(error, DomainError):
                raise
            raise DomainError(code, '图文索引暂不可用，请检查索引作业或重建索引') from None
        emit('multimodal.search.completed', **attrs, candidate_count=count, result_count=len(hits),
             duration_ms=(perf_counter()-start)*1000)
        return hits

    @staticmethod
    def filters(proposed, required_max_price):
        if required_max_price is None:
            return proposed
        values = proposed.model_dump() if proposed else {}
        ceiling = values.get('max_price_cents')
        values['max_price_cents'] = required_max_price if ceiling is None else min(ceiling, required_max_price)
        return Filters.model_validate(values)

    def candidates(self, kind, principal, args, image_bytes, filters):
        if not self.has_catalog(principal.tenant_id, kind):
            return [], 0
        encoder = self.index.encoder
        revision = encoder.revision
        collection = self.index.collection(kind)
        if collection is None or not collection.count():
            raise DomainError('INDEX_NOT_READY', '商品资料已登记，请运行多模态索引构建命令')
        self.index.check_model(collection)
        vectors = []
        if image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as image:
                vectors.append(normalized(encoder.encode_images([image]), 1)[0])
        if isinstance(args, ByText):
            vectors.append(normalized(encoder.encode_texts([args.query]), 1)[0])
        vector = vectors[0]
        if len(vectors) == 2:
            weight = self.repo.config.multimodal_image_weight
            vector = normalized([weight*vector+(1-weight)*vectors[1]], 1)[0]
        candidates = self.index.nearest(kind, principal.tenant_id, vector, max(20, args.top_k*4))
        if not candidates:
            raise DomainError('INDEX_NOT_READY', '当前租户的多模态索引尚未就绪')
        emit('multimodal.search.candidates', model_revision=revision, collection=self.index.names[kind],
             candidate_count=len(candidates))
        hits = []
        seen = set()
        with self.db.tx() as connection:
            for ident, metadata, distance in candidates:
                row = self.index.row(connection, kind, ident)
                if (not row or row['status'] != 'ACTIVE' or row['tenant_id'] != principal.tenant_id
                        or metadata.get('content_version') != self.index.version(kind, row)
                        or metadata.get('model_revision') != revision):
                    continue
                score = max(-1.0, min(1.0, 1.0-float(distance)))
                if kind == IMAGES:
                    if row['source_type'] != 'product' or row['product_id'] in seen:
                        continue
                    hit = self.product_hit(connection, row, principal, score, filters)
                    if hit:
                        seen.add(row['product_id'])
                else:
                    hit = self.text_hit(connection, row, principal, score)
                if hit:
                    hits.append(hit)
                    if len(hits) == args.top_k:
                        break
        return hits, len(candidates)

    def product(self, connection, tenant, product_id, filters=None):
        table = products.c
        condition = [table.id == product_id, table.tenant_id == tenant]
        if filters:
            if filters.max_price_cents is not None:
                condition.append(table.price_cents <= filters.max_price_cents)
            if filters.category is not None:
                condition.append(table.category == filters.category)
            if filters.in_stock is not None:
                condition.append(table.stock > 0 if filters.in_stock else table.stock <= 0)
        return connection.execute(select(products).where(*condition)).mappings().first()

    def document(self, connection, principal, row):
        doc = connection.execute(select(documents).where(documents.c.id == row['doc_id'])).mappings().first()
        return doc if doc and doc['version'] == row['source_version'] and visible(
            doc, principal.tenant_id, self.repo.business.clock()) else None

    def product_hit(self, connection, row, principal, score, filters=None):
        try:
            self.repo.resolve(connection, row['image_id'], principal, 'retrieval')
            if self.repo.storage.path(row).stat().st_size != row['byte_size']:
                return None
        except (DomainError, OSError):
            return None
        product = self.product(connection, principal.tenant_id, row['product_id'], filters)
        if product is None:
            return None
        return dict(image_id=row['image_id'], product_id=product['id'], product_name=product['name'],
                    image_url=self.repo.public(row)['url'], price_cents=product['price_cents'],
                    currency='CNY', stock=product['stock'], category=product['category'],
                    similarity_score=score, score_kind='cosine_similarity', retrieval_source=self.index.names[IMAGES])

    def text_hit(self, connection, row, principal, score):
        if row['product_id'] and self.product(connection, principal.tenant_id, row['product_id']) is None:
            return None
        if row['doc_id'] and not self.document(connection, principal, row):
            return None
        return dict(visual_text_id=row['visual_text_id'], product_id=row['product_id'], doc_id=row['doc_id'],
                    content=row['text_content'], score=score, score_kind='cosine_similarity', source=self.index.names[TEXTS])

    def knowledge_scope(self, principal, product_id):
        texts = visual_text_records.c
        with self.db.tx() as connection:
            product = self.product(connection, principal.tenant_id, product_id)
            if product is None:
                raise DomainError('NOT_FOUND', '商品已不可访问')
            rows = connection.execute(select(visual_text_records).where(
                texts.tenant_id == principal.tenant_id, texts.product_id == product_id,
                texts.status == 'ACTIVE', texts.doc_id.is_not(None))).mappings()
            return product['name'], {row['doc_id'] for row in rows if self.document(connection, principal, row)}

    def revalidate_cards(self, principal, observations, required_max_price=None):
        results = {}
        with self.db.tx() as connection:
            for event in observations:
                if event['name'] not in TOOL_ARGS or event['name'] == 'search_text_by_image' or not event['result']['ok']:
                    continue
                proposed = Filters.model_validate(event['args']['filters']) if event['args'].get('filters') else None
                filters = self.filters(proposed, required_max_price)
                for hit in event['result']['data']:
                    row = self.index.row(connection, IMAGES, hit['image_id'])
                    if row and row['tenant_id'] == principal.tenant_id:
                        card = self.product_hit(connection, row, principal, hit['similarity_score'], filters)
                        if card:
                            results.setdefault(card['product_id'], card)
        return list(results.values())[:20]
