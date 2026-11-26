"""Independent MCP service for read-only knowledge and multimodal searches."""
import asyncio
from threading import Lock
from time import perf_counter
from typing import Annotated

import anyio
from fastmcp import FastMCP
from pydantic import Field
from starlette.responses import JSONResponse

from app.business import Business, DomainError
from app.config import ROOT, settings
from app.db import DB, conversations, uid
from app.mcp import RETRIEVAL_AUDIENCE, RetrievalDiagnostics, RetrievalResponse, ToolFailure, request_credential, verify_request
from app.multimodal.images import ImageRepository
from app.multimodal.index import MultimodalIndex
from app.multimodal.retrieval import Filters, MultimodalRetrievalService
from app.observability import active_trace, retrieval_diagnostics
from app.rag import RAG


TopK = Annotated[int, Field(strict=True, ge=1, le=20)]
ImageID = Annotated[str, Field(strict=True, min_length=1, max_length=64)]
VisualQuery = Annotated[str, Field(strict=True, min_length=1, max_length=300)]
KnowledgeText = Annotated[str, Field(strict=True, min_length=1, max_length=1000)]
ProductID = Annotated[str, Field(strict=True, min_length=1, max_length=100)]
REBUILD_SCOPE = 'knowledge_rebuild'


class RetrievalService:
    def __init__(self, business, config, *, rag=None, index=None):
        self.business = business
        self.rag = rag if rag is not None else RAG(business.db, config=config)
        self.images = index.repo if index is not None else ImageRepository(business, config)
        self.index = index if index is not None else MultimodalIndex(self.images, self.rag.client)
        self.multimodal = MultimodalRetrievalService(self.index)
        self.index_lock = Lock()

    def search(self, name, token, secret, args):
        req = uid('tool-')
        start = perf_counter()
        diagnostics = {}
        scope = None
        trace_marker = active_trace.set(None)
        diagnostic_marker = retrieval_diagnostics.set(diagnostics)
        try:
            # An abandoned HTTP request may leave a worker running. The lock also
            # protects embedded indexes from an overlapping maintenance operation.
            with self.index_lock:
                principal, scope = verify_request(token, secret, self.business, audience=RETRIEVAL_AUDIENCE, tool=name)
                correlation = {key: scope.get(key) for key in ('invocation_id', 'operation_id', 'tool_call_id', 'attempt_no')}
                RetrievalDiagnostics(latency_ms=0, **correlation)
                data = self.search_scoped(name, principal, scope, args)
            return RetrievalResponse(ok=True, data=data, request_id=req,
                                     diagnostics=self.diagnostics(diagnostics, correlation, start, len(data)))
        except DomainError as error:
            info = self.diagnostics(diagnostics, correlation, start, 0) if scope is not None else None
            return RetrievalResponse(ok=False, request_id=req, diagnostics=info,
                                     error=ToolFailure(code=error.code, message=error.message,
                                                       retryable=error.code == 'UPSTREAM_UNAVAILABLE'))
        except (ValueError, TypeError):
            if scope is not None:
                return RetrievalResponse(ok=False, request_id=req,
                                         error=ToolFailure(code='INVALID_ARGUMENT', message='检索范围或参数不符合定义'))
            raise
        finally:
            retrieval_diagnostics.reset(diagnostic_marker)
            active_trace.reset(trace_marker)

    @staticmethod
    def diagnostics(values, correlation, start, count):
        return RetrievalDiagnostics(
            retrieval_candidate_count=values.get('candidate_count', 0),
            retrieval_result_count=count, model_revision=values.get('model_revision'),
            knowledge_snapshot=values.get('knowledge_snapshot'), stages=values.get('stages', {}),
            latency_ms=(perf_counter()-start)*1000, **correlation,
        )

    def search_scoped(self, name, principal, scope, args):
        conversation_id = scope.get('conversation_id')
        image_id = args.get('image_id')
        if image_id != scope.get('image_id'):
            raise DomainError('FORBIDDEN', '图片不在调用范围内')
        if scope.get('access') == 'agent' and conversation_id:
            with self.business.db.tx() as connection:
                self.business.owned(connection, conversations, principal, conversation_id)
                if image_id is not None:
                    row = self.images.resolve(connection, image_id, principal, 'retrieval', conversation_id=conversation_id)
                    if row['source_type'] != 'user_upload' or row['conversation_id'] != conversation_id:
                        raise DomainError('IMAGE_NOT_FOUND', '图片不存在或不可访问')
        elif scope.get('access') != 'direct_image' or name == 'search_knowledge' or conversation_id is not None:
            raise DomainError('FORBIDDEN', '检索请求缺少可信会话范围')
        if name != 'search_knowledge':
            # The standalone image API already authenticates the owner. If the
            # image has a conversation, still verify that the caller owns it.
            if image_id and conversation_id is None:
                with self.business.db.tx() as connection:
                    row = self.images.resolve(connection, image_id, principal, 'retrieval')
                    conversation_id = row['conversation_id']
                    if conversation_id is not None:
                        self.business.owned(connection, conversations, principal, conversation_id)
            return self.multimodal.search(name, principal, conversation_id,
                                          required_max_price=scope.get('required_max_price'), **args)

        doc_ids = None
        if args['product_id']:
            _, linked_docs = self.multimodal.knowledge_scope(principal, args['product_id'])
            doc_ids = linked_docs or None
        query = scope.get('rewritten_query') or args['query']
        try:
            return self.rag.search(query, principal.tenant_id, document_ids=doc_ids)
        except (FileNotFoundError, ValueError):
            raise DomainError('KNOWLEDGE_NOT_READY', '知识索引或本地模型暂不可用') from None
        except (OSError, RuntimeError):
            raise DomainError('UPSTREAM_UNAVAILABLE', '知识检索暂不可用') from None

    def rebuild(self, token, secret):
        principal, _ = verify_request(token, secret, self.business, audience=RETRIEVAL_AUDIENCE, tool=REBUILD_SCOPE)
        if principal.role != 'admin':
            raise DomainError('FORBIDDEN', '仅管理员可以重建知识索引')
        with self.index_lock:
            return self.rag.build(ROOT/'knowledge')


def create_retrieval_mcp(service, config=settings):
    server = FastMCP(RETRIEVAL_AUDIENCE)
    limiter = anyio.CapacityLimiter(1)
    secret = config.retrieval_mcp_service_secret

    async def execute(name, **args):
        token = request_credential()
        return await anyio.to_thread.run_sync(service.search, name, token, secret, args, limiter=limiter)

    @server.tool
    async def search_knowledge(query: KnowledgeText, product_id: ProductID | None = None) -> RetrievalResponse:
        """Search published knowledge for the authenticated customer."""
        return await execute('search_knowledge', query=query, product_id=product_id)

    @server.tool
    async def search_products_by_image(image_id: ImageID, top_k: TopK = 5) -> RetrievalResponse:
        """Find catalog images similar to an authorized image."""
        return await execute('search_products_by_image', image_id=image_id, top_k=top_k)

    @server.tool
    async def search_text_by_image(image_id: ImageID, top_k: TopK = 5) -> RetrievalResponse:
        """Find catalog text similar to an authorized image."""
        return await execute('search_text_by_image', image_id=image_id, top_k=top_k)

    @server.tool
    async def search_images_by_text(query: VisualQuery, top_k: TopK = 5,
                                    filters: Filters | None = None) -> RetrievalResponse:
        """Find catalog images by visual description."""
        return await execute('search_images_by_text', query=query, top_k=top_k,
                             filters=filters.model_dump() if filters else None)

    @server.tool
    async def search_products_by_image_and_text(image_id: ImageID, query: VisualQuery, top_k: TopK = 5,
                                                filters: Filters | None = None) -> RetrievalResponse:
        """Find catalog products using an authorized image and a visual description."""
        return await execute('search_products_by_image_and_text', image_id=image_id, query=query, top_k=top_k,
                             filters=filters.model_dump() if filters else None)

    @server.custom_route('/internal/knowledge/rebuild', methods=['POST'])
    async def rebuild(request):
        token = request.headers.get('x-internal-auth', '').removeprefix('Bearer ')
        try:
            result = await anyio.to_thread.run_sync(service.rebuild, token, secret, limiter=limiter)
        except DomainError as error:
            return JSONResponse({'code': error.code}, status_code=403)
        except (OSError, ValueError, RuntimeError):
            return JSONResponse({'code': 'KNOWLEDGE_NOT_READY'}, status_code=503)
        return JSONResponse(result)

    return server


async def main(config=settings):
    db = DB(config.sqlite_url)
    db.initialize()
    service = RetrievalService(Business(db), config)
    try:
        await create_retrieval_mcp(service, config).run_http_async(
            host=config.retrieval_mcp_host, port=config.retrieval_mcp_port, show_banner=False, stateless_http=True,
        )
    finally:
        service.rag.close()
        db.engine.dispose()


if __name__ == '__main__':
    asyncio.run(main())
