"""Authenticated image endpoints on the existing FastAPI application."""
import asyncio
from contextlib import contextmanager

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import Response
from fastapi.routing import APIRoute
from starlette.formparsers import MultiPartException

from app.business import DomainError
from app.db import uid
from app.observability import TraceRecorder, active_trace
from app.multimodal.retrieval import ByImage, ByImageAndText, ByText


ERROR_STATUS = {
    'IMAGE_NOT_FOUND': 404, 'IMAGE_TOO_LARGE': 413, 'IMAGE_QUOTA_EXCEEDED': 413,
    'IMAGE_INVALID_FORMAT': 422, 'IMAGE_DECODE_FAILED': 422, 'IMAGE_PIXEL_LIMIT_EXCEEDED': 422,
    'MULTIMODAL_DISABLED': 503, 'MULTIMODAL_MODEL_NOT_READY': 503, 'MULTIMODAL_ENCODER_ERROR': 503,
    'INDEX_NOT_READY': 503, 'VLM_IMAGE_SEND_DISABLED': 403, 'VLM_NOT_CONFIGURED': 503,
    'VLM_VISION_UNSUPPORTED': 503, 'VLM_TOOL_CALLING_UNSUPPORTED': 503,
}


@contextmanager
def image_trace(repo, principal, response, conversation_id=None):
    recorder = TraceRecorder(repo.business, principal, conversation_id, repo.config, invocation_id=uid('image-op-'))
    marker = active_trace.set(recorder)
    try:
        yield
    except DomainError as error:
        recorder.emit('error', status='error', error_code=error.code)
        error.trace_id = recorder.trace_id
        error.invocation_id = recorder.payload['invocation_id']
        raise
    finally:
        active_trace.reset(marker)
        recorder.finish()
        response.headers['X-Trace-ID'] = recorder.trace_id


def image_router(repo, index, actor, envelope):
    class UploadRoute(APIRoute):
        def get_route_handler(self):
            handler = super().get_route_handler()
            if self.methods != {'POST'}:
                return handler

            async def bounded(request):
                limit = repo.config.multimodal_max_image_bytes + 65536
                size = 0
                too_large = False

                async def receive():
                    nonlocal size, too_large
                    message = await request.receive()
                    size += len(message.get('body', b''))
                    if size > limit:
                        too_large = True
                        raise MultiPartException('Upload body limit exceeded')
                    return message

                try:
                    declared = request.headers.get('content-length')
                    if declared and declared.isdecimal() and int(declared) > limit:
                        raise DomainError('IMAGE_TOO_LARGE')
                    async with asyncio.timeout(repo.config.multimodal_upload_timeout_seconds):
                        return await handler(Request(request.scope, receive))
                except TimeoutError:
                    raise DomainError('TIMEOUT', '图片上传超时') from None
                except Exception:
                    if too_large:
                        raise DomainError('IMAGE_TOO_LARGE') from None
                    raise
            return bounded

    router = APIRouter(prefix='/api/v1/images', tags=['images'], route_class=UploadRoute)

    @router.post('', description='上传私有图片。聊天中另行授权后，图片才会发送给配置的外部 VLM。')
    async def upload(response: Response, file: UploadFile = File(), conversation_id: str | None = Form(None),
                     principal=Depends(actor)):
        limit = repo.config.multimodal_max_image_bytes
        data = bytearray()
        while chunk := await file.read(65536):
            data.extend(chunk)
            if len(data) > limit:
                raise DomainError('IMAGE_TOO_LARGE')
        with image_trace(repo, principal, response, conversation_id):
            result = await asyncio.to_thread(repo.save, bytes(data), file.filename, principal,
                                             mime=file.content_type, conversation_id=conversation_id)
        return envelope(result)

    @router.get('/{image_id}')
    def fetch(image_id: str, request: Request, principal=Depends(actor)):
        response = Response()
        with image_trace(repo, principal, response):
            row, data = repo.read(image_id, principal)
            etag = '"' + row['sha256_hex'] + '"'
            headers = {'Cache-Control': 'private, no-store', 'ETag': etag, 'X-Content-Type-Options': 'nosniff'}
            if request.headers.get('if-none-match') == etag:
                response = Response(status_code=304, headers=headers)
            else:
                response = Response(data, media_type=row['mime_type'], headers=headers)
        return response

    @router.delete('/{image_id}')
    def delete(image_id: str, response: Response, principal=Depends(actor)):
        with image_trace(repo, principal, response):
            result = repo.delete(image_id, principal)
            # A failed projection remains queued. SQLite revokes access first.
            if principal.role == 'admin':
                try:
                    index.sync('product_image', target_id=image_id)
                except Exception as error:
                    recorder = active_trace.get()
                    recorder.emit('multimodal.index.cleanup_deferred', error_type=type(error).__name__)
        return envelope(result)

    return router


def retrieval_router(service, actor, envelope):
    router = APIRouter(prefix='/api/v1/multimodal/search', tags=['multimodal'])

    def search(name, body, principal, response):
        with image_trace(service.repo, principal, response):
            items = service.search(name, principal, **body.model_dump())
        return envelope({'items': items, 'count': len(items)})

    @router.post('/by-image')
    def by_image(body: ByImage, response: Response, principal=Depends(actor)):
        return search('search_products_by_image', body, principal, response)

    @router.post('/by-text')
    def by_text(body: ByText, response: Response, principal=Depends(actor)):
        return search('search_images_by_text', body, principal, response)

    @router.post('/by-image-and-text')
    def by_image_and_text(body: ByImageAndText, response: Response, principal=Depends(actor)):
        return search('search_products_by_image_and_text', body, principal, response)

    @router.post('/text-by-image')
    def text_by_image(body: ByImage, response: Response, principal=Depends(actor)):
        return search('search_text_by_image', body, principal, response)

    return router
