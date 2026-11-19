"""Build ephemeral, provider-specific image parts using only authorized image IDs."""
import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

from PIL import Image

from app.business import DomainError


@dataclass(frozen=True)
class PreparedImage:
    image_id: str
    content: bytes
    width: int
    height: int
    mime_type: str = 'image/jpeg'


class ProviderImageTransport(Protocol):
    def part(self, image: PreparedImage) -> dict: ...


class OpenAIChatCompletionsImageTransport:
    def part(self, image):
        uri = f'data:{image.mime_type};base64,' + base64.b64encode(image.content).decode('ascii')
        return {'type': 'image_url', 'image_url': {'url': uri}}


class VLMImageAdapter:
    def __init__(self, repo):
        self.repo, self.config = repo, repo.config
        self.transport: ProviderImageTransport = OpenAIChatCompletionsImageTransport()

    def check_capabilities(self, consent, *, tools):
        if not (self.config.multimodal_external_vlm_images_enabled and consent):
            raise DomainError('VLM_IMAGE_SEND_DISABLED', '图片外发需要服务端启用并获得本次请求授权')
        if not all((self.config.vlm_model, self.config.llm_base_url, self.config.llm_api_key)):
            raise DomainError('VLM_NOT_CONFIGURED', '请配置 VLM_MODEL、LLM_BASE_URL 和 LLM_API_KEY')
        if not self.config.vlm_supports_vision or self.config.vlm_provider_transport != 'data_uri':
            raise DomainError('VLM_VISION_UNSUPPORTED', '当前供应商尚未确认支持 Chat Completions 图片 data URI')
        if tools and not self.config.vlm_supports_tool_calling:
            raise DomainError('VLM_TOOL_CALLING_UNSUPPORTED', '当前 VLM 尚未确认支持图片与原生工具调用同时使用')

    def prepare(self, ctx, image_ids, *, consent):
        if not image_ids or len(image_ids) > self.config.multimodal_max_chat_images:
            raise DomainError('INVALID_ARGUMENT', '请求中的图片数量超出限制')
        # Read all attachments under one authorization snapshot, then compress outside the SQLite transaction.
        payloads = []
        with self.repo.db.tx() as connection:
            for ident in image_ids:
                ctx.remaining()
                row = self.repo.resolve(connection, ident, ctx.principal, 'vlm_send',
                                        conversation_id=ctx.conversation_id, consent=consent)
                payloads.append((ident, self.repo.storage.read(row)))
        images = []
        total = 0
        limit = self.config.multimodal_image_send_max_bytes
        for ident, content in payloads:
            ctx.remaining()
            image = self.compress(ident, content, limit-total)
            images.append(image)
            total += len(image.content)
        return images

    def compress(self, ident, content, limit):
        with Image.open(BytesIO(content)) as source:
            image = source.convert('RGB')
        edge = self.config.multimodal_image_send_max_edge
        image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        for quality in (85, 70, 50, 35):
            output = BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=True)
            if output.tell() <= limit:
                return PreparedImage(ident, output.getvalue(), image.width, image.height)
            edge = max(1, edge // 2)
            image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        raise DomainError('IMAGE_TOO_LARGE', '图片超过外发体积限制')

    def attach(self, messages, images):
        # ContextManager emits the current task as its first user message, before tool history.
        request = [dict(message) for message in messages]
        current = next(message for message in request if message['role'] == 'user')
        current['content'] = [
            {'type': 'text', 'text': current['content']},
            *(self.transport.part(image) for image in images),
        ]
        return request
