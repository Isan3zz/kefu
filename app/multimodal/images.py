"""Normalize untrusted uploads and resolve image references under SQLite ACLs."""
from datetime import timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert

from app.business import DomainError
from app.db import conversations, image_assets, multimodal_index_jobs, product_images, products, now
from app.observability import emit


FORMATS = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp'}
EXTENSIONS = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.webp': 'WEBP'}
ASSETS = image_assets.c


def valid_id(value):
    try:
        return UUID(value).version == 4 and str(UUID(value)) == value
    except (TypeError, ValueError, AttributeError):
        return False


def queue_job(connection, kind, ident, operation, version):
    statement = insert(multimodal_index_jobs).values(
        job_id=str(uuid4()), target_kind=kind, target_id=ident, operation=operation,
        content_version=version, status='PENDING', updated_at=now(),
    )
    connection.execute(statement.on_conflict_do_update(
        index_elements=['target_kind', 'target_id', 'operation', 'content_version'],
        set_={'status': 'PENDING', 'updated_at': now(), 'last_error': None},
    ))


class ImageStorage:
    def __init__(self, config):
        self.config = config
        self.root = Path(config.multimodal_image_path).resolve()

    def normalize(self, content, filename, mime=None):
        if len(content) > self.config.multimodal_max_image_bytes:
            raise DomainError('IMAGE_TOO_LARGE', '图片文件超过体积限制')
        suffix = Path(filename or '').suffix.lower()
        if suffix not in EXTENSIONS:
            raise DomainError('IMAGE_INVALID_FORMAT', '支持 JPEG、PNG 和 WebP 图片')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(BytesIO(content)) as image:
                    if image.format != EXTENSIONS[suffix] or (mime and FORMATS[image.format] != mime):
                        raise DomainError('IMAGE_INVALID_FORMAT', '图片内容与声明格式不一致')
                    if image.width * image.height > self.config.multimodal_max_image_pixels:
                        raise DomainError('IMAGE_PIXEL_LIMIT_EXCEEDED', '图片像素超过限制')
                    if getattr(image, 'is_animated', False):
                        raise DomainError('IMAGE_INVALID_FORMAT', '仅支持静态图片')
                    rgb = ImageOps.exif_transpose(image).convert('RGB')
                    output = BytesIO()
                    rgb.save(output, format='JPEG', quality=90)
                    return output.getvalue(), rgb.size
        except (Image.DecompressionBombWarning, Image.DecompressionBombError):
            raise DomainError('IMAGE_PIXEL_LIMIT_EXCEEDED', '图片像素超过限制') from None
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            raise DomainError('IMAGE_DECODE_FAILED', '无法解码图片') from None

    @staticmethod
    def relative_path(ident, kind):
        directory = 'products' if kind == 'product' else 'uploads'
        return f'{directory}/{ident}.jpg'

    def path(self, row):
        ident = row['image_id']
        relative = self.relative_path(ident, row['source_type'])
        if not valid_id(ident) or row['file_rel_path'] != relative:
            raise DomainError('IMAGE_NOT_FOUND')
        path = self.root / relative
        if path.resolve() != path:
            raise DomainError('IMAGE_NOT_FOUND')
        return path

    def read(self, row):
        try:
            with self.path(row).open('rb') as source:
                content = source.read(row['byte_size'] + 1)
        except OSError:
            raise DomainError('IMAGE_NOT_FOUND', '图片文件不可用') from None
        if len(content) != row['byte_size'] or sha256(content).hexdigest() != row['sha256_hex']:
            raise DomainError('IMAGE_NOT_FOUND', '图片文件不可用')
        return content


class ImageRepository:
    def __init__(self, business, config):
        self.business, self.config = business, config
        self.db = business.db
        self.storage = ImageStorage(config)

    def resolve(self, connection, ident, principal, purpose, *, conversation_id=None,
                consent=False, include_deleted=False):
        row = None
        if valid_id(ident):
            row = connection.execute(select(image_assets).where(
                ASSETS.image_id == ident, ASSETS.tenant_id == principal.tenant_id,
            )).mappings().first()
        stamp = self.business.clock().isoformat()
        if (not row or (not include_deleted and (
                row['status'] != 'ACTIVE' or (row['expires_at'] and row['expires_at'] <= stamp)))
                or (row['source_type'] == 'user_upload' and row['owner_user_id'] != principal.user_id)
                or (conversation_id and row['conversation_id'] not in (None, conversation_id))
                or (purpose == 'delete' and row['source_type'] == 'product' and principal.role != 'admin')):
            emit('image.access_denied', purpose=purpose, status='blocked', error_code='IMAGE_NOT_FOUND')
            raise DomainError('IMAGE_NOT_FOUND', '图片不存在或不可访问')
        if row['source_type'] == 'product':
            product = connection.execute(select(products.c.id).join(product_images).where(
                product_images.c.image_id == ident, products.c.tenant_id == principal.tenant_id,
            )).first()
            if product is None:
                raise DomainError('IMAGE_NOT_FOUND')
        if purpose == 'vlm_send' and not (consent and self.config.multimodal_external_vlm_images_enabled):
            raise DomainError('VLM_IMAGE_SEND_DISABLED', '图片外发需要服务端启用并获得本次请求授权')
        return dict(row)

    def read(self, ident, principal, purpose='read', **scope):
        with self.db.tx() as connection:
            row = self.resolve(connection, ident, principal, purpose, **scope)
            return row, self.storage.read(row)

    def bind(self, principal, conversation_id, image_ids):
        if not image_ids:
            return
        with self.db.tx() as connection:
            self.business.owned(connection, conversations, principal, conversation_id)
            for ident in image_ids:
                row = self.resolve(connection, ident, principal, 'attachment', conversation_id=conversation_id)
                if row['source_type'] != 'user_upload':
                    raise DomainError('IMAGE_NOT_FOUND', '聊天附件必须为当前用户上传的图片')
                if not self.storage.path(row).is_file():
                    raise DomainError('IMAGE_NOT_FOUND', '图片文件不可用')
                if row['conversation_id'] is None:
                    connection.execute(image_assets.update().where(ASSETS.image_id == ident)
                                       .values(conversation_id=conversation_id))

    def save(self, content, filename, principal, *, mime=None, conversation_id=None, product_id=None,
             replace_id=None):
        data, (width, height) = self.storage.normalize(content, filename, mime)
        is_product = product_id is not None
        ident = str(uuid4())
        clock = self.business.clock()
        kind = 'product' if is_product else 'user_upload'
        row = dict(image_id=ident, tenant_id=principal.tenant_id,
                   owner_user_id=None if is_product else principal.user_id,
                   conversation_id=conversation_id, source_type=kind,
                   file_rel_path=self.storage.relative_path(ident, kind), mime_type='image/jpeg',
                   sha256_hex=sha256(data).hexdigest(), width_px=width, height_px=height,
                   byte_size=len(data), status='ACTIVE', created_at=clock.isoformat(),
                   expires_at=None if is_product else (clock+timedelta(seconds=self.config.multimodal_upload_ttl_seconds)).isoformat())
        path = self.storage.path(row)
        created = False
        try:
            with self.db.tx() as connection:
                if is_product:
                    product = connection.execute(select(products.c.id).where(
                        products.c.id == product_id, products.c.tenant_id == principal.tenant_id,
                    )).first()
                    if principal.role != 'admin' or not product:
                        raise DomainError('NOT_FOUND', '商品不存在或不可管理')
                    if replace_id is not None:
                        replaced = self.resolve(connection, replace_id, principal, 'delete')
                        linked = connection.execute(select(product_images.c.product_id).where(
                            product_images.c.image_id == replace_id)).scalar_one_or_none()
                        if linked != product_id:
                            raise DomainError('IMAGE_NOT_FOUND')
                        self.mark_deleted(connection, replaced)
                    prior = connection.execute(select(image_assets).join(product_images).where(
                        product_images.c.product_id == product_id, ASSETS.sha256_hex == row['sha256_hex'],
                        ASSETS.status == 'ACTIVE',
                    )).mappings().first()
                    if prior and self.storage.path(prior).is_file():
                        return self.public(prior)
                    if prior:
                        self.mark_deleted(connection, prior)
                else:
                    if conversation_id is not None:
                        self.business.owned(connection, conversations, principal, conversation_id)
                    # Expired and deleted files still count until maintenance removes them.
                    count, size = connection.execute(select(func.count(), func.coalesce(func.sum(ASSETS.byte_size), 0))
                        .where(ASSETS.tenant_id == principal.tenant_id, ASSETS.owner_user_id == principal.user_id,
                               ASSETS.byte_size > 0)).one()
                    if (count >= self.config.multimodal_upload_quota_count
                            or size + len(data) > self.config.multimodal_upload_quota_bytes):
                        raise DomainError('IMAGE_QUOTA_EXCEEDED', '上传配额已满，请删除旧图片或执行过期清理')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as target:
                    created = True
                    target.write(data)
                connection.execute(image_assets.insert().values(**row))
                if is_product:
                    connection.execute(product_images.insert().values(product_id=product_id, image_id=ident))
                    queue_job(connection, 'product_image', ident, 'UPSERT', row['sha256_hex'])
        except BaseException:
            if created:
                path.unlink(missing_ok=True)
            raise
        emit('image.uploaded', image_id=ident, byte_size=len(data), mime_type=row['mime_type'])
        return self.public(row)

    @staticmethod
    def public(row):
        ident = row['image_id']
        return dict(image_id=ident, mime_type=row['mime_type'], width=row['width_px'],
                    height=row['height_px'], size_bytes=row['byte_size'], status=row['status'],
                    url=f'/api/v1/images/{ident}')

    def mark_deleted(self, connection, row):
        connection.execute(image_assets.update().where(ASSETS.image_id == row['image_id'])
                           .values(status='DELETED', deleted_at=self.business.clock().isoformat()))
        kind = 'product_image' if row['source_type'] == 'product' else 'upload'
        queue_job(connection, kind, row['image_id'], 'DELETE', row['sha256_hex'])

    def delete(self, ident, principal):
        with self.db.tx() as connection:
            row = self.resolve(connection, ident, principal, 'delete', include_deleted=True)
            if row['status'] != 'DELETED':
                self.mark_deleted(connection, row)
        self.remove_file(row)
        emit('image.deleted', image_id=ident)
        return {'image_id': ident, 'status': 'DELETED'}

    def remove_file(self, row):
        try:
            self.storage.path(row).unlink(missing_ok=True)
        except (OSError, DomainError):
            return False
        with self.db.tx() as connection:
            connection.execute(image_assets.update().where(
                ASSETS.image_id == row['image_id'], ASSETS.status == 'DELETED').values(byte_size=0))
            if row['source_type'] == 'user_upload':
                jobs = multimodal_index_jobs.c
                connection.execute(multimodal_index_jobs.update().where(
                    jobs.target_id == row['image_id'], jobs.operation == 'DELETE',
                ).values(status='DONE', updated_at=now()))
        return True

    def expire(self):
        with self.db.tx() as connection:
            expired = list(connection.execute(select(image_assets).where(
                ASSETS.status == 'ACTIVE', ASSETS.expires_at <= self.business.clock().isoformat(),
            )).mappings())
            for row in expired:
                self.mark_deleted(connection, row)
            uploads = list(connection.execute(select(image_assets).where(
                ASSETS.source_type == 'user_upload', ASSETS.status == 'DELETED', ASSETS.byte_size > 0,
            )).mappings())
        for row in uploads:
            self.remove_file(row)
        return len(expired)
