from datetime import timedelta
from io import BytesIO

from PIL import Image
from sqlalchemy import select
import pytest

from app.business import Actor
from app.db import image_assets, multimodal_index_jobs
from conftest import auth, picture, upload


@pytest.mark.parametrize(('fmt', 'suffix'), [('JPEG', 'jpg'), ('PNG', 'png'), ('WEBP', 'webp')])
def test_MM03_upload_normalizes_and_reads(mm, fmt, suffix):
    response = upload(mm, picture(fmt), 'shape.'+suffix, 'image/'+('jpeg' if suffix=='jpg' else suffix))
    assert response.status_code == 200, response.json()
    row = response.json()['data']
    image = mm['client'].get(row['url'], headers=auth(mm))
    assert image.headers['cache-control'] == 'private, no-store'
    assert mm['client'].get(row['url'], headers={**auth(mm), 'If-None-Match': image.headers['etag']}).status_code == 304
    with Image.open(BytesIO(image.content)) as decoded:
        assert decoded.format == 'JPEG' and decoded.mode == 'RGB' and not decoded.getexif()


def test_MM04_format_validation(mm):
    assert upload(mm, mime='image/jpeg').json()['code'] == 'IMAGE_INVALID_FORMAT'
    assert upload(mm, b'bad').json()['code'] == 'IMAGE_DECODE_FAILED'
    assert upload(mm, name='shape.png.svg').json()['code'] == 'IMAGE_INVALID_FORMAT'


def test_MM05_limits(mm, monkeypatch):
    cfg = mm['cfg']
    monkeypatch.setattr(cfg, 'multimodal_max_image_bytes', 20)
    assert upload(mm).status_code == 413
    assert upload(mm, b'x'*70000).status_code == 413
    monkeypatch.setattr(cfg, 'multimodal_max_image_bytes', 100000)
    monkeypatch.setattr(cfg, 'multimodal_max_image_pixels', 100)
    assert upload(mm).json()['code'] == 'IMAGE_PIXEL_LIMIT_EXCEEDED'
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 20)
    assert upload(mm).json()['code'] == 'IMAGE_PIXEL_LIMIT_EXCEEDED'


def test_MM06_traversal_and_sql_path(mm, tmp_path):
    row = upload(mm, name='../../safe.png').json()['data']
    repo = mm['agent'].images
    assert all(p.parent == tmp_path/'images'/'uploads' for p in repo.storage.root.rglob('*.jpg'))
    with mm['db'].tx() as c:
        c.execute(image_assets.update().values(file_rel_path='../../outside.jpg'))
    assert mm['client'].get(row['url'], headers=auth(mm)).status_code == 404


def test_MM07_private_read_delete_and_same_tenant_owner(mm):
    row = upload(mm).json()['data']
    for user in ('admin', 'user_b'):
        assert mm['client'].get(row['url'], headers=auth(mm, user)).status_code == 404
        assert mm['client'].delete(row['url'], headers=auth(mm, user)).status_code == 404


def test_MM10_delete_before_cleanup_and_expire(mm, monkeypatch):
    repo = mm['agent'].images
    row = upload(mm).json()['data']
    with monkeypatch.context() as patch:
        patch.setattr(repo, 'remove_file', lambda row: False)
        assert mm['client'].delete(row['url'], headers=auth(mm)).json()['data']['status'] == 'DELETED'
        assert mm['client'].get(row['url'], headers=auth(mm)).status_code == 404
        with mm['db'].tx() as c:
            assert c.execute(select(multimodal_index_jobs.c.status)).scalar_one() == 'PENDING'
    row2 = upload(mm).json()['data']
    clock = repo.business.clock()
    monkeypatch.setattr(repo.business, 'clock', lambda: clock + timedelta(days=2))
    assert repo.expire() == 1
    assert not list(repo.storage.root.rglob('*.jpg'))
    monkeypatch.setattr(repo.business, 'clock', lambda: clock)
    assert mm['client'].delete(row2['url'], headers=auth(mm)).status_code == 200


def test_debug_console_and_openapi(mm):
    response = mm['client'].get('/multimodal')
    assert response.status_code == 200 and '商品图文检索' in response.text
    assert "frame-ancestors 'none'" in response.headers['content-security-policy']
    schema = mm['client'].get('/openapi.json').json()
    assert '/api/v1/multimodal/search/by-image-and-text' in schema['paths']
    assert 'multipart/form-data' in schema['paths']['/api/v1/images']['post']['requestBody']['content']


def test_product_attachment_uses_existing_product(mm):
    repo = mm['agent'].images
    actor = Actor('admin', 'a', 'admin')
    row = repo.save(picture(), 'asset.png', actor, product_id='headphones-a')
    assert repo.save(picture(), 'asset.png', actor, product_id='headphones-a') == row
    assert mm['client'].get(row['url'], headers=auth(mm)).status_code == 200
    assert mm['client'].delete(row['url'], headers=auth(mm)).status_code == 404
    assert mm['client'].delete(row['url'], headers=auth(mm, 'admin')).status_code == 200
