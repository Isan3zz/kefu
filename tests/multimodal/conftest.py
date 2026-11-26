from io import BytesIO

import numpy as np
from app.business import Actor

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from app.main import create_app
from app.mcp import Gateway, create_mcp
from app.retrieval_mcp import RetrievalService, create_retrieval_mcp


def picture(fmt='PNG', color='red', size=(80, 60)):
    image = Image.new('RGB', size, color)
    ImageDraw.Draw(image).ellipse((4, 4, 20, 20), fill='white')
    output = BytesIO()
    image.save(output, fmt)
    return output.getvalue()


@pytest.fixture
def mm(env, tmp_path):
    cfg = env['cfg'].model_copy(update={'multimodal_image_path': str(tmp_path/'images'),
                                      'chinese_clip_model_path': str(tmp_path/'missing-clip'),
                                      'multimodal_enabled': True})
    retrieval = RetrievalService(env['b'], cfg, rag=env['rag'])
    gateway = Gateway(env['b'], cfg, create_mcp(env['b'], cfg.business_secret, testing=True),
                      retrieval_server=create_retrieval_mcp(retrieval, cfg))
    app = create_app(cfg, env['db'], gateway, env['rag'])
    with TestClient(app) as client:
        tokens = {user: client.post('/api/v1/auth/demo-login', json={'demo_user_key': user}).json()['data']['token']
                  for user in ('user_a', 'user_b', 'admin')}
        yield dict(env, cfg=cfg, app=app, agent=app.state.agent, client=client, tokens=tokens, index=retrieval.index)


def auth(mm, user='user_a'):
    return {'Authorization': 'Bearer '+mm['tokens'][user]}


def upload(mm, content=None, name='shape.png', mime='image/png', user='user_a', **data):
    return mm['client'].post('/api/v1/images', headers=auth(mm, user),
                             files={'file': (name, picture() if content is None else content, mime)}, data=data)


class FakeMultimodalEncoder:
    revision = 'test-fake-colors-v1'

    def encode_images(self, images):
        return np.asarray([self.vector(np.argmax(image.getpixel((0, 0)))) for image in images])

    def encode_texts(self, texts):
        return np.asarray([self.vector(2 if '蓝' in text else 1 if '绿' in text else 0) for text in texts])

    @staticmethod
    def vector(channel):
        vector = np.zeros(512, dtype=np.float32)
        vector[channel] = 1
        return vector


@pytest.fixture
def indexed(mm):
    index = mm['index']
    index.encoder = FakeMultimodalEncoder()
    admin = Actor('admin', 'a', 'admin')
    row = index.repo.save(picture(), 'shape.png', admin, product_id='headphones-a')
    text_id = index.put_text(admin, '红色圆形', product_id='headphones-a')
    assert index.sync()['failed'] == 0
    return index, row, text_id
