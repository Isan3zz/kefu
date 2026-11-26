import numpy as np
import pytest
from sqlalchemy import select

from app.business import Actor, DomainError
from app.db import multimodal_index_jobs
from app.config import ROOT, Settings
from app.multimodal.encoder import DIMENSION, ChineseCLIPEncoder, normalized
from app.multimodal.index import IMAGES, TEXTS
from conftest import picture


ADMIN = Actor('admin', 'a', 'admin')


def test_MM01_missing_model_is_explicit(mm):
    example = Settings(_env_file=ROOT/'.env.example')
    assert example.chinese_clip_local_only and not example.multimodal_index_on_upload
    encoder = mm['index'].encoder
    with pytest.raises(DomainError, match='download_chinese_clip.py') as error:
        encoder.encode_texts(['红色'])
    assert isinstance(encoder, ChineseCLIPEncoder)
    assert error.value.code == 'MULTIMODAL_MODEL_NOT_READY'


def test_MM11_vector_contract():
    values = np.full((2, DIMENSION), 3, dtype=np.float64)
    vectors = normalized(values, 2)
    assert vectors.shape == (2, 512) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, rtol=1e-6)
    for invalid in (np.zeros((1, 512)), np.full((1, 512), np.nan),
                    np.full((1, 512), np.inf), np.ones((1, 1024))):
        with pytest.raises(DomainError, match='向量') as error:
            normalized(invalid, 1)
        assert error.value.code == 'MULTIMODAL_ENCODER_ERROR'


def test_MM12_collection_spaces_are_separate(indexed):
    index, row, text_id = indexed
    client = index.client()
    bge = client.create_collection('test_bge_space', embedding_function=None)
    bge.add(ids=['bge'], embeddings=[[1.0] * 1024])
    for kind, ident in ((IMAGES, row['image_id']), (TEXTS, text_id)):
        collection = index.collection(kind)
        assert collection.configuration['hnsw']['sync_threshold'] == 1
        item = collection.get(ids=[ident], include=['embeddings', 'metadatas'])
        assert item['embeddings'].shape == (1, 512)
        assert item['metadatas'][0]['model_revision'] == index.encoder.revision
    assert bge.get(include=['embeddings'])['embeddings'].shape == (1, 1024)
    assert index.rebuild(dry_run=True)['planned'] == {IMAGES: 1, TEXTS: 1}
    assert index.rebuild()['failed'] == 0
    assert client.get_collection('test_bge_space', embedding_function=None).count() == 1


def test_MM13_upsert_replace_delete_and_retry(indexed, monkeypatch):
    index, row, text_id = indexed
    assert index.repo.save(picture(), 'shape.png', ADMIN, product_id='headphones-a') == row
    assert index.put_text(ADMIN, '蓝色方形', product_id='headphones-a', source_version='2') == text_id
    replaced = index.repo.save(picture(color='blue'), 'shape.png', ADMIN, product_id='headphones-a',
                               replace_id=row['image_id'])
    with monkeypatch.context() as patch:
        patch.setattr(index.encoder, 'encode_images', lambda images: np.ones((1, 1024)))
        result = index.sync()
        assert result['errors'] == {'MULTIMODAL_ENCODER_ERROR': 1}
    assert index.sync()['failed'] == 0
    assert index.collection(IMAGES).get()['ids'] == [replaced['image_id']]
    assert index.collection(TEXTS).get()['metadatas'][0]['content_version'].startswith('2:')
    assert index.sync()['done'] == 0
    index.delete_text(ADMIN, text_id)
    index.repo.delete(replaced['image_id'], ADMIN)
    assert index.sync()['failed'] == 0
    assert index.collection(IMAGES).count() == index.collection(TEXTS).count() == 0
    with index.db.tx() as connection:
        assert set(connection.execute(select(multimodal_index_jobs.c.status)).scalars()) == {'DONE'}
