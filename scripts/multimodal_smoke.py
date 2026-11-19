"""Separate opt-in probes for real local CLIP and real external VLM."""
import argparse
import asyncio
import json
from pathlib import Path
from secrets import choice
import sys
from tempfile import TemporaryDirectory
from time import monotonic

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from sqlalchemy import select

from app.business import Actor, Business, DomainError
from app.config import settings
from app.db import DB, image_assets, products, uid
from app.harness.execution_context import ExecutionContext, RunBudget
from app.harness.model_gateway import ModelGateway
from app.harness.telemetry import Telemetry
from app.multimodal.images import ImageRepository
from app.multimodal.index import IMAGES, TEXTS, MultimodalIndex
from app.multimodal.retrieval import MultimodalRetrievalService
from app.multimodal.vlm_image_adapter import VLMImageAdapter
from app.observability import TraceRecorder, active_trace
from app.rag import RAG
from scripts.generate_multimodal_test_assets import generate


ACTOR = Actor('user_a', 'a')
ADMIN = Actor('admin', 'a', 'admin')


def workspace(directory, config):
    root = Path(directory)
    cfg = config.model_copy(update={
        'sqlite_url': 'sqlite:///'+(root/'app.db').as_posix(),
        'multimodal_image_path': str(root/'images'), 'chroma_path': str(root/'chroma'),
        'checkpoint_db_path': str(root/'checkpoint.db'), 'multimodal_enabled': True,
    })
    db = DB(cfg.sqlite_url)
    db.seed()
    business = Business(db)
    repo = ImageRepository(business, cfg)
    return repo, business.conversation(ACTOR)['id'], generate(root/'fixtures')


def engineering_metrics(repo, index, events):
    with repo.db.tx() as connection:
        rows = list(connection.execute(select(image_assets)).mappings())
        active = {row['image_id'] for row in rows if row['status'] == 'ACTIVE'}
        missing = sum(not repo.storage.path(row).is_file() for row in rows if row['image_id'] in active)
        col = index.collection(IMAGES)
        projected = set(col.get(include=[])['ids'])
        text_collection = index.collection(TEXTS)
        return {
            'total_uploads': sum(row['source_type']=='user_upload' for row in rows),
            'indexed_images': len(projected), 'indexed_texts': text_collection.count(),
            'missing_files': missing, 'orphan_vectors': len(projected-active),
            'access_denials': sum(e['type']=='image.access_denied' for e in events),
            'query_duration_ms': [round(e['duration_ms'], 3) for e in events
                                  if e['type']=='multimodal.search.completed'],
            'image_encode_duration_ms': [round(e['duration_ms'], 3) for e in events
                                         if e['type']=='multimodal.encode.image'],
        }


def local_smoke(config=settings):
    with TemporaryDirectory(prefix='kefu-mm-local-') as directory:
        repo, conversation, pictures = workspace(directory, config)
        rag = RAG(repo.db, config=repo.config)
        recorder = TraceRecorder(repo.business, ACTOR, conversation, repo.config, invocation_id=uid('smoke-'))
        marker = active_trace.set(recorder)
        try:
            index = MultimodalIndex(repo, rag.client)
            service = MultimodalRetrievalService(index)
            with Image.open(pictures['red']) as image:
                image_vectors = index.encoder.encode_images([image])
            text_vectors = index.encoder.encode_texts(['红色背景白色圆形', '蓝色背景白色方块'])
            for vectors, size in ((image_vectors, 1), (text_vectors, 2)):
                assert vectors.shape == (size, 512) and vectors.dtype == np.float32
                assert np.isfinite(vectors).all()
                np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
            for color, title in [('red', '红色圆形'), ('blue', '蓝色方块'), ('green', '绿色三角形')]:
                product_id = 'fixture-'+color
                with repo.db.tx() as connection:
                    connection.execute(products.insert().values(
                        id=product_id, tenant_id='a', name=title, category='engineering_fixture',
                        price_cents=10000, stock=1, attributes_json='{}'))
                path = pictures[color]
                repo.save(path.read_bytes(), path.name, ADMIN, product_id=product_id)
                index.put_text(ADMIN, title, product_id=product_id)
            report = index.sync()
            assert report['failed'] == 0, report
            path = pictures['red']
            image_id = repo.save(path.read_bytes(), path.name, ACTOR, conversation_id=conversation)['image_id']
            requests = [
                ('search_products_by_image', {'image_id': image_id}),
                ('search_text_by_image', {'image_id': image_id}),
                ('search_images_by_text', {'query': '红色圆形'}),
                ('search_products_by_image_and_text', {'image_id': image_id, 'query': '红色圆形',
                                                       'filters': {'max_price_cents': 10000}}),
            ]
            for name, args in requests:
                hits = service.search(name, ACTOR, conversation, top_k=2, **args)
                assert len(hits) == 2 and all(hit['product_id'].startswith('fixture-') for hit in hits)
            return {'mode': 'real_local_engineering_smoke', 'model_revision': index.encoder.revision,
                    'image_shape': list(image_vectors.shape), 'text_shape': list(text_vectors.shape),
                    **engineering_metrics(repo, index, recorder.payload['events'])}
        finally:
            active_trace.reset(marker)
            recorder.finish()
            rag.close()
            repo.db.engine.dispose()


async def external_smoke(config=settings):
    with TemporaryDirectory(prefix='kefu-mm-vlm-') as directory:
        repo, conversation, pictures = workspace(directory, config)
        adapter = VLMImageAdapter(repo)
        adapter.check_capabilities(True, tools=True)
        selected = [choice(tuple(pictures)) for _ in range(2)]
        image_ids = [repo.save(pictures[label].read_bytes(), pictures[label].name, ACTOR,
                               conversation_id=conversation)['image_id'] for label in selected]
        ctx = ExecutionContext(uid('smoke-'), uid('trace-'), conversation, 'conv:'+conversation,
                               ACTOR, 'multimodal-smoke', monotonic()+config.agent_timeout_seconds)
        recorder = TraceRecorder(repo.business, ACTOR, conversation, repo.config,
                                 trace_id=ctx.trace_id, invocation_id=ctx.invocation_id)
        marker = active_trace.set(recorder)
        model = ModelGateway(repo.config, image_adapter=adapter)
        tool = {'type': 'function', 'function': {
            'name': 'record_backgrounds', 'description': 'Record the two background colors in attachment order.',
            'parameters': {'type': 'object',
                           'properties': {side: {'type': 'string', 'enum': list(pictures)}
                                          for side in ('first', 'second')},
                           'required': ['first', 'second'], 'additionalProperties': False},
        }}
        try:
            decision = await model.complete_with_tools(
                ctx=ctx, messages=[{'role': 'user', 'content':
                    'Inspect both attached pictures. Call record_backgrounds with their background colors in order.'}],
                image_ids=image_ids, image_consent=True, tool_schemas=[tool],
                budget=RunBudget(), telemetry=Telemetry(ctx),
            )
            calls = decision.message.tool_calls
            if (len(calls) != 1 or calls[0]['name'] != 'record_backgrounds'
                    or calls[0]['args'] != dict(zip(('first', 'second'), selected))):
                raise DomainError('VLM_SMOKE_FAILED', '未收到与实际图片背景匹配的原生工具调用')
            return {'mode': 'real_external_engineering_smoke', 'model': config.vlm_model,
                    'image_count': len(selected), 'native_tool_call': True,
                    'provider_request_bytes': [e['provider_request_bytes'] for e in recorder.payload['events']
                                               if e['type']=='vlm.request.completed']}
        finally:
            await model.close()
            active_trace.reset(marker)
            recorder.finish()
            repo.db.engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run-local-model', action='store_true')
    mode.add_argument('--run-external-vlm', action='store_true')
    args = parser.parse_args()
    try:
        report = local_smoke() if args.run_local_model else asyncio.run(external_smoke())
        print(json.dumps(report, ensure_ascii=False))
    except DomainError as error:
        parser.exit(1, f'{error.code}: {error.message}\n')


if __name__ == '__main__':
    main()
