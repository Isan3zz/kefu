"""Exercise the unified gateway over actual MCP sessions and HTTP processes."""
import asyncio
from contextlib import ExitStack, contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from fastmcp import Client
from langchain_core.messages import AIMessage
import pytest
from sqlalchemy import select

from app.agent import Agent
from app.business import Actor, DomainError
from app.db import documents, traces
from app.harness.tool_protocol import ModelDecision
from app.mcp import AUTH_METADATA, BUSINESS_AUDIENCE, RETRIEVAL_AUDIENCE, Gateway, create_mcp, decode_tool_result, sign_actor
from app.retrieval_mcp import RetrievalService, create_retrieval_mcp


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT/'tests/fixtures/multimodal/circle.png'
QUERY = '耳机保修期限'
RETRIEVAL_PROCESS = """
import sys,time
from pathlib import Path
sys.path.insert(0, 'tests/multimodal')
from conftest import FakeMultimodalEncoder
from app.config import settings
from app.business import Business
from app.db import DB
from app.fakes import FakeModels
from app.rag import RAG
from app.retrieval_mcp import RetrievalService, create_retrieval_mcp
business = Business(DB(settings.sqlite_url))
rag = RAG(business.db, FakeModels(), settings)
search = rag.search
def search_with_delay(*args, **kwargs):
    flag = Path(settings.chroma_path).parent/'delay-once'
    if flag.exists():
        flag.unlink()
        time.sleep(1.25)
    return search(*args, **kwargs)
rag.search = search_with_delay
service = RetrievalService(business, settings, rag=rag)
service.index.encoder = FakeMultimodalEncoder()
create_retrieval_mcp(service, settings).run(transport='http', host=settings.retrieval_mcp_host,
    port=settings.retrieval_mcp_port, stateless_http=True, show_banner=False)
"""


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@contextmanager
def process(args, variables, port, log_path):
    with log_path.open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, *args], env=variables, cwd=ROOT, stdout=log, stderr=log)
        try:
            deadline = time.monotonic()+30
            while time.monotonic() < deadline and child.poll() is None:
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.2):
                        break
                except OSError:
                    time.sleep(.05)
            else:
                pytest.fail(f'MCP failed to start; see {log_path}')
            yield child
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)


def network_config(env, tmp_path):
    ports = {kind: free_port() for kind in ('business', 'retrieval')}
    update = {'multimodal_image_path': str(tmp_path/'images'),
              'memory_chroma_path': str(tmp_path/'memory'), 'memory_job_mode': 'sync',
              'retrieval_mcp_service_secret': 'fixture-retrieval-secret',
              'business_mcp_service_secret': 'fixture-business-secret'}
    for kind, port in ports.items():
        update[f'{kind}_mcp_port'] = port
        update[f'{kind}_mcp_url'] = f'http://127.0.0.1:{port}/mcp'
    config = env['cfg'].model_copy(update=update)
    names = ('sqlite_url', 'chroma_path', 'bm25_index_path', 'multimodal_image_path', 'memory_chroma_path',
             'business_mcp_url', 'business_mcp_service_secret', 'business_mcp_host', 'business_mcp_port',
             'retrieval_mcp_url', 'retrieval_mcp_service_secret', 'retrieval_mcp_host', 'retrieval_mcp_port')
    variables = os.environ | {name.upper(): str(getattr(config, name)) for name in names}
    return config, variables, ports


def build_catalog(service, principal):
    admin = Actor('admin', 'a', 'admin')
    catalog = service.images.save(IMAGE.read_bytes(), 'catalog.png', admin, product_id='headphones-a')
    service.index.put_text(admin, '红色耳机保修', product_id='headphones-a', doc_id='warranty')
    assert service.index.sync()['failed'] == 0
    uploaded = service.images.save(IMAGE.read_bytes(), 'query.png', principal)
    return catalog['image_id'], uploaded['image_id']


@pytest.fixture
def http_services(env, tmp_path):
    import runpy
    # The fake replaces the encoders only. Production indexes and authorization still run.
    fake = runpy.run_path(str(ROOT/'tests/multimodal/conftest.py'))['FakeMultimodalEncoder']
    config, variables, ports = network_config(env, tmp_path)
    service = RetrievalService(env['b'], config, rag=env['rag'])
    service.rag.build(ROOT/'knowledge')
    service.index.encoder = fake()
    catalog_id, image_id = build_catalog(service, env['a'])
    baseline = service.rag.search(QUERY, env['a'].tenant_id)
    service.rag.close()
    with ExitStack() as stack:
        business = stack.enter_context(process(
            ['-m', 'app.mcp'], variables, ports['business'], tmp_path/'business.log'))
        retrieval = stack.enter_context(process(
            ['-c', RETRIEVAL_PROCESS], variables, ports['retrieval'], tmp_path/'retrieval.log'))
        yield config, service, catalog_id, image_id, business, retrieval, baseline


def invocation(env, principal=None, conversation=None):
    controller = env['agent'].harness.controller
    ident = conversation or env['conv']
    return controller.begin(principal or env['a'], ident, 'conv:'+ident, {})[0]


async def test_http_discovery_reuse_and_cross_service_auth(env, http_services):
    config, *_ = http_services
    gateway = Gateway(env['b'], config)
    try:
        business, retrieval = await asyncio.gather(gateway.discover('business'), gateway.discover('retrieval'))
        assert len(business) == 7 and len(retrieval) == 5
        assert 'submit_refund_request' not in business | retrieval
        assert all(not {'tenant_id', 'user_id', 'conversation_id', 'file_path'} & schema[0]['properties'].keys()
                   for schema in retrieval.values())
        first_session = gateway.servers['business'].client
        results = await asyncio.gather(
            gateway.call(env['a'], 'get_order', {'order_id': 'O1001'}, 'order-tracking'),
            gateway.call(env['other'], 'get_order', {'order_id': 'O1001'}, 'order-tracking'),
        )
        assert results[0]['ok'] and results[1]['error']['code'] == 'NOT_FOUND'
        assert gateway.servers['business'].client is first_session
        for endpoint, tool, args, wrong_aud, actual_secret, wrong_secret, own_aud in (
            (config.business_endpoint, 'get_orders', {}, RETRIEVAL_AUDIENCE, config.business_secret,
             config.retrieval_mcp_service_secret, BUSINESS_AUDIENCE),
            (config.retrieval_mcp_url, 'search_knowledge', {'query': QUERY}, BUSINESS_AUDIENCE,
             config.retrieval_mcp_service_secret, config.business_secret, RETRIEVAL_AUDIENCE),
        ):
            cases = [(wrong_aud, actual_secret, tool, None), (own_aud, wrong_secret, tool, None),
                     (own_aud, actual_secret, 'another_tool', None),
                     (own_aud, actual_secret, tool, time.time()-1)]
            async with Client(endpoint) as client:
                for aud, secret, bound_tool, exp in cases:
                    token = sign_actor(env['a'], secret, expires=exp, audience=aud, scope={'tool': bound_tool})
                    raw = await client.call_tool(tool, args, meta={AUTH_METADATA: token})
                    assert decode_tool_result(raw)['error']['code'] == 'FORBIDDEN'
                if tool == 'search_knowledge':
                    unsigned = await client.call_tool(tool, args)
                    assert decode_tool_result(unsigned)['error']['code'] == 'FORBIDDEN'
                    invalid = await client.call_tool(tool, args | {'user_id': 'admin'}, raise_on_error=False)
                    assert invalid.is_error
        ctx = invocation(env)
        before = await gateway.call(env['a'], 'search_knowledge', {'query': QUERY}, context=ctx)
        with pytest.raises(DomainError) as denied:
            await gateway.rebuild_knowledge(env['a'])
        assert denied.value.code == 'FORBIDDEN'
        snapshot = await gateway.rebuild_knowledge(Actor('admin', 'a', 'admin'))
        after = await gateway.call(env['a'], 'search_knowledge', {'query': QUERY}, context=ctx)
        assert after['diagnostics']['knowledge_snapshot'] == snapshot['snapshot']
        assert after['diagnostics']['knowledge_snapshot'] != before['diagnostics']['knowledge_snapshot']
    finally:
        await gateway.close()


async def test_http_rag_and_four_modalities_with_sqlite_authority(env, http_services):
    config, service, catalog_id, image_id, _, _, before = http_services
    actor = env['a']
    ctx = invocation(env)
    service.images.bind(actor, ctx.conversation_id, [image_id])
    gateway = Gateway(env['b'], config)
    try:
        result = await gateway.call(actor, 'search_knowledge', {'query': QUERY}, context=ctx)
        assert result['ok']
        assert [(r['chunk_id'], r['version']) for r in before] == [(r['chunk_id'], r['version']) for r in result['data']]
        assert set(result['diagnostics']['stages']) == {'dense_ms', 'bm25_ms', 'rrf_ms', 'rerank_ms'}
        for name, args in (
            ('search_products_by_image', {'image_id': image_id}),
            ('search_text_by_image', {'image_id': image_id}),
            ('search_images_by_text', {'query': '红色'}),
            ('search_products_by_image_and_text', {'query': '红色', 'image_id': image_id}),
        ):
            result = await gateway.call(actor, name, args, context=ctx)
            assert result['ok'] and result['diagnostics']['retrieval_candidate_count'] >= 1
            assert result['data'][0].get('image_id', catalog_id) == catalog_id
            assert result['data'][0]['product_id'] == 'headphones-a'
        new_conversation = env['b'].conversation(actor)['id']
        foreign_conversation = env['b'].conversation(env['other'])['id']
        for outsider in (invocation(env, actor, new_conversation), invocation(env, env['other'], foreign_conversation)):
            attempt = await gateway.call(outsider.principal, 'search_products_by_image', {'image_id': image_id}, context=outsider)
            assert not attempt['ok'] and attempt['error']['code'] == 'IMAGE_NOT_FOUND'
        with env['db'].tx() as c:
            c.execute(documents.update().where(documents.c.id=='warranty').values(
                expires_at=(env['b'].clock()-timedelta(days=1)).isoformat()))
            c.execute(documents.update().where(documents.c.id.in_([r['doc_id'] for r in before if r['doc_id']!='warranty']))
                      .values(tenant_id='b'))
        result = await gateway.call(actor, 'search_knowledge', {'query': QUERY}, context=ctx)
        assert not {r['doc_id'] for r in before} & {r['doc_id'] for r in result['data']}
    finally:
        await gateway.close()


@pytest.mark.parametrize(('variant', 'expected'), [
    ('unknown', 'MCP_UNKNOWN_TOOL'), ('conflict', 'MCP_TOOL_CONFLICT'), ('schema', 'MCP_SCHEMA_MISMATCH'),
])
async def test_discovery_rejects_untrusted_catalog(env, variant, expected):
    catalog = create_retrieval_mcp(None, env['cfg'])
    if variant == 'schema':
        catalog.remove_tool('search_knowledge')

    def unexpected(query: int) -> str:
        return str(query)

    name = {'unknown': 'extra_tool', 'conflict': 'get_order', 'schema': 'search_knowledge'}[variant]
    catalog.tool(unexpected, name=name)
    gateway = Gateway(env['b'], env['cfg'], create_mcp(env['b'], env['cfg'].business_secret, testing=True),
                      retrieval_server=catalog)
    with pytest.raises(DomainError) as failure:
        await gateway.discover('retrieval')
    assert failure.value.code == expected
    assert 'get_order' in await gateway.discover('business')


async def test_http_timeout_isolation_correlation_and_multiturn(env, tmp_path, http_services, monkeypatch):
    config, _, _, image_id, _, retrieval, _ = http_services
    gateway = Gateway(env['b'], config)
    agent = Agent(env['b'], gateway, config)
    principal, conv = env['a'], env['conv']
    try:
        await gateway.discover('retrieval')
        assert (await gateway.call(principal, 'search_knowledge', {'query': QUERY}, context=invocation(env)))['ok']
        await agent.chat(principal, conv, '耳机库存', 'deterministic')
        gateway.servers['retrieval'].timeout = 1.0
        (tmp_path/'delay-once').touch()
        answer = await agent.chat(principal, conv, '它保修多久', 'deterministic')
        assert answer['citations'] and answer['tool_call_count'] == 1
        with env['db'].tx() as c:
            trace = json.loads(c.scalar(select(traces.c.payload).where(traces.c.id==answer['trace_id'])))
        attempts = [event for event in trace['events'] if event['type']=='tool' and event.get('layer')=='transport']
        assert [(event['attempt_no'], event['error_code']) for event in attempts] == [(1,'TOOL_TIMEOUT'), (2,None)]
        execution = next(event for event in trace['events'] if event['type']=='retrieval.execution')
        evidence = next(event for event in trace['events'] if event['type']=='retrieval')
        assert execution['invocation_id'] == answer['invocation_id']
        assert execution['tool_call_id'] == attempts[0]['tool_call_id'] == evidence['tool_call_id']
        assert execution['operation_id'] == attempts[0]['operation_id'] == evidence['operation_id']
        assert execution['knowledge_snapshot'] == evidence['chunks'][0]['snapshot']
        assert 'fixture-retrieval-secret' not in json.dumps(trace)
        assert any(event['type']=='query_rewrite' for event in trace['events'])
        gateway.servers['retrieval'].timeout = config.tool_timeout_seconds

        calls = iter([
            [('search_products_by_image', {'image_id': image_id})],
            [('search_knowledge', {'query': QUERY, 'product_id': 'headphones-a'})],
            [],
        ])

        async def scripted(**unused):
            batch = next(calls)
            proposal = [{'name': name, 'args': args, 'id': f'protocol-{name}'} for name, args in batch]
            return ModelDecision(AIMessage(content='', tool_calls=proposal), 'fixture')

        with monkeypatch.context() as fake:
            fake.setattr(agent.harness.fake_model, 'complete_with_tools', scripted)
            combined = await agent.harness.invoke_chat(principal=principal, conversation_id=conv,
                message='找这个商品的保修资料', image_ids=[image_id], mode='deterministic')
        assert combined['cards'][0]['product_id'] == 'headphones-a'
        assert {row['doc_id'] for row in combined['citations']} == {'warranty'}

        gateway.servers['retrieval'].timeout = 2.0
        retrieval.terminate()
        retrieval.wait(timeout=10)
        lost = await agent.chat(principal, conv, '耳机保修多久', 'deterministic')
        assert '知识服务暂不可用' in lost['answer'] and not lost['citations']
        assert (await agent.chat(principal, conv, '查订单 O1001', 'deterministic'))['tool_actions'][0]['result']['ok']
        with env['db'].tx() as c:
            events = json.loads(c.scalar(select(traces.c.payload).where(traces.c.id==lost['trace_id'])))['events']
        dropped = [e for e in events if e['type']=='tool' and e.get('layer')=='transport']
        assert len(dropped) == 3 and len({e['operation_id'] for e in dropped}) == lost['tool_call_count'] == 1

        proposals = iter([[('search_knowledge', {'query': QUERY}), ('get_order', {'order_id': 'O1001'})], []])

        async def mixed(**unused):
            batch = next(proposals)
            return ModelDecision(AIMessage(content='' if batch else '订单 O1001 已送达。', tool_calls=[
                {'name': name, 'args': args, 'id': f'mixed-{name}'} for name, args in batch
            ]), 'fixture')

        monkeypatch.setattr(agent.harness.fake_model, 'complete_with_tools', mixed)
        partial = await agent.chat(principal, conv, '查订单 O1001 和耳机保修', 'deterministic')
        assert partial['tool_actions'][0]['result']['ok']
        assert '知识服务暂不可用' in partial['answer'] and 'O1001' in partial['answer']
        assert partial['error_code'] is None
    finally:
        await agent.close()


async def test_http_works_with_business_offline(env, http_services):
    config, _, _, _, business, _, _ = http_services
    business.terminate()
    business.wait(timeout=10)
    agent = Agent(env['b'], Gateway(env['b'], config), config)
    try:
        failed_order = await agent.chat(env['a'], env['conv'], '查订单 O1001', 'deterministic')
        assert not failed_order['tool_actions'][0]['result']['ok']
        assert (await agent.chat(env['a'], env['conv'], QUERY, 'deterministic'))['citations']
    finally:
        await agent.close()


@pytest.mark.models
async def test_http_real_local_models(env, tmp_path):
    if os.getenv('RUN_RETRIEVAL_REAL') != '1':
        pytest.skip('Set RUN_RETRIEVAL_REAL=1 for the real BGE and Chinese-CLIP HTTP smoke test')
    from app.multimodal.encoder import ChineseCLIPEncoder
    from app.rag import require_model, model_manifest
    config, variables, ports = network_config(env, tmp_path)
    try:
        require_model(config.embedding_model_path)
        require_model(config.reranker_model_path)
        clip_revision = ChineseCLIPEncoder(config.chinese_clip_model_path).revision
    except (FileNotFoundError, DomainError) as error:
        pytest.skip(str(error))
    # Build in a separate process so real model memory and embedded indexes
    # are released before the production Retrieval process opens them.
    code = """
from pathlib import Path
from app.business import Actor, Business
from app.config import settings
from app.db import DB
from app.retrieval_mcp import RetrievalService
service = RetrievalService(Business(DB(settings.sqlite_url)), settings)
service.rag.build(Path(settings.chroma_path).parent/'knowledge')
admin = Actor('admin','a','admin')
data = Path('tests/fixtures/multimodal/circle.png').read_bytes()
service.images.save(data,'catalog.png',admin,product_id='headphones-a')
service.index.put_text(admin,'耳机商品图与保修说明',product_id='headphones-a',doc_id='warranty')
assert service.index.sync()['failed'] == 0
service.rag.close()
"""
    knowledge = tmp_path/'knowledge'
    knowledge.mkdir()
    warranty = next(path for path in (ROOT/'knowledge').rglob('*.md') if 'id: warranty\n' in path.read_text(encoding='utf-8'))
    (knowledge/warranty.name).write_bytes(warranty.read_bytes())
    with (tmp_path/'build.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-c', code], env=variables, cwd=ROOT, stdout=log, stderr=log,
                       timeout=180, check=True)
    config = config.model_copy(update={'retrieval_mcp_timeout_seconds': 90})
    with process(['-m','app.retrieval_mcp'], variables, ports['retrieval'], tmp_path/'real-retrieval.log'):
        agent = Agent(env['b'], Gateway(env['b'], config), config)
        try:
            image = agent.images.save(IMAGE.read_bytes(), 'query.png', env['a'])['image_id']
            agent.images.bind(env['a'], env['conv'], [image])
            ctx = invocation(env)
            args = {'query': QUERY}
            hits = await agent.gateway.call(env['a'], 'search_knowledge', args, context=ctx)
            assert hits['ok'] and {row['doc_id'] for row in hits['data']} == {'warranty'}
            expected = {key+'_revision': value['revision'] for key, value in model_manifest().items()}
            assert hits['diagnostics']['model_revision'] == expected
            for name, args in (
                ('search_products_by_image', {'image_id': image}),
                ('search_text_by_image', {'image_id': image}),
                ('search_images_by_text', {'query': '耳机'}),
                ('search_products_by_image_and_text', {'image_id': image, 'query': '耳机'}),
            ):
                result = await agent.gateway.call(env['a'], name, args, context=ctx)
                assert result['ok'] and result['data'][0]['product_id'] == 'headphones-a'
                assert result['diagnostics']['model_revision'] == clip_revision
        finally:
            await agent.close()
