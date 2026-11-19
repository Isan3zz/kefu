from datetime import timedelta

from langchain_core.messages import AIMessage
from sqlalchemy import select
import pytest

from app.business import Actor
from app.db import image_assets, products, traces
from app.harness.tool_protocol import ModelDecision
from app.multimodal.index import IMAGES
from conftest import auth, picture, upload


ADMIN = Actor('admin', 'a', 'admin')
PREFIX = '/api/v1/multimodal/search/'


def search(mm, endpoint, body, user='user_a'):
    return mm['client'].post(PREFIX+endpoint, json=body, headers=auth(mm, user))


def test_MM02_empty_catalog(mm):
    assert search(mm, 'by-text', {'query': '红色'}).json()['data'] == {'items': [], 'count': 0}
    ident = upload(mm).json()['data']['image_id']
    for name in ('by-image', 'text-by-image'):
        assert search(mm, name, {'image_id': ident}).json()['data']['items'] == []


def test_MM06_file_path_forbidden(mm):
    response = search(mm, 'by-text', {'query': '红色', 'file_path': '../../outside.jpg'})
    assert response.status_code == 422
    assert search(mm, 'by-image', {'image_id': '../outside.jpg'}).status_code == 404


def test_MM07_private_retrieval(mm):
    ident = upload(mm).json()['data']['image_id']
    for user in ('admin', 'user_b'):
        assert search(mm, 'by-image', {'image_id': ident}, user).status_code == 404


def test_MM08_tenant_partition_and_db_recheck(mm, indexed, monkeypatch):
    index, row, _ = indexed
    b_image = index.repo.save(picture(color='blue'), 'shape.png', Actor('local', 'b', 'admin'),
                              product_id='headphones-b')
    assert index.sync()['failed'] == 0
    assert search(mm, 'by-text', {'query': '蓝色'}, 'user_b').json()['data']['items'][0]['image_id'] == b_image['image_id']
    nearest = index.nearest

    def leaky(kind, tenant, vector, limit):
        return nearest(kind, 'a', vector, limit) + nearest(kind, 'b', vector, limit)

    monkeypatch.setattr(index, 'nearest', leaky)
    result = search(mm, 'by-text', {'query': '红色'}).json()['data']['items']
    assert [item['image_id'] for item in result] == [row['image_id']]


def test_MM09_current_sql_product_fields(mm, indexed):
    with mm['db'].tx() as c:
        c.execute(products.update().where(products.c.id == 'headphones-a')
                  .values(name='SQLite当前商品名', price_cents=46700, stock=0))
    [item] = search(mm, 'by-text', {'query': '红色'}).json()['data']['items']
    assert (item['product_name'], item['price_cents'], item['stock']) == ('SQLite当前商品名', 46700, 0)


def test_MM10_stale_chroma_after_cleanup_failure(mm, indexed, monkeypatch):
    index, row, _ = indexed
    monkeypatch.setattr(index, 'project', lambda job: (_ for _ in ()).throw(OSError('Simulated failed projection')))
    assert mm['client'].delete(row['url'], headers=auth(mm, 'admin')).status_code == 200
    assert index.collection(IMAGES).count() == 1
    assert search(mm, 'by-text', {'query': '红色'}).json()['data']['items'] == []
    assert mm['client'].get(row['url'], headers=auth(mm)).status_code == 404


def test_MM14_image_api(mm, indexed):
    _, row, _ = indexed
    ident = upload(mm).json()['data']['image_id']
    response = search(mm, 'by-image', {'image_id': ident})
    assert response.status_code == 200, response.json()
    [item] = response.json()['data']['items']
    assert (item['image_id'], item['product_id'], item['score_kind']) == (row['image_id'], 'headphones-a', 'cosine_similarity')


def test_MM15_text_search_ranking_changes(mm, indexed):
    index, row, _ = indexed
    with mm['db'].tx() as c:
        c.execute(products.insert().values(id='blue', tenant_id='a', name='蓝色示例', category='fixture',
                                           price_cents=40000, stock=1, attributes_json='{}'))
    blue = index.repo.save(picture(color='blue'), 'blue.png', ADMIN, product_id='blue')
    assert index.sync()['failed'] == 0
    for query, expected in [('红色', row), ('蓝色', blue)]:
        response = search(mm, 'by-text', {'query': query, 'top_k': 1}).json()['data']
        assert response['items'][0]['image_id'] == expected['image_id']


def test_MM16_visual_text_sql_link(mm, indexed):
    index, _, text_id = indexed
    mm['rag'].build('knowledge')
    doc_text = index.put_text(ADMIN, '红色耳机说明', product_id='headphones-a', doc_id='warranty')
    assert index.sync()['failed'] == 0
    ident = upload(mm).json()['data']['image_id']
    items = search(mm, 'text-by-image', {'image_id': ident}).json()['data']['items']
    assert {item['visual_text_id'] for item in items} == {text_id, doc_text}
    assert any(item['doc_id']=='warranty' and item['content']=='红色耳机说明' for item in items)


@pytest.mark.asyncio
async def test_MM17_MM27_agent_multistep(mm, indexed, monkeypatch):
    index, row, _ = indexed
    mm['rag'].build('knowledge')
    index.put_text(ADMIN, '红色耳机保修', product_id='headphones-a', doc_id='warranty')
    assert index.sync()['failed'] == 0
    query = index.repo.save(picture(), 'query.png', mm['a'])['image_id']
    calls = iter([
        {'name': 'search_products_by_image', 'args': {'image_id': query}, 'id': 'clip-call'},
        {'name': 'search_knowledge', 'args': {'query': '保修期限', 'product_id': 'headphones-a'}, 'id': 'rag-call'},
        None,
    ])

    async def decide(*, messages, tool_schemas, **unused):
        next_call = next(calls)
        assert 'search_products_by_image' in {tool['function']['name'] for tool in tool_schemas}
        if next_call is None:
            assert [m['tool_call_id'] for m in messages if m['role'] == 'tool'] == ['clip-call', 'rag-call']
            return ModelDecision(AIMessage(content='根据已查询的资料。'), 'test-script')
        return ModelDecision(AIMessage(content='', tool_calls=[next_call]), 'test-script')

    monkeypatch.setattr(mm['agent'].harness.fake_model, 'complete_with_tools', decide)
    result = await mm['agent'].harness.invoke_chat(principal=mm['a'], conversation_id=mm['conv'],
        message='请查图片对应商品的保修期限', image_ids=[query], mode='deterministic')
    assert result['cards'][0]['image_id'] == row['image_id']
    assert {chunk['doc_id'] for chunk in result['citations']} == {'warranty'}
    with mm['db'].tx() as c:
        import json
        trace = json.loads(c.execute(select(traces.c.payload).where(traces.c.id==result['trace_id'])).scalar_one())
    events = trace['events']
    assert any(e['type']=='policy.checked' and e['tool_name']=='search_products_by_image' for e in events)
    assert any(e['type']=='loop.tool_observed' and e['tool_call_id']=='clip-call' for e in events)


@pytest.mark.asyncio
async def test_MM18_chat_enforces_explicit_ceiling_even_on_image_only_tool(mm, indexed, monkeypatch):
    index, _, _ = indexed
    ident = index.repo.save(picture(), 'query.png', mm['a'])['image_id']
    responses = iter([
        AIMessage(content='', tool_calls=[{'name': 'search_products_by_image',
                                          'args': {'image_id': ident}, 'id': 'clip-budget'}]),
        AIMessage(content='这款耳机符合预算。'),
    ])

    async def decide(**unused):
        return ModelDecision(next(responses), 'test-script')

    monkeypatch.setattr(mm['agent'].harness.fake_model, 'complete_with_tools', decide)
    result = await mm['agent'].harness.invoke_chat(principal=mm['a'], conversation_id=mm['conv'],
        message='找类似的商品，不超过 250 元', image_ids=[ident], mode='deterministic')
    assert result['cards'] == []
    outcome = result['tool_actions'][0]['result']
    assert outcome['ok'] and outcome['data'] == [], outcome
    assert '符合预算' not in result['answer']


def test_MM18_strict_mixed_price_stock_category(mm, indexed):
    ident = upload(mm).json()['data']['image_id']
    query = {'image_id': ident, 'query': '类似红色', 'filters': {'max_price_cents': 29900, 'in_stock': True}}
    assert search(mm, 'by-image-and-text', query).json()['data']['count'] == 1
    query['filters']['max_price_cents'] -= 1
    assert search(mm, 'by-image-and-text', query).json()['data']['items'] == []
    query['filters'] = {'category': 'irrelevant'}
    assert search(mm, 'by-image-and-text', query).json()['data']['items'] == []


def test_MM19_deduplicate_images_and_cap_results(mm, indexed):
    index, _, _ = indexed
    alt = index.repo.save(picture(color='blue'), 'alternate.png', ADMIN, product_id='headphones-a')
    assert index.sync()['failed'] == 0
    result = search(mm, 'by-text', {'query': '蓝色', 'top_k': 20}).json()['data']
    assert result['count'] == 1 and result['items'][0]['image_id'] == alt['image_id']
    assert search(mm, 'by-text', {'query': '红色', 'top_k': 21}).status_code == 422


def test_MM20_expired_missing_stale_and_unready(mm, indexed):
    index, row, _ = indexed
    ident = upload(mm).json()['data']['image_id']
    with mm['db'].tx() as c:
        c.execute(image_assets.update().where(image_assets.c.image_id==ident)
                  .values(expires_at=(index.repo.business.clock()-timedelta(seconds=1)).isoformat()))
        product_row = index.row(c, IMAGES, row['image_id'])
    assert search(mm, 'by-image', {'image_id': ident}).status_code == 404
    index.repo.storage.path(product_row).unlink()
    assert search(mm, 'by-text', {'query': '红色'}).json()['data']['items'] == []
    index.client().delete_collection(index.names[IMAGES])
    assert search(mm, 'by-text', {'query': '红色'}).json()['code'] == 'INDEX_NOT_READY'
