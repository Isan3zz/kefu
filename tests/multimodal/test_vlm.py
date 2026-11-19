import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import httpx
from openai import AsyncOpenAI
import pytest

from app.business import Actor, DomainError
from app.harness.execution_context import ExecutionContext
from app.multimodal.vlm_image_adapter import VLMImageAdapter
from conftest import auth, upload


def chat(mm, **kwargs):
    user = kwargs.pop('user', 'user_a')
    body = {'conversation_id': mm['conv'], 'message': '请描述这张图片', **kwargs}
    return mm['client'].post('/api/v1/chat', headers=auth(mm, user), json=body)


@pytest.fixture
def provider(mm, monkeypatch):
    cfg = mm['cfg']
    for key, value in {
        'llm_base_url': 'https://provider.example/v1', 'llm_api_key': 'test-only',
        'vlm_model': 'vision-test', 'llm_model': 'text-test',
        'vlm_supports_vision': True, 'vlm_supports_tool_calling': True,
        'multimodal_external_vlm_images_enabled': True, 'model_max_retries': 0,
    }.items():
        monkeypatch.setattr(cfg, key, value)
    requests = []
    responses = []

    async def mock(request):
        body = json.loads(request.content)
        requests.append(body)
        if responses:
            return responses.pop(0)(body)
        return answer('图中有红色背景和白色圆形。')

    sdk = AsyncOpenAI(api_key='test-only', base_url=cfg.llm_base_url,
                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(mock)))
    mm['agent']._llm_client = sdk
    return requests, responses


def answer(text):
    return httpx.Response(200, json={'id': 'provider-fake', 'model': 'vision-test',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}]})


def user_content(body):
    return next(m['content'] for m in body['messages'] if m['role'] == 'user')


def test_MM21_plain_text_api(mm):
    result = chat(mm, message='你好', mode='deterministic')
    assert result.status_code == 200, result.json()
    assert 'answer' in result.json()['data'] and 'cards' not in result.json()['data']


def test_MM22_conversation_and_principal_bound(mm, provider):
    ident = upload(mm).json()['data']['image_id']
    conv2 = mm['b'].conversation(mm['a'])['id']
    assert chat(mm, image_ids=[ident], mode='deterministic').status_code == 200
    assert chat(mm, conversation_id=conv2, image_ids=[ident], allow_external_image_send=True).status_code == 404
    b_conv = mm['b'].conversation(mm['other'])['id']
    assert chat(mm, user='user_b', conversation_id=b_conv, image_ids=[ident],
                allow_external_image_send=True).status_code == 404
    assert chat(mm, image_ids=[ident], actor={'tenant_id': 'a'}).status_code == 422
    assert not provider[0]


def test_MM23_MM24_valid_jpeg_and_no_local_paths(mm, provider):
    ident = upload(mm).json()['data']['image_id']
    response = chat(mm, image_ids=[ident], allow_external_image_send=True)
    assert response.status_code == 200, response.json()
    body, = provider[0]
    assert body['model'] == 'vision-test' and body['tools'] and body['tool_choice'] == 'auto'
    text, image = user_content(body)
    assert text['type'] == 'text' and ident in text['text']
    assert image['type'] == 'image_url'
    uri = image['image_url']['url']
    assert uri.startswith('data:image/jpeg;base64,')
    payload = base64.b64decode(uri.split(',', 1)[1], validate=True)
    with Image.open(BytesIO(payload)) as decoded:
        assert decoded.format == 'JPEG' and decoded.getpixel((0, 0))[0] > 200
    for forbidden in ('file://', 'localhost', '/api/v1/images/', 'uploads/', str(mm['agent'].images.storage.root)):
        assert forbidden not in json.dumps(body)


@pytest.mark.parametrize(('setting', 'value', 'code'), [
    ('vlm_supports_vision', False, 'VLM_VISION_UNSUPPORTED'),
    ('vlm_supports_tool_calling', False, 'VLM_TOOL_CALLING_UNSUPPORTED'),
    ('vlm_model', '', 'VLM_NOT_CONFIGURED'),
    ('vlm_provider_transport', 'unsupported', 'VLM_VISION_UNSUPPORTED'),
])
def test_MM25_capability_error(mm, provider, monkeypatch, setting, value, code):
    monkeypatch.setattr(mm['cfg'], setting, value)
    ident = upload(mm).json()['data']['image_id']
    response = chat(mm, image_ids=[ident], allow_external_image_send=True)
    assert response.status_code == 503 and response.json()['code'] == code
    assert not provider[0]


@pytest.mark.parametrize(('reason', 'code'), [('images unsupported', 'VLM_VISION_UNSUPPORTED'),
                                               ('tool calling unsupported', 'VLM_TOOL_CALLING_UNSUPPORTED')])
def test_MM25_provider_rejection_not_retried_as_text(mm, provider, reason, code):
    provider[1].append(lambda _: httpx.Response(400, json={'error': {'message': reason}}))
    ident = upload(mm).json()['data']['image_id']
    response = chat(mm, image_ids=[ident], allow_external_image_send=True)
    assert response.status_code == 503 and response.json()['code'] == code
    assert len(provider[0]) == 1


def test_MM26_consent_disabled_and_persistence(mm, provider, monkeypatch):
    ident = upload(mm).json()['data']['image_id']
    response = chat(mm, image_ids=[ident])
    assert response.status_code == 200
    assert isinstance(user_content(provider[0][0]), str) and provider[0][0]['model'] == 'text-test'
    with monkeypatch.context() as patch:
        patch.setattr(mm['cfg'], 'multimodal_external_vlm_images_enabled', False)
        assert chat(mm, image_ids=[ident], allow_external_image_send=True).json()['code'] == 'VLM_IMAGE_SEND_DISABLED'
    assert len(provider[0]) == 1

    def echo(body):
        return answer(user_content(body)[1]['image_url']['url'])

    provider[1].append(echo)
    response = chat(mm, image_ids=[ident], allow_external_image_send=True)
    assert response.status_code == 200, response.json()
    data = response.json()['data']
    trace = mm['client'].get('/api/v1/traces/'+data['trace_id'], headers=auth(mm)).json()['data']
    assert any(e['type'] == 'vlm.image.sent' for e in trace['events'])
    prefix, encoded = user_content(provider[0][-1])[1]['image_url']['url'].split(',', 1)
    for content in (response.content, json.dumps(trace).encode(), *(p.read_bytes() for p in
                    Path(mm['cfg'].checkpoint_db_path).parent.glob('cp.db*'))):
        assert encoded.encode() not in content
        assert (prefix + ',/9j/').encode() not in content


def test_MM07_MM20_vlm_adapter_checks_acl_and_deleted(mm, provider):
    from time import monotonic
    ident = upload(mm).json()['data']['image_id']
    adapter = VLMImageAdapter(mm['agent'].images)
    for principal in (mm['other'], Actor('admin', 'a', 'admin'), mm['a']):
        if principal == mm['a']:
            assert mm['client'].delete('/api/v1/images/'+ident, headers=auth(mm)).status_code == 200
        ctx = ExecutionContext('test', 'trace', mm['conv'], 'thread', principal, 'test', monotonic()+10)
        with pytest.raises(DomainError) as error:
            adapter.prepare(ctx, [ident], consent=True)
        assert error.value.code == 'IMAGE_NOT_FOUND'
    assert not provider[0]
