import pytest
from sqlalchemy import select
from app.db import documents,document_chunks

@pytest.mark.parametrize('mode',['bm25','dense','hybrid','hybrid_rerank'])
def test_expired_future_private_and_other_tenant_chunks_never_escape_old_index(env,mode):
    rag=env['rag']; rag.build('knowledge')
    excluded={'refund_policy_v1','colors','warranty','pairing'}
    with env['db'].tx() as connection:
        connection.execute(documents.update().where(documents.c.id=='refund_policy_v1').values(expires_at='2020-01-02T00:00:00+00:00'))
        connection.execute(documents.update().where(documents.c.id=='colors').values(tenant_id='b'))
        connection.execute(documents.update().where(documents.c.id=='warranty').values(visibility='internal'))
        connection.execute(documents.update().where(documents.c.id=='pairing').values(effective_at='2999-01-01T00:00:00+00:00'))
        bad=set(connection.scalars(select(document_chunks.c.chunk_id).where(document_chunks.c.document_id.in_(excluded))))
    results=rag.search('耳机颜色保修配对七天退款政策','a',mode,k=30)
    assert results
    assert not ({row['chunk_id'] for row in results}&bad)
