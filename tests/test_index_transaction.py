from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from sqlalchemy import inspect,select
from app.db import DB,documents,orders,metadata

def test_database_commit_failure_does_not_activate_new_file_pointer(env,tmp_path,monkeypatch):
    rag=env['rag']; before=rag.build('knowledge')
    source=tmp_path/'next-kb'; shutil.copytree('knowledge',source)
    policy=source/'policies/refund_policy_v1.md'
    policy.write_text(policy.read_text(encoding='utf-8').replace('version: "1"','version: "2"'),encoding='utf-8')
    original=env['db'].tx
    @contextmanager
    def fail_commit():
        with original() as connection:
            yield connection
            raise OSError('simulated transaction commit failure')
    monkeypatch.setattr(env['db'],'tx',fail_commit)
    with pytest.raises(OSError): rag.build(source)
    monkeypatch.setattr(env['db'],'tx',original)
    # The convenience JSON may be ahead of a failed transaction; it is never authority.
    assert json.loads((rag.path/'active.json').read_text(encoding='utf-8'))['snapshot']!=before['snapshot']
    assert rag.active_manifest()['snapshot']==before['snapshot']
    with env['db'].tx() as connection:
        assert connection.scalar(select(documents.c.version).where(documents.c.id=='refund_policy_v1'))=='1'
    assert rag.search('七天退款','a')

def test_alembic_empty_database_and_upgrade_preserve_existing_orders(tmp_path,monkeypatch):
    url='sqlite:///'+str(tmp_path/'migration.db')
    variables=os.environ.copy(); variables['SQLITE_URL']=url
    def migrate(*args):
        command=subprocess.run([sys.executable,'-m','alembic',*args],env=variables,capture_output=True,text=True,timeout=30)
        assert command.returncode==0,command.stderr
    migrate('upgrade','0001')
    db=DB(url)
    assert 'orders' in inspect(db.engine).get_table_names()
    assert 'knowledge_snapshots' not in inspect(db.engine).get_table_names()
    with monkeypatch.context() as patch:
        patch.setattr(db,'initialize',lambda:None)
        db.seed()
    assert 'knowledge_snapshots' not in inspect(db.engine).get_table_names()
    with db.tx() as connection: existing=[dict(r) for r in connection.execute(select(orders)).mappings()]
    migrate('upgrade','head')
    assert set(metadata.tables).issubset(inspect(db.engine).get_table_names())
    with db.tx() as connection:
        assert [dict(r) for r in connection.execute(select(orders)).mappings()]==existing
        assert connection.exec_driver_sql('SELECT version_num FROM alembic_version').scalar()=='0005'
    migrate('downgrade','0001')
    with db.tx() as connection: assert len(connection.execute(select(orders)).all())==len(existing)
    migrate('upgrade','head')

def test_model_singleton_respects_custom_paths_and_revision(env,monkeypatch):
    import app.rag as module
    created=[]; manifest={'embedding':{'revision':'v1'}}
    def construct(config):
        value={'path':config.embedding_model_path,'index':len(created)}; created.append(value); return value
    monkeypatch.setattr(module,'LocalModels',construct)
    monkeypatch.setattr(module,'model_manifest',lambda:manifest)
    module.local_models.cache_clear()
    try:
        first=module.local_models(env['cfg'])
        assert first is module.local_models(env['cfg'])
        manifest['embedding']['downloaded_at']='updated verification time'
        assert first is module.local_models(env['cfg'])
        second=module.local_models(env['cfg'].model_copy(update={'embedding_model_path':'another-local-model'}))
        assert second['path']=='another-local-model' and second is not first
        manifest['embedding']['revision']='v2'
        assert module.local_models(env['cfg'].model_copy(update={'embedding_model_path':'another-local-model'})) is not second
        assert len(created)==3
    finally: module.local_models.cache_clear()
