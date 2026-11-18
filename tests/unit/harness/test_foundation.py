import asyncio
from dataclasses import FrozenInstanceError
import pytest
from sqlalchemy import select
from app.business import DomainError
from app.harness.run_controller import RunController
from app.harness.contracts import ContextBudget,HarnessFailure
from app.db import orders,agent_invocations

async def test_H17_request_identity_and_conflicting_body(env):
    controller=RunController(env['b'],env['cfg'])
    ctx,replay=controller.begin(env['a'],env['conv'],'conv:'+env['conv'],{'message':'hello'},'key')
    assert replay is None
    again,response=controller.begin(env['a'],env['conv'],'conv:'+env['conv'],{'message':'hello'},'key')
    assert again is None and response['invocation_id']==ctx.invocation_id and response['processing']
    with pytest.raises(DomainError) as error: controller.begin(env['a'],env['conv'],'conv:'+env['conv'],{'message':'changed'},'key')
    assert error.value.code=='REQUEST_CONFLICT'
    with pytest.raises(FrozenInstanceError): ctx.principal.user_id='user_b'
    with pytest.raises(DomainError): controller.read(env['other'],ctx.invocation_id)

async def test_H18_thread_lock_parallelism_and_cleanup(env):
    locks=RunController(env['b'],env['cfg']).locks; entered=asyncio.Event(); release=asyncio.Event(); second=asyncio.Event()
    async def first():
        async with locks.acquire('conv:a'): entered.set(); await release.wait()
    async def next_same():
        async with locks.acquire('conv:a'): second.set()
    task=asyncio.create_task(first()); await entered.wait(); other=asyncio.create_task(next_same())
    async with locks.acquire('conv:b'): assert not second.is_set()
    release.set(); await asyncio.gather(task,other)
    assert second.is_set() and not locks.entries

def test_context_budget_configuration_rejected(env):
    with pytest.raises(ValueError): ContextBudget(model_context_window=10,reserved_output_tokens=5,safety_margin_tokens=2,max_input_tokens=4)
    with pytest.raises(ValueError): type(env['cfg'])(context_max_input_tokens=20000)

def test_H06_expired_context_cannot_continue(env):
    from dataclasses import replace
    controller=RunController(env['b'],env['cfg'])
    ctx,_=controller.begin(env['a'],env['conv'],'conv:'+env['conv'],{})
    with pytest.raises(HarnessFailure,match='超时'): replace(ctx,deadline_monotonic=0).remaining()

def test_harness_migration_preserves_existing_orders(tmp_path,monkeypatch):
    from alembic.config import Config
    from alembic import command
    from app.config import settings
    from app.db import DB
    path=tmp_path/'migrated.db'; monkeypatch.setattr(settings,'sqlite_url','sqlite:///'+str(path))
    cfg=Config('alembic.ini'); command.upgrade(cfg,'0002')
    db=DB(settings.sqlite_url)
    with db.tx() as c: c.execute(orders.insert().values(id='preserved',status='PAID',paid_amount_cents=100))
    command.upgrade(cfg,'head')
    with db.tx() as c:
        assert c.scalar(select(orders.c.id))=='preserved'
        assert c.execute(select(agent_invocations)).all()==[]
    command.downgrade(cfg,'0002'); command.upgrade(cfg,'head')
    with db.tx() as c: assert c.scalar(select(orders.c.id))=='preserved'
    db.engine.dispose()
