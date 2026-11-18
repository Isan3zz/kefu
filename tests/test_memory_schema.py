import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.db import DB, MEMORY_TABLES, conversations, memory_pending, memory_settings, messages, now, users


def test_memory_migration_preserves_existing_database(tmp_path, monkeypatch):
    url = 'sqlite:///' + str(tmp_path / 'migration.db')
    monkeypatch.setattr(settings, 'sqlite_url', url)
    alembic = Config('alembic.ini')
    command.upgrade(alembic, '0003')
    db = DB(url)
    with db.tx() as connection:
        connection.exec_driver_sql("INSERT INTO tenants (id) VALUES ('a')")
        connection.execute(users.insert().values(id='existing', tenant_id='a'))
    command.upgrade(alembic, 'head')
    assert {table.name for table in MEMORY_TABLES} <= set(inspect(db.engine).get_table_names())
    with db.tx() as connection:
        assert connection.scalar(select(users.c.id)) == 'existing'
        connection.execute(memory_settings.insert().values(tenant_id='a', user_id='existing', created_at=now(), updated_at=now()))
        flags = memory_settings.c
        assert tuple(connection.execute(select(flags.auto_write_enabled, flags.retrieval_enabled, flags.consent_generation)).one()) == (0, 1, 0)
    command.downgrade(alembic, '0003')
    assert 'users' in inspect(db.engine).get_table_names()
    assert memory_settings.name not in inspect(db.engine).get_table_names()


def test_memory_pending_schema_guards_duplicates_and_foreign_keys(tmp_path):
    db = DB('sqlite:///' + str(tmp_path / 'memory.db'))
    db.seed()
    with db.tx() as connection:
        connection.execute(conversations.insert().values(id='conversation', tenant_id='a', user_id='user_a'))
        connection.execute(messages.insert().values(id='source', conversation_id='conversation', role='user'))
        values = dict(tenant_id='a', user_id='user_a', conversation_id='conversation', message_id='source',
                      consent_generation=1, trigger_class='CANDIDATE', trigger_reason_code='STABLE',
                      status='PENDING', mode='deterministic', created_at=now(), updated_at=now())
        connection.execute(memory_pending.insert().values(id='first', **values))
        with pytest.raises(IntegrityError):
            connection.execute(memory_pending.insert().values(id='second', **values))
        with pytest.raises(IntegrityError):
            connection.execute(memory_pending.insert().values(id='third', **{**values, 'user_id': 'missing'}))
