"""SQLAlchemy schema; explicit transactions serialize security-sensitive writes."""
import hashlib
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, event, MetaData, Table, Column, String, Integer, ForeignKey, Text, Index, text, CheckConstraint, UniqueConstraint
from sqlalchemy.pool import NullPool

metadata = MetaData()
def table(name, *columns):
    return Table(name, metadata, *columns)
def s(name, **kw): return Column(name, String, **kw)
def i(name, **kw): return Column(name, Integer, **kw)
tenants = table("tenants", s("id", primary_key=True), s("name"), s("created_at"))
users = table("users", s("id", primary_key=True), Column("tenant_id", String, ForeignKey("tenants.id")), s("display_name"), s("role"))
auth_sessions = table("auth_sessions", s("token_hash", primary_key=True), Column("user_id", String, ForeignKey("users.id")), s("expires_at"), s("revoked_at"))
products = table("products", s("id", primary_key=True), Column("tenant_id", String, ForeignKey("tenants.id")), s("name"), s("category"), i("price_cents"), i("stock"), s("attributes_json"))
orders = table("orders", s("id", primary_key=True), Column("tenant_id", String, ForeignKey("tenants.id")), Column("user_id", String, ForeignKey("users.id")), s("status"), i("paid_amount_cents"), s("paid_at"), s("delivered_at"), i("version", default=1))
order_items = table("order_items", s("id", primary_key=True), Column("order_id", String, ForeignKey("orders.id")), Column("product_id", String, ForeignKey("products.id")), i("quantity"), i("unit_price_cents"))
shipments = table("shipments", s("id", primary_key=True), Column("order_id", String, ForeignKey("orders.id")), s("carrier"), s("tracking_no"), s("status"), s("updated_at"))
conversations = table("conversations", s("id", primary_key=True), Column("tenant_id", String, ForeignKey("tenants.id")), Column("user_id", String, ForeignKey("users.id")), s("title"), s("created_at"), s("updated_at"))
messages = table("messages", s("id", primary_key=True), Column("conversation_id", String, ForeignKey("conversations.id")), s("role"), Column("content", Text), s("created_at"), s("trace_id"))
workflow_instances = table("workflow_instances", s("id", primary_key=True), Column("tenant_id", String, ForeignKey("tenants.id")), Column("user_id", String, ForeignKey("users.id")), Column("conversation_id", String, ForeignKey("conversations.id")), s("graph_thread_id"), s("type"), s("status"), Column("order_id", String, ForeignKey("orders.id")), i("amount_cents"), s("policy_version"), s("confirm_nonce_hash"), s("nonce_expires_at"), s("operation_key", unique=True), i("order_version"), i("version", default=1), s("updated_at"))
refund_requests = table("refund_requests", s("id", primary_key=True), Column("tenant_id", String, ForeignKey("tenants.id")), Column("user_id", String, ForeignKey("users.id")), Column("order_id", String, ForeignKey("orders.id"), unique=True), i("amount_cents"), s("status"), s("idempotency_key", unique=True), s("policy_version"), s("created_at"))
tickets = table("tickets", s("id", primary_key=True), Column("tenant_id", String, ForeignKey("tenants.id")), Column("user_id", String, ForeignKey("users.id")), Column("conversation_id", String, ForeignKey("conversations.id")), s("summary"), s("reason"), s("status"), s("created_at"))
documents = table("documents", s("id", primary_key=True), s("tenant_id"), s("title"), s("source_path"), s("version"), s("status"), s("effective_at"), s("expires_at"), s("visibility"))
document_chunks = table("document_chunks", s("chunk_id", primary_key=True), Column("document_id", String, ForeignKey("documents.id")), i("chunk_index"), Column("content", Text), s("metadata_json"), s("content_hash"))
traces = table("traces", s("id", primary_key=True), s("tenant_id"), s("user_id"), s("conversation_id"), Column("payload", Text))
evaluation_runs = table("evaluation_runs", s("id", primary_key=True), s("tenant_id"), s("user_id"), Column("payload", Text))
knowledge_snapshots = table("knowledge_snapshots", s("id", primary_key=True), Column("payload", Text, nullable=False))

agent_invocations = table('agent_invocations', s('invocation_id',primary_key=True),s('trace_id'),s('conversation_id'),s('graph_thread_id'),s('workflow_id'),s('tenant_id'),s('user_id'),s('parent_invocation_id'),s('client_request_id'),s('request_body_hash'),s('response_message_id'),Column('response_json',Text),s('status',nullable=False),s('error_code'),s('agent_release_id'),s('started_at'),s('finished_at'),s('deadline_at_utc'),i('model_calls',default=0),i('tool_calls',default=0),i('prompt_tokens'),i('completion_tokens'),i('write_started',default=0),s('created_at'),s('updated_at'))
Index('uq_invocation_client_request',agent_invocations.c.tenant_id,agent_invocations.c.user_id,agent_invocations.c.conversation_id,agent_invocations.c.client_request_id,unique=True,sqlite_where=text('client_request_id IS NOT NULL'))
Index('ix_invocation_owner',agent_invocations.c.tenant_id,agent_invocations.c.user_id,agent_invocations.c.conversation_id,agent_invocations.c.created_at)
Index('ix_invocation_trace',agent_invocations.c.trace_id)
Index('ix_invocation_thread',agent_invocations.c.graph_thread_id)
context_summaries = table('context_summaries',s('id',primary_key=True),s('tenant_id'),s('user_id'),s('conversation_id'),s('graph_thread_id'),s('covers_through_message_id'),Column('summary_text',Text),s('summary_version'),s('source_hash'),s('created_at'))
Index('ix_summary_scope',context_summaries.c.tenant_id,context_summaries.c.user_id,context_summaries.c.graph_thread_id,context_summaries.c.source_hash)

def memory_owner(primary=False):
    return (Column('tenant_id', ForeignKey('tenants.id'), nullable=False, primary_key=primary),
            Column('user_id', ForeignKey('users.id'), nullable=False, primary_key=primary))

def memory_timestamps():
    return s('created_at', nullable=False), s('updated_at', nullable=False)

memory_settings = table('user_memory_settings', *memory_owner(primary=True),
    i('auto_write_enabled', nullable=False, server_default='0'), i('retrieval_enabled', nullable=False, server_default='1'),
    i('consent_generation', nullable=False, server_default='0'), *memory_timestamps())
memories = table('user_memories', s('id', primary_key=True), *memory_owner(), s('memory_type', nullable=False),
    Column('content', Text, nullable=False), s('subject_key', nullable=False),
    Column('source_conversation_id', ForeignKey('conversations.id')), Column('source_message_id', ForeignKey('messages.id')),
    Column('evidence_excerpt', Text), s('source_created_at'), s('source_trace_id'),
    s('status', nullable=False), i('version', nullable=False), *memory_timestamps(), s('expires_at'),
    s('extractor_version', nullable=False), s('embedding_version'), i('index_pending', nullable=False, server_default='1'),
    CheckConstraint("status IN ('active','superseded','deleted')"),
    CheckConstraint("memory_type IN ('preference','stable_constraint','stable_fact')"),
    CheckConstraint("status = 'deleted' OR length(trim(content)) > 0"))
Index('ix_memory_owner_status', memories.c.tenant_id, memories.c.user_id, memories.c.status)
Index('uq_memory_active_subject', memories.c.tenant_id, memories.c.user_id, memories.c.subject_key,
      unique=True, sqlite_where=text("status = 'active'"))
Index('ix_memory_index_pending', memories.c.index_pending, memories.c.updated_at)
memory_revisions = table('user_memory_revisions', s('id', primary_key=True),
    Column('memory_id', ForeignKey('user_memories.id'), nullable=False), i('old_version'), i('new_version', nullable=False),
    s('operation', nullable=False), Column('prior_content', Text), Column('new_content', Text),
    Column('source_message_id', ForeignKey('messages.id')), s('created_at', nullable=False), UniqueConstraint('memory_id', 'new_version'))
memory_jobs = table('memory_jobs', s('id', primary_key=True), *memory_owner(),
    Column('conversation_id', ForeignKey('conversations.id'), nullable=False), i('consent_generation', nullable=False),
    s('batch_fingerprint', nullable=False), Column('source_message_ids_json', Text, nullable=False),
    s('extractor_version', nullable=False), s('flush_reason', nullable=False), s('status', nullable=False),
    i('attempts', nullable=False, server_default='0'), s('lease_expires_at'), s('next_retry_at'), *memory_timestamps(),
    s('error_code'), s('mode', nullable=False), s('origin_trace_id'), s('origin_invocation_id'),
    UniqueConstraint('tenant_id', 'user_id', 'conversation_id', 'consent_generation', 'batch_fingerprint'),
    CheckConstraint("status IN ('PENDING','PROCESSING','COMPLETED','FAILED','CANCELLED')"))
Index('ix_memory_jobs_ready', memory_jobs.c.status, memory_jobs.c.next_retry_at, memory_jobs.c.lease_expires_at)
memory_pending = table('memory_pending_messages', s('id', primary_key=True), *memory_owner(),
    Column('conversation_id', ForeignKey('conversations.id'), nullable=False),
    Column('message_id', ForeignKey('messages.id'), nullable=False), i('consent_generation', nullable=False),
    s('trigger_class', nullable=False), s('trigger_reason_code', nullable=False), s('status', nullable=False),
    Column('batch_id', ForeignKey('memory_jobs.id')), *memory_timestamps(), s('mode', nullable=False),
    s('origin_trace_id'), s('origin_invocation_id'), UniqueConstraint('tenant_id', 'user_id', 'message_id', 'consent_generation'),
    CheckConstraint("trigger_class IN ('CANDIDATE','URGENT')"),
    CheckConstraint("status IN ('PENDING','CLAIMED','DONE','CANCELLED')"))
Index('ix_memory_pending_scope', memory_pending.c.status, memory_pending.c.tenant_id, memory_pending.c.user_id,
      memory_pending.c.conversation_id, memory_pending.c.consent_generation, memory_pending.c.created_at)
Index('ix_memory_pending_batch', memory_pending.c.batch_id)
memory_operations = table('memory_candidate_operations', Column('job_id', ForeignKey('memory_jobs.id'), primary_key=True),
    s('candidate_fingerprint', primary_key=True), s('operation', nullable=False),
    Column('memory_id', ForeignKey('user_memories.id')), s('created_at', nullable=False))
MEMORY_TABLES = (memory_settings, memories, memory_revisions, memory_jobs, memory_pending, memory_operations)

def now(): return datetime.now(timezone.utc).isoformat()
def uid(prefix=""): return prefix + secrets.token_hex(12)
def digest(value): return hashlib.sha256(value.encode()).hexdigest()

class DB:
    def __init__(self, url):
        from pathlib import Path
        if url.startswith("sqlite:///") and ":memory:" not in url:
            Path(url[10:]).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, poolclass=NullPool)
        @event.listens_for(self.engine, "connect")
        def pragmas(conn, _):
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.execute("PRAGMA journal_mode=WAL")
    def initialize(self): metadata.create_all(self.engine)
    @contextmanager
    def tx(self):
        with self.engine.connect() as c:
            c.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield c
                c.commit()
            except BaseException:
                c.rollback()
                raise
    def seed(self, clock=None):
        self.initialize()
        fixture_time=clock() if clock else datetime.now(timezone.utc)
        stamp=fixture_time.isoformat()
        with self.tx() as c:
            if c.execute(users.select()).first(): return
            for t in ("a", "b"):
                c.execute(tenants.insert().values(id=t, name="Demo "+t, created_at=stamp))
            for u,t,r in [("user_a","a","customer"),("user_b","b","customer"),("admin","a","admin")]:
                c.execute(users.insert().values(id=u,tenant_id=t,display_name=u,role=r))
            for t in ("a","b"):
                c.execute(products.insert().values(id="headphones-"+t,tenant_id=t,name="云听耳机",category="headphones",price_cents=29900,stock=20,attributes_json='{"colors":["黑色","蓝色"],"warranty_months":12}'))
            for oid,t,u,status,days in [("O1001","a","user_a","DELIVERED",2),("O1002","a","user_a","SHIPPED",None),("O1003","a","user_a","PAID",None),("O1004","a","user_a","DELIVERED",10),("O2001","b","user_b","PAID",None)]:
                c.execute(orders.insert().values(id=oid,tenant_id=t,user_id=u,status=status,paid_amount_cents=29900,paid_at=stamp,delivered_at=(fixture_time-timedelta(days=days)).isoformat() if days else None,version=1))
                c.execute(order_items.insert().values(id="item-"+oid,order_id=oid,product_id="headphones-"+t,quantity=1,unit_price_cents=29900))
                c.execute(shipments.insert().values(id="ship-"+oid,order_id=oid,carrier="模拟快递",tracking_no="DEMO"+oid if status in ('SHIPPED','DELIVERED') else None,status="已签收" if status=="DELIVERED" else "运输中" if status=="SHIPPED" else "待发货",updated_at=stamp))
