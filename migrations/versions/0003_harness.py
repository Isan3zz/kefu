"""Add request lifecycle and scoped context summaries without altering business tables."""
from alembic import op
import sqlalchemy as sa

revision='0003'
down_revision='0002'
branch_labels=None
depends_on=None

def upgrade():
    names=set(sa.inspect(op.get_bind()).get_table_names())
    if 'agent_invocations' not in names:
        strings=['trace_id','conversation_id','graph_thread_id','workflow_id','tenant_id','user_id','parent_invocation_id','client_request_id','request_body_hash','response_message_id','error_code','agent_release_id','started_at','finished_at','deadline_at_utc','created_at','updated_at']
        op.create_table('agent_invocations',sa.Column('invocation_id',sa.String,primary_key=True),*(sa.Column(n,sa.String) for n in strings),sa.Column('status',sa.String,nullable=False),sa.Column('response_json',sa.Text),*(sa.Column(n,sa.Integer) for n in ['model_calls','tool_calls','prompt_tokens','completion_tokens','write_started']))
        op.create_index('uq_invocation_client_request','agent_invocations',['tenant_id','user_id','conversation_id','client_request_id'],unique=True,sqlite_where=sa.text('client_request_id IS NOT NULL'))
        op.create_index('ix_invocation_owner','agent_invocations',['tenant_id','user_id','conversation_id','created_at'])
        op.create_index('ix_invocation_trace','agent_invocations',['trace_id'])
        op.create_index('ix_invocation_thread','agent_invocations',['graph_thread_id'])
    if 'context_summaries' not in names:
        op.create_table('context_summaries',sa.Column('id',sa.String,primary_key=True),*(sa.Column(n,sa.String) for n in ['tenant_id','user_id','conversation_id','graph_thread_id','covers_through_message_id','summary_version','source_hash','created_at']),sa.Column('summary_text',sa.Text))
        op.create_index('ix_summary_scope','context_summaries',['tenant_id','user_id','graph_thread_id','source_hash'])

def downgrade():
    op.drop_table('context_summaries')
    op.drop_table('agent_invocations')
