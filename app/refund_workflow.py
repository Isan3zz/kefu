"""Checkpointed refund execution and authoritative result verification."""
import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from sqlite3 import Error as SQLiteError
from typing import Literal, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy.exc import SQLAlchemyError

from app.business import DomainError
from app.observability import emit


@dataclass(frozen=True)
class RefundWorkflowResult:
    workflow_id: str | None
    order_id: str
    status: Literal['WAITING_CONFIRMATION', 'VERIFIED_SUCCESS', 'FAILED', 'UNKNOWN']
    refund_request_id: str | None = None
    amount_cents: int | None = None
    business_status: Literal['ACCEPTED', 'REJECTED'] | None = None
    verified: bool = False
    error_code: str | None = None
    reason: str | None = None
    idempotent_replay: bool = False
    created_at: str | None = None

    def reply(self):
        if self.status == 'VERIFIED_SUCCESS':
            return f'退款申请已受理，申请编号：{self.refund_request_id}。受理不代表退款已到账。'
        if self.status == 'UNKNOWN':
            return '退款申请的受理状态暂未确认，请查询原退款工作流。'
        if self.status == 'FAILED':
            return f'退款申请未获受理，{self.reason}。'
        return '请通过确认按钮确认退款申请。'


PREVIEW_CHANGED = '订单或资格已变化，请重新申请预览'
REJECTION_REASONS = {
    'ALREADY_EXISTS': '该订单已有退款申请',
    'HUMAN_REVIEW': '该订单需要人工审核',
    'MISSING_DELIVERY_TIME': '缺少送达时间，需要人工审核',
    'EXPIRED': '订单已超出退款申请期限',
    'INVALID_STATUS': '当前订单状态不支持退款',
    'INVALID_AMOUNT': '订单实付金额无效',
}


class RefundState(TypedDict, total=False):
    workflow_id: str
    order_id: str
    receipt: dict | None
    response_missing: bool
    result: dict


class RefundWorkflow:
    def __init__(self, business, checkpoint_path):
        self.business = business
        self.checkpoint_path = checkpoint_path

    @staticmethod
    def failed(wid, oid, error):
        return RefundWorkflowResult(wid, oid, 'FAILED', business_status='REJECTED',
                                    error_code=error.code, reason=REJECTION_REASONS.get(error.message, error.message))

    async def prepare(self, ctx, services, oid, preview):
        wid = preview['workflow_id']
        if wid is None:
            if preview['status'] != 'FAILED' or not preview.get('reason_code'):
                raise DomainError('TOOL_BAD_RESULT', '退款预览回执无效')
            error = DomainError('BUSINESS_RULE_FAILED', preview['reason_code'])
            return self.failed(None, oid, error), None
        current = self.business.workflow(ctx.principal, wid)
        if current['type'] != 'refund' or current['order_id'] != oid:
            raise DomainError('FORBIDDEN', '退款预览不属于当前订单')
        if current['status'] != 'WAITING_CONFIRMATION':
            return self.read_result(ctx.principal, current), None
        if current['conversation_id'] != ctx.conversation_id:
            raise DomainError('FORBIDDEN', '退款预览不属于当前会话')
        if preview['status'] != 'WAITING_CONFIRMATION' or preview.get('amount_cents') != current['amount_cents']:
            raise DomainError('TOOL_BAD_RESULT', '退款预览回执与业务记录不一致')
        nonce = preview.get('confirm_nonce')
        if nonce:
            self.business.check_refund_nonce(ctx.principal, wid, nonce)
        current = await self.start(ctx, services, wid)
        if current['status'] != 'WAITING_CONFIRMATION':
            return self.read_result(ctx.principal, current), None
        emit('workflow', to='WAITING_CONFIRMATION', workflow_id=wid)
        pending = dict(workflow_id=wid, order_id=oid, status=current['status'], amount_cents=current['amount_cents'])
        if nonce:
            pending['confirm_nonce'] = nonce
        else:
            pending['confirmation_credential_available'] = False
        return RefundWorkflowResult(wid, oid, 'WAITING_CONFIRMATION'), pending

    def verify(self, actor, wid, oid, *, receipt=None, response_missing=False):
        if response_missing:
            emit('recovery.started', action='lookup_original_operation', workflow_id=wid,
                 error_class='WRITTEN_BUT_RESPONSE_UNKNOWN')
        try:
            actual = self.business.refund_outcome(actor, wid, order_id=oid)
        except (SQLAlchemyError, DomainError, TimeoutError, OSError):
            actual = None
            reason = 'LOOKUP_UNAVAILABLE'
        else:
            reason = 'MISSING_OR_INCONSISTENT_RECORD'
        if actual and receipt is not None and (
            any(receipt.get(key) != actual.get(key) for key in ('refund_request_id', 'status'))
            or ('amount_cents' in receipt and (type(receipt['amount_cents']) is not int
                                              or receipt['amount_cents'] != actual.get('amount_cents')))
        ):
            actual = None
            reason = 'RECEIPT_MISMATCH'
        if actual and actual['status'] == 'ACCEPTED':
            result = RefundWorkflowResult(
                wid, oid, 'VERIFIED_SUCCESS', refund_request_id=actual['refund_request_id'],
                amount_cents=actual['amount_cents'], business_status='ACCEPTED', verified=True,
                idempotent_replay=receipt is None or receipt.get('idempotent_replay') is True,
                created_at=actual['created_at'],
            )
        elif actual and actual['status'] == 'REJECTED':
            reason = '业务系统已拒绝该申请' if actual.get('refund_request_id') else PREVIEW_CHANGED
            result = self.failed(wid, oid, DomainError('BUSINESS_RULE_FAILED', reason))
        else:
            result = RefundWorkflowResult(wid, oid, 'UNKNOWN', error_code='UNKNOWN_OUTCOME', reason=reason)
        if response_missing:
            emit('recovery.completed', action='lookup_original_operation',
                 result='found' if result.verified else result.status.lower(), workflow_id=wid)
        emit('workflow.verified', workflow_id=wid, order_id=oid, result=result.status,
             verified=result.verified, error_code=result.error_code, reason=result.reason)
        return result

    def graph(self, actor, saver, ctx=None, services=None):
        async def confirmation(state):
            wid = state['workflow_id']
            interrupt({'kind': 'refund_confirmation', 'workflow_id': wid})
            workflow = self.business.workflow(actor, wid)
            if workflow['status'] not in ('CONFIRMED', 'SUBMITTING', 'REQUEST_ACCEPTED', 'FAILED'):
                raise DomainError('FORBIDDEN')
            return {'order_id': workflow['order_id']}

        async def commit(state):
            wid = state['workflow_id']
            if ctx is None or services is None or wid != ctx.workflow_id:
                raise DomainError('FORBIDDEN')
            ctx.remaining()
            if not self.business.claim_refund_submission(
                actor, wid, conversation_id=ctx.conversation_id, graph_thread_id=ctx.graph_thread_id,
            ):
                return {}
            services.budget.write_started = True
            services.controller.update(ctx, write_started=1)
            services.telemetry.emit('workflow.transition', workflow_id=wid, previous='CONFIRMED', next='SUBMITTING')
            emit('workflow.commit.started', workflow_id=wid)
            task = services.controller.retain(
                asyncio.create_task(asyncio.to_thread(self.business.submit, actor, wid))
            )
            try:
                async with asyncio.timeout(ctx.remaining()):
                    response = await asyncio.shield(task)
            except DomainError as error:
                emit('workflow.commit.completed', workflow_id=wid, error_code=error.code, status='error')
                return {'receipt': None, 'response_missing': error.code == 'UNKNOWN_OUTCOME'}
            except (TimeoutError, SQLAlchemyError, OSError):
                emit('workflow.commit.completed', workflow_id=wid, error_code='UNKNOWN_OUTCOME', status='error')
                return {'receipt': None, 'response_missing': True}
            receipt = {key: response[key] for key in ('refund_request_id', 'status', 'amount_cents', 'idempotent_replay')
                       if key in response} if isinstance(response, dict) else {}
            emit('workflow.commit.completed', workflow_id=wid, receipt=receipt,
                 status='returned' if receipt else 'error')
            return {'receipt': receipt, 'response_missing': False}

        def verify(state):
            wid = state['workflow_id']
            oid = state.get('order_id') or self.business.workflow(actor, wid)['order_id']
            result = self.verify(actor, wid, oid, receipt=state.get('receipt'),
                                 response_missing=state.get('response_missing', False))
            return {'result': asdict(result)}

        graph = StateGraph(RefundState)
        graph.add_node('confirmation', confirmation)
        graph.add_node('commit', commit)
        graph.add_node('verify', verify)
        graph.add_edge(START, 'confirmation')
        graph.add_edge('confirmation', 'commit')
        graph.add_edge('commit', 'verify')
        graph.add_edge('verify', END)
        return graph.compile(checkpointer=saver)

    async def ensure_checkpoint(self, graph, config, wid):
        state = await graph.aget_state(config)
        if not state.values or ('confirmation' in state.next and not state.interrupts):
            await graph.ainvoke(None if state.values else {'workflow_id': wid}, config, durability='sync')
            state = await graph.aget_state(config)
        return state

    async def start(self, ctx, services, wid):
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        thread = 'wf:' + wid
        async with services.controller.locks.acquire(thread):
            current = self.business.workflow(ctx.principal, wid)
            if current['status'] != 'WAITING_CONFIRMATION':
                return current
            async with AsyncSqliteSaver.from_conn_string(self.checkpoint_path) as saver:
                graph = self.graph(ctx.principal, saver)
                state = await self.ensure_checkpoint(graph, {'configurable': {'thread_id': thread}}, wid)
                if not state.interrupts or state.interrupts[0].value['kind'] != 'refund_confirmation':
                    raise DomainError('SYSTEM_ERROR', '工作流未持久化确认中断')
                services.telemetry.emit('workflow.interrupted', workflow_id=wid, graph_thread_id=thread,
                                        interrupt_kind='refund_confirmation')
        return current

    async def confirm(self, actor, workflow, nonce, ctx, services):
        wid, oid = workflow['id'], workflow['order_id']
        try:
            confirmed = self.business.confirm(actor, wid, nonce)
        except DomainError as error:
            if error.code != 'BUSINESS_RULE_FAILED':
                raise
            return self.failed(wid, oid, error)
        services.telemetry.emit('workflow.transition', workflow_id=wid, previous=workflow['status'],
                                next=confirmed['status'], version=confirmed['version'])
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(self.checkpoint_path) as saver:
            graph = self.graph(actor, saver, ctx, services)
            config = {'configurable': {'thread_id': confirmed['graph_thread_id']}}
            if confirmed['status'] == 'CONFIRMED':
                state = await self.ensure_checkpoint(graph, config, wid)
            else:
                state = await graph.aget_state(config)
                if state.next != ('verify',):
                    return self.verify(actor, wid, oid)
            services.telemetry.emit('workflow.resumed', workflow_id=wid, graph_thread_id=confirmed['graph_thread_id'],
                                    new_invocation_id=ctx.invocation_id)
            if state.next:
                output = await graph.ainvoke(Command(resume=True) if state.interrupts else None, config, durability='sync')
                return RefundWorkflowResult(**output['result'])
        return self.verify(actor, wid, oid)

    async def reconcile(self, actor, workflow, *, response_missing=False):
        """Read the original result without dispatching or advancing its checkpoint."""
        wid, oid = workflow['id'], workflow['order_id']
        receipt = None
        if Path(self.checkpoint_path).is_file():
            try:
                async with AsyncSqliteSaver.from_conn_string(self.checkpoint_path) as saver:
                    graph = self.graph(actor, saver)
                    state = await graph.aget_state({'configurable': {'thread_id': workflow['graph_thread_id']}})
                    if state.next == ('verify',):
                        receipt = state.values.get('receipt')
            except (SQLiteError, OSError):
                return RefundWorkflowResult(wid, oid, 'UNKNOWN', error_code='UNKNOWN_OUTCOME',
                                            reason='CHECKPOINT_UNAVAILABLE')
        return self.verify(actor, wid, oid, receipt=receipt, response_missing=response_missing)

    def read_result(self, actor, workflow):
        wid, oid, status = workflow['id'], workflow['order_id'], workflow['status']
        if status == 'WAITING_CONFIRMATION':
            return RefundWorkflowResult(wid, oid, 'WAITING_CONFIRMATION')
        if status == 'CANCELLED':
            return self.failed(wid, oid, DomainError('BUSINESS_RULE_FAILED', '原退款申请已取消'))
        return self.verify(actor, wid, oid)

    def status_for_order(self, actor, oid):
        workflow, legacy_receipt = self.business.refund_status_for_order(actor, oid)
        if workflow:
            result = self.verify(actor, workflow['id'], oid)
            receipt = None
            if result.verified:
                receipt = dict(refund_request_id=result.refund_request_id, status=result.business_status,
                               amount_cents=result.amount_cents, created_at=result.created_at)
            return result.reply(), receipt
        if legacy_receipt:
            if legacy_receipt['status'] == 'ACCEPTED':
                return RefundWorkflowResult(None, oid, 'VERIFIED_SUCCESS',
                                            refund_request_id=legacy_receipt['refund_request_id'],
                                            business_status='ACCEPTED', verified=True).reply(), legacy_receipt
            return '退款申请状态：' + legacy_receipt['status'], legacy_receipt
        return '该订单目前没有已受理的退款申请。', None
