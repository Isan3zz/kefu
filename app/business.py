import hmac
import secrets
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from app.db import *

class DomainError(Exception):
    def __init__(self, code, message=None):
        self.code, self.message = code, message or code
        super().__init__(self.message)

@dataclass(frozen=True)
class Actor:
    user_id: str
    tenant_id: str
    role: str = "customer"
    ticket_operation: str | None = None
    refund_preview_scope: tuple[str, str] | None = None


PriceCents = Annotated[int, Field(strict=True, ge=0, le=(1 << 63) - 1)]


class ProductSearch(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    keyword: str = Field(default='', max_length=1000)
    limit: int = Field(default=10, ge=1, le=20)
    min_price_cents: PriceCents | None = Field(default=None, description='Inclusive minimum price in cents.')
    max_price_cents: PriceCents | None = Field(default=None, description='Inclusive maximum price in cents.')
    in_stock_only: bool = Field(default=False, description='When true, return only products with stock greater than zero.')

    @model_validator(mode='after')
    def ordered_price_range(self):
        if self.min_price_cents is not None and self.max_price_cents is not None and self.min_price_cents > self.max_price_cents:
            raise ValueError('min_price_cents must be less than or equal to max_price_cents')
        return self


class Business:
    policy = "P-REFUND-v1"
    def __init__(self, db, clock=lambda: datetime.now(timezone.utc)):
        self.db, self.clock = db, clock
    def owned(self, c, table, actor, ident):
        row = c.execute(select(table).where(table.c.id==ident, table.c.tenant_id==actor.tenant_id, table.c.user_id==actor.user_id)).mappings().first()
        if not row: raise DomainError("NOT_FOUND")
        return dict(row)
    def login(self, key):
        with self.db.tx() as c:
            row = c.execute(select(users).where(users.c.id==key)).mappings().first()
            if not row: raise DomainError("NOT_FOUND")
            token=secrets.token_urlsafe(32)
            c.execute(auth_sessions.insert().values(token_hash=digest(token),user_id=key,expires_at=(self.clock()+timedelta(hours=8)).isoformat()))
            return {"token":token,"user":dict(row)}
    def authenticate(self, token):
        with self.db.tx() as c:
            row=c.execute(select(users,auth_sessions.c.expires_at,auth_sessions.c.revoked_at).join(auth_sessions).where(auth_sessions.c.token_hash==digest(token))).mappings().first()
            if not row or row['revoked_at'] or datetime.fromisoformat(row['expires_at'])<=self.clock(): raise DomainError("UNAUTHORIZED")
            return Actor(row['id'],row['tenant_id'],row['role'])
    def conversation(self, actor):
        result=dict(id=uid("conv-"),tenant_id=actor.tenant_id,user_id=actor.user_id,title="客服咨询",created_at=now(),updated_at=now())
        with self.db.tx() as c: c.execute(conversations.insert().values(**result))
        return result
    def eligibility(self, c, a, oid):
        o=self.owned(c,orders,a,oid)
        reason="ELIGIBLE"
        if type(o['paid_amount_cents']) is not int or o['paid_amount_cents']<=0: reason="INVALID_AMOUNT"
        elif c.execute(select(refund_requests.c.id).where(refund_requests.c.order_id==oid)).first(): reason="ALREADY_EXISTS"
        elif o['status']=="SHIPPED": reason="HUMAN_REVIEW"
        elif o['status']=='PAID' and any(r['status'] not in ('待发货','PENDING','CREATED') for r in c.execute(select(shipments.c.status).where(shipments.c.order_id==oid)).mappings()): reason='HUMAN_REVIEW'
        elif o['status']=="DELIVERED":
            if not o['delivered_at']: reason="MISSING_DELIVERY_TIME"
            elif not timedelta(0)<=self.clock()-datetime.fromisoformat(o['delivered_at'])<=timedelta(days=7): reason="EXPIRED"
        elif o['status']!="PAID": reason="INVALID_STATUS"
        return dict(eligible=reason=="ELIGIBLE",reason_code=reason,amount_cents=o['paid_amount_cents'],policy_version=self.policy,requires_human=reason in ("HUMAN_REVIEW","MISSING_DELIVERY_TIME"))
    def preview(self,a,conv,oid,*,reuse_pending=False,invocation_id=None):
        with self.db.tx() as c:
            self.owned(c,conversations,a,conv)
            o=self.owned(c,orders,a,oid)
            if reuse_pending:
                prior=self._refund_workflow_for_order(c,a,oid)
                if prior:
                    if prior['conversation_id']==conv:
                        self._link_preview(c,a,conv,invocation_id,prior['id'])
                    return dict(workflow_id=prior['id'],order_id=oid,status=prior['status'])
            e=self.eligibility(c,a,oid)
            if not e['eligible']: raise DomainError("BUSINESS_RULE_FAILED",e['reason_code'])
            if reuse_pending:
                prior=c.execute(select(workflow_instances).where(
                    workflow_instances.c.tenant_id==a.tenant_id,workflow_instances.c.user_id==a.user_id,
                    workflow_instances.c.conversation_id==conv,workflow_instances.c.order_id==oid,
                    workflow_instances.c.status=='WAITING_CONFIRMATION',workflow_instances.c.order_version==o['version'],
                    workflow_instances.c.amount_cents==e['amount_cents'],workflow_instances.c.policy_version==self.policy,
                    workflow_instances.c.nonce_expires_at>self.clock().isoformat()).order_by(workflow_instances.c.updated_at.desc())).mappings().first()
                if prior:
                    self._link_preview(c,a,conv,invocation_id,prior['id'])
                    return dict(workflow_id=prior['id'],order_id=oid,amount_cents=prior['amount_cents'],
                                status=prior['status'],confirmation_credential_available=False)
            nonce=secrets.token_urlsafe(32); wid=uid("wf-")
            row=dict(id=wid,tenant_id=a.tenant_id,user_id=a.user_id,conversation_id=conv,graph_thread_id="wf:"+wid,type="refund",status="WAITING_CONFIRMATION",order_id=oid,amount_cents=e['amount_cents'],policy_version=self.policy,confirm_nonce_hash=digest(nonce),nonce_expires_at=(self.clock()+timedelta(minutes=15)).isoformat(),operation_key=wid+":refund",order_version=o['version'],version=1,updated_at=now())
            c.execute(workflow_instances.insert().values(**row))
            self._link_preview(c,a,conv,invocation_id,wid)
            return {"workflow_id":wid,"order_id":oid,"amount_cents":e['amount_cents'],"confirm_nonce":nonce,"status":row['status']}
    def _link_preview(self,c,a,conv,invocation_id,wid):
        if invocation_id is None:
            return
        changed=c.execute(agent_invocations.update().where(
            agent_invocations.c.invocation_id==invocation_id,agent_invocations.c.conversation_id==conv,
            agent_invocations.c.tenant_id==a.tenant_id,agent_invocations.c.user_id==a.user_id,
            agent_invocations.c.status=='RUNNING',agent_invocations.c.graph_thread_id=='conv:'+conv,
        ).values(workflow_id=wid,write_started=1,updated_at=now()))
        if changed.rowcount != 1:
            raise DomainError("NOT_FOUND")
    def preview_outcome(self,c,a,conv,wid):
        w=c.execute(select(workflow_instances).where(
            workflow_instances.c.id==wid,workflow_instances.c.conversation_id==conv,
            workflow_instances.c.tenant_id==a.tenant_id,workflow_instances.c.user_id==a.user_id,
            workflow_instances.c.type=='refund')).mappings().first()
        if w is None:
            return None
        return dict(workflow_id=wid,order_id=w['order_id'],amount_cents=w['amount_cents'],status=w['status'],
                    confirmation_credential_available=False,
                    confirmation_expired=datetime.fromisoformat(w['nonce_expires_at'])<=self.clock())
    def workflow(self,a,wid):
        with self.db.tx() as c:
            row=self.owned(c,workflow_instances,a,wid)
            return {k:v for k,v in row.items() if k not in ('confirm_nonce_hash','operation_key')}
    def confirm(self,a,wid,nonce):
        with self.db.tx() as c:
            w=self.owned(c,workflow_instances,a,wid)
            self._check_refund_nonce(w,nonce)
            if w['status'] in ("CONFIRMED","SUBMITTING","REQUEST_ACCEPTED","FAILED"): return w
            if w['status']!="WAITING_CONFIRMATION" or datetime.fromisoformat(w['nonce_expires_at'])<=self.clock(): raise DomainError("BUSINESS_RULE_FAILED","确认已失效")
            self._validate_preview(c,a,w)
            c.execute(workflow_instances.update().where(workflow_instances.c.id==wid,workflow_instances.c.status=="WAITING_CONFIRMATION").values(status="CONFIRMED",version=w['version']+1,updated_at=now()))
            w['status']="CONFIRMED"
            return w
    def check_refund_nonce(self,a,wid,nonce):
        with self.db.tx() as c:
            self._check_refund_nonce(self.owned(c,workflow_instances,a,wid),nonce)
    @staticmethod
    def _check_refund_nonce(w,nonce):
        if w['type']!='refund' or not hmac.compare_digest(w['confirm_nonce_hash'],digest(nonce)):
            raise DomainError("FORBIDDEN")
    def _refund_workflow_for_order(self,c,a,oid):
        recorded=select(refund_requests.c.id).where(
            refund_requests.c.idempotency_key==workflow_instances.c.operation_key).exists()
        return c.execute(select(workflow_instances).where(
            workflow_instances.c.tenant_id==a.tenant_id,workflow_instances.c.user_id==a.user_id,
            workflow_instances.c.order_id==oid,workflow_instances.c.type=='refund',
            workflow_instances.c.status.in_(('CONFIRMED','SUBMITTING','REQUEST_ACCEPTED')) | recorded,
        ).order_by((workflow_instances.c.status=='REQUEST_ACCEPTED').desc(),
                   (workflow_instances.c.status=='SUBMITTING').desc(),
                   workflow_instances.c.updated_at.desc())).mappings().first()
    def refund_status_for_order(self,a,oid):
        with self.db.tx() as c:
            self.owned(c,orders,a,oid)
            workflow=self._refund_workflow_for_order(c,a,oid)
            if workflow:
                return dict(workflow),None
            # Imported legacy receipts have no workflow to reconcile.
            linked=select(workflow_instances.c.id).where(
                workflow_instances.c.operation_key==refund_requests.c.idempotency_key).exists()
            receipt=c.execute(select(refund_requests.c.id.label('refund_request_id'),refund_requests.c.status,
                                     refund_requests.c.amount_cents,refund_requests.c.created_at).where(
                refund_requests.c.order_id==oid,refund_requests.c.user_id==a.user_id,
                refund_requests.c.tenant_id==a.tenant_id,~linked)).mappings().first()
            return None,dict(receipt) if receipt else None
    def claim_refund_submission(self,a,wid,*,conversation_id=None,graph_thread_id=None):
        with self.db.tx() as c:
            w=self.owned(c,workflow_instances,a,wid)
            self.owned(c,orders,a,w['order_id'])
            if w['type']!='refund' or w['operation_key']!=wid+':refund':
                raise DomainError('FORBIDDEN')
            if (conversation_id is not None and w['conversation_id']!=conversation_id
                    or graph_thread_id is not None and w['graph_thread_id']!=graph_thread_id):
                raise DomainError('FORBIDDEN')
            if w['status']!='CONFIRMED':
                return False
            prior=self._refund_workflow_for_order(c,a,w['order_id'])
            if prior and prior['id']!=wid and prior['status'] in ('SUBMITTING','REQUEST_ACCEPTED'):
                return False
            legacy_attempt=c.execute(select(agent_invocations.c.invocation_id).where(
                agent_invocations.c.graph_thread_id==w['graph_thread_id'],
                agent_invocations.c.workflow_id==wid,agent_invocations.c.write_started==1,
                agent_invocations.c.tenant_id==a.tenant_id,agent_invocations.c.user_id==a.user_id,
            ).limit(1)).first()
            if legacy_attempt:
                return False
            c.execute(workflow_instances.update().where(workflow_instances.c.id==wid).values(
                status='SUBMITTING',version=w['version']+1,updated_at=now()))
            return True
    def _validate_preview(self,c,a,w):
        e=self.eligibility(c,a,w['order_id']); o=self.owned(c,orders,a,w['order_id'])
        if not e['eligible'] or e['amount_cents']!=w['amount_cents'] or o['version']!=w['order_version'] or w['policy_version']!=self.policy:
            raise DomainError("BUSINESS_RULE_FAILED","订单或资格已变化，请重新申请预览")
    def submit(self,a,wid):
        rejection=None
        with self.db.tx() as c:
            w=self.owned(c,workflow_instances,a,wid)
            if w['type']!='refund' or w['operation_key']!=wid+':refund':
                raise DomainError('FORBIDDEN')
            old=c.execute(select(refund_requests).where(refund_requests.c.idempotency_key==w['operation_key'])).mappings().first()
            if old: return dict(refund_request_id=old['id'],status=old['status'],idempotent_replay=True)
            if w['status'] not in ('CONFIRMED','SUBMITTING'): raise DomainError("FORBIDDEN","退款尚未确认")
            try:
                self._validate_preview(c,a,w)
            except DomainError as error:
                if error.code!='BUSINESS_RULE_FAILED':
                    raise
                rejection=error
                c.execute(workflow_instances.update().where(workflow_instances.c.id==wid).values(
                    status='FAILED',updated_at=now(),version=w['version']+1))
            else:
                rid=uid("refund-")
                c.execute(refund_requests.insert().values(id=rid,tenant_id=a.tenant_id,user_id=a.user_id,order_id=w['order_id'],amount_cents=w['amount_cents'],status="ACCEPTED",idempotency_key=w['operation_key'],policy_version=self.policy,created_at=now()))
                c.execute(workflow_instances.update().where(workflow_instances.c.id==wid).values(status="REQUEST_ACCEPTED",updated_at=now(),version=w['version']+1))
                return dict(refund_request_id=rid,status="ACCEPTED",idempotent_replay=False)
        raise rejection
    def refund_outcome(self,a,wid,*,order_id=None):
        """Read the outcome by the original operation, including a persisted rejection."""
        with self.db.tx() as c:
            w=self.owned(c,workflow_instances,a,wid)
            self.owned(c,orders,a,w['order_id'])
            if w['type']!='refund' or w['operation_key']!=wid+':refund' or (order_id is not None and w['order_id']!=order_id):
                return None
            row=c.execute(select(refund_requests).where(refund_requests.c.idempotency_key==w['operation_key'])).mappings().first()
            if row is None:
                if w['status']=='FAILED':
                    return dict(status='REJECTED')
                return None
            if (not row['id'] or row['tenant_id']!=a.tenant_id or row['user_id']!=a.user_id or row['order_id']!=w['order_id']
                    or type(row['amount_cents']) is not int or row['amount_cents']<=0
                    or row['amount_cents']!=w['amount_cents'] or row['policy_version']!=w['policy_version']):
                return None
            if (w['status'],row['status']) not in (('REQUEST_ACCEPTED','ACCEPTED'),('FAILED','REJECTED')):
                return None
            return dict(refund_request_id=row['id'],status=row['status'],amount_cents=row['amount_cents'],
                        created_at=row['created_at'])
    def cancel(self,a,wid):
        with self.db.tx() as c:
            w=self.owned(c,workflow_instances,a,wid)
            if w['status'] not in ("WAITING_CONFIRMATION","CANCELLED"): raise DomainError("BUSINESS_RULE_FAILED")
            c.execute(workflow_instances.update().where(workflow_instances.c.id==wid).values(status="CANCELLED",updated_at=now()))
        return self.workflow(a,wid)
    @staticmethod
    def ticket_id(a,conv,invocation_id):
        return 'ticket-'+digest(f"{a.tenant_id}:{a.user_id}:{conv}:{invocation_id}")[:24]
    def ticket_outcome(self,c,a,conv,invocation_id):
        tid=self.ticket_id(a,conv,invocation_id)
        row=c.execute(select(tickets.c.status).where(tickets.c.id==tid,tickets.c.tenant_id==a.tenant_id,
            tickets.c.user_id==a.user_id,tickets.c.conversation_id==conv)).first()
        return dict(ticket_id=tid,status=row.status) if row else None
    def tool(self,a,name,args):
        if name=='begin_refund_workflow':
            if not a.refund_preview_scope:
                raise DomainError('FORBIDDEN', '退款预览需要受信任的会话与执行上下文')
            conversation_id,invocation_id=a.refund_preview_scope
            try:
                return self.preview(a,conversation_id,args['order_id'],reuse_pending=True,invocation_id=invocation_id)
            except DomainError as error:
                if error.code!='BUSINESS_RULE_FAILED':
                    raise
                return dict(workflow_id=None,order_id=args['order_id'],status='FAILED',reason_code=error.message)
        with self.db.tx() as c:
            if name=="get_orders":
                return [dict(r) for r in c.execute(select(orders).where(orders.c.tenant_id==a.tenant_id,orders.c.user_id==a.user_id).order_by(orders.c.paid_at.desc(),orders.c.id).limit(max(1,min(int(args.get('limit',10)),20)))).mappings()]
            if name=="get_order":
                o=self.owned(c,orders,a,args['order_id'])
                o['items']=[dict(r) for r in c.execute(select(order_items).where(order_items.c.order_id==o['id'])).mappings()]
                refund=c.execute(select(refund_requests.c.id.label('refund_request_id'),refund_requests.c.status,refund_requests.c.amount_cents,refund_requests.c.created_at).where(refund_requests.c.order_id==o['id'],refund_requests.c.tenant_id==a.tenant_id,refund_requests.c.user_id==a.user_id)).mappings().first()
                o['refund_request']=dict(refund) if refund else None
                return o
            if name=="track_shipment":
                self.owned(c,orders,a,args['order_id'])
                return [dict(r) for r in c.execute(select(shipments).where(shipments.c.order_id==args['order_id'])).mappings()]
            if name=="check_refund_eligibility": return self.eligibility(c,a,args['order_id'])
            if name=="search_products":
                try:
                    limit = min(max(int(args.get('limit', 10)), 1), 20)
                    search = ProductSearch.model_validate({**args, 'limit':limit})
                except (TypeError, ValueError) as error:
                    raise DomainError('INVALID_ARGUMENT', '商品筛选参数无效') from error
                query = select(products).where(
                    products.c.tenant_id == a.tenant_id,
                    products.c.name.contains(search.keyword),
                )
                if search.min_price_cents is not None:
                    query = query.where(products.c.price_cents >= search.min_price_cents)
                if search.max_price_cents is not None:
                    query = query.where(products.c.price_cents <= search.max_price_cents)
                if search.in_stock_only:
                    query = query.where(products.c.stock > 0)
                return [dict(row) for row in c.execute(query.order_by(products.c.id).limit(search.limit)).mappings()]
            if name=="create_ticket":
                self.owned(c,conversations,a,args['conversation_id'])
                if a.ticket_operation:
                    old=self.ticket_outcome(c,a,args['conversation_id'],a.ticket_operation)
                    if old: return old
                tid=self.ticket_id(a,args['conversation_id'],a.ticket_operation) if a.ticket_operation else uid('ticket-')
                c.execute(tickets.insert().values(id=tid,tenant_id=a.tenant_id,user_id=a.user_id,conversation_id=args['conversation_id'],summary=args['summary'][:2000],reason=args['category'][:100],status="OPEN",created_at=now()))
                return dict(ticket_id=tid,status="OPEN")
            raise DomainError("INVALID_ARGUMENT")
