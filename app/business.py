import hmac
import secrets
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
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
    def preview(self,a,conv,oid):
        with self.db.tx() as c:
            self.owned(c,conversations,a,conv)
            e=self.eligibility(c,a,oid)
            if not e['eligible']: raise DomainError("BUSINESS_RULE_FAILED",e['reason_code'])
            o=self.owned(c,orders,a,oid)
            nonce=secrets.token_urlsafe(32); wid=uid("wf-")
            row=dict(id=wid,tenant_id=a.tenant_id,user_id=a.user_id,conversation_id=conv,graph_thread_id="wf:"+wid,type="refund",status="WAITING_CONFIRMATION",order_id=oid,amount_cents=e['amount_cents'],policy_version=self.policy,confirm_nonce_hash=digest(nonce),nonce_expires_at=(self.clock()+timedelta(minutes=15)).isoformat(),operation_key=wid+":refund",order_version=o['version'],version=1,updated_at=now())
            c.execute(workflow_instances.insert().values(**row))
            return {"workflow_id":wid,"order_id":oid,"amount_cents":e['amount_cents'],"confirm_nonce":nonce,"status":row['status']}
    def workflow(self,a,wid):
        with self.db.tx() as c:
            row=self.owned(c,workflow_instances,a,wid)
            return {k:v for k,v in row.items() if k not in ('confirm_nonce_hash','operation_key')}
    def confirm(self,a,wid,nonce):
        with self.db.tx() as c:
            w=self.owned(c,workflow_instances,a,wid)
            if not hmac.compare_digest(w['confirm_nonce_hash'],digest(nonce)): raise DomainError("FORBIDDEN")
            if w['status'] in ("CONFIRMED","REQUEST_ACCEPTED"): return w
            if w['status']!="WAITING_CONFIRMATION" or datetime.fromisoformat(w['nonce_expires_at'])<=self.clock(): raise DomainError("BUSINESS_RULE_FAILED","确认已失效")
            self._validate_preview(c,a,w)
            c.execute(workflow_instances.update().where(workflow_instances.c.id==wid,workflow_instances.c.status=="WAITING_CONFIRMATION").values(status="CONFIRMED",version=w['version']+1,updated_at=now()))
            w['status']="CONFIRMED"
            return w
    def _validate_preview(self,c,a,w):
        e=self.eligibility(c,a,w['order_id']); o=self.owned(c,orders,a,w['order_id'])
        if not e['eligible'] or e['amount_cents']!=w['amount_cents'] or o['version']!=w['order_version'] or w['policy_version']!=self.policy:
            raise DomainError("BUSINESS_RULE_FAILED","订单或资格已变化，请重新申请预览")
    def submit(self,a,wid):
        with self.db.tx() as c:
            w=self.owned(c,workflow_instances,a,wid)
            old=c.execute(select(refund_requests).where(refund_requests.c.idempotency_key==w['operation_key'])).mappings().first()
            if old: return dict(refund_request_id=old['id'],status=old['status'],idempotent_replay=True)
            if w['status']!="CONFIRMED": raise DomainError("FORBIDDEN","退款尚未确认")
            self._validate_preview(c,a,w)
            rid=uid("refund-")
            c.execute(refund_requests.insert().values(id=rid,tenant_id=a.tenant_id,user_id=a.user_id,order_id=w['order_id'],amount_cents=w['amount_cents'],status="ACCEPTED",idempotency_key=w['operation_key'],policy_version=self.policy,created_at=now()))
            c.execute(workflow_instances.update().where(workflow_instances.c.id==wid).values(status="REQUEST_ACCEPTED",updated_at=now(),version=w['version']+1))
            return dict(refund_request_id=rid,status="ACCEPTED",idempotent_replay=False)
    def refund_outcome(self,a,wid):
        with self.db.tx() as c:
            w=self.owned(c,workflow_instances,a,wid)
            row=c.execute(select(refund_requests).where(refund_requests.c.idempotency_key==w['operation_key'],refund_requests.c.tenant_id==a.tenant_id,refund_requests.c.user_id==a.user_id,refund_requests.c.order_id==w['order_id'])).mappings().first()
            if not row or w['status']!='REQUEST_ACCEPTED' or row['status']!='ACCEPTED' or row['amount_cents']!=w['amount_cents'] or row['policy_version']!=w['policy_version']: return None
            return dict(refund_request_id=row['id'],status=row['status'],idempotent_replay=True)
    def cancel(self,a,wid):
        with self.db.tx() as c:
            w=self.owned(c,workflow_instances,a,wid)
            if w['status'] not in ("WAITING_CONFIRMATION","CANCELLED"): raise DomainError("BUSINESS_RULE_FAILED")
            c.execute(workflow_instances.update().where(workflow_instances.c.id==wid).values(status="CANCELLED",updated_at=now()))
        return self.workflow(a,wid)
    def tool(self,a,name,args):
        if name=="submit_refund_request": return self.submit(a,args['workflow_id'])
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
                return [dict(r) for r in c.execute(select(products).where(products.c.tenant_id==a.tenant_id,products.c.name.contains(args.get('keyword',''))).limit(min(max(int(args.get('limit',10)),1),20))).mappings()]
            if name=="create_ticket":
                self.owned(c,conversations,a,args['conversation_id'])
                tid=uid('ticket-')
                c.execute(tickets.insert().values(id=tid,tenant_id=a.tenant_id,user_id=a.user_id,conversation_id=args['conversation_id'],summary=args['summary'][:2000],reason=args['category'][:100],status="OPEN",created_at=now()))
                return dict(ticket_id=tid,status="OPEN")
            raise DomainError("INVALID_ARGUMENT")
