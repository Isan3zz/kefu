"""FastMCP boundary: actor credentials are transport metadata, never tool arguments."""
import base64
import hashlib
import hmac
import json
import time
import asyncio
from contextvars import ContextVar
from fastmcp import FastMCP, Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.dependencies import get_http_headers
from app.config import settings
from app.business import Actor, Business, DomainError, PriceCents
from app.db import DB, uid
from app.observability import emit
from pydantic import BaseModel,ConfigDict,StrictBool,model_validator
from typing import Any

class ToolFailure(BaseModel):
    model_config=ConfigDict(extra='forbid')
    code:str
    message:str
    retryable:bool=False

class ToolResponse(BaseModel):
    model_config=ConfigDict(extra='forbid')
    ok:bool
    data:Any=None
    error:ToolFailure|None=None
    request_id:str
    @model_validator(mode='after')
    def consistent_outcome(self):
        if self.ok and self.error is not None or not self.ok and self.error is None: raise ValueError('Inconsistent tool success/error state')
        if not self.ok and self.data is not None: raise ValueError('Failed tools must not expose business data')
        return self

TOOL_SCHEMA_VERSION='v4'

def decode_tool_result(result):
    if getattr(result,'is_error',False) or getattr(result,'isError',False): raise DomainError('TOOL_UPSTREAM_ERROR','工具服务未返回有效业务结果')
    value=result.data
    if isinstance(value,BaseModel): value=value.model_dump(mode='json')
    if not isinstance(value,dict): value=json.loads(result.content[0].text)
    return ToolResponse.model_validate(value).model_dump(mode='json')

def sign_actor(a, secret, expires=None, *, ticket_operation=None, refund_preview_scope=None):
    body=json.dumps(dict(user_id=a.user_id,tenant_id=a.tenant_id,exp=expires or time.time()+60,
                         aud="mcp",nonce=uid(),ticket_operation=ticket_operation,
                         refund_preview_scope=refund_preview_scope),sort_keys=True).encode()
    payload=base64.urlsafe_b64encode(body).decode()
    signature=hmac.new(secret.encode(),payload.encode(),hashlib.sha256).hexdigest()
    return payload+"."+signature

def verify_actor(token,secret,business):
    try:
        payload,sig=token.split('.')
        expected=hmac.new(secret.encode(),payload.encode(),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected,sig): raise ValueError()
        d=json.loads(base64.urlsafe_b64decode(payload))
        if d['aud']!='mcp' or d['exp']<time.time() or not d['nonce']: raise ValueError()
        scope=d.get('refund_preview_scope')
        if scope is not None and (
                not isinstance(scope,list) or len(scope)!=2 or any(not isinstance(value,str) or not value for value in scope)
        ): raise ValueError()
        from app.db import users
        with business.db.tx() as c:
            r=c.execute(users.select().where(users.c.id==d['user_id'],users.c.tenant_id==d['tenant_id'])).mappings().first()
            if not r: raise ValueError()
            return Actor(r['id'],r['tenant_id'],r['role'],d.get('ticket_operation'),tuple(scope) if scope else None)
    except (ValueError,KeyError,TypeError): raise DomainError("FORBIDDEN","内部身份凭据无效") from None

# Used only by explicit in-process test transport; production always reads HTTP headers.
test_credential=ContextVar("test_credential",default="")
def create_mcp(business, secret, testing=False):
    m=FastMCP("kefu-business")
    def execute(name,args):
        req=uid("tool-")
        try:
            token=test_credential.get() if testing else get_http_headers().get('x-internal-auth','').removeprefix('Bearer ')
            actor=verify_actor(token,secret,business)
            return ToolResponse(ok=True,data=business.tool(actor,name,args),error=None,request_id=req)
        except DomainError as e:
            return ToolResponse(ok=False,data=None,error=ToolFailure(code=e.code,message=e.message,retryable=False),request_id=req)
    @m.tool
    def get_orders(limit:int=10)->ToolResponse:
        """List authenticated customer's orders."""
        return execute('get_orders',dict(limit=limit))
    @m.tool
    def get_order(order_id:str)->ToolResponse:
        """Read an owned order."""
        return execute('get_order',dict(order_id=order_id))
    @m.tool
    def track_shipment(order_id:str)->ToolResponse:
        """Track an owned order."""
        return execute('track_shipment',dict(order_id=order_id))
    @m.tool
    def search_products(
        keyword: str = '',
        limit: int = 10,
        min_price_cents: PriceCents | None = None,
        max_price_cents: PriceCents | None = None,
        in_stock_only: StrictBool = False,
    ) -> ToolResponse:
        """Search tenant products with optional inclusive price bounds and an in-stock filter."""
        return execute('search_products', dict(
            keyword=keyword, limit=limit, min_price_cents=min_price_cents,
            max_price_cents=max_price_cents, in_stock_only=in_stock_only,
        ))
    @m.tool
    def check_refund_eligibility(order_id:str)->ToolResponse:
        """Compute deterministic refund eligibility."""
        return execute('check_refund_eligibility',dict(order_id=order_id))
    @m.tool
    def begin_refund_workflow(order_id:str)->ToolResponse:
        """Prepare a confirmation preview for an owned order."""
        return execute('begin_refund_workflow',dict(order_id=order_id))
    @m.tool
    def create_ticket(category:str,summary:str,conversation_id:str)->ToolResponse:
        """Create a scoped handoff ticket."""
        return execute('create_ticket',dict(category=category,summary=summary,conversation_id=conversation_id))
    return m

ALLOW={
 'order-tracking':{'get_orders','get_order','track_shipment'},
 'refund-processing':{'get_order','check_refund_eligibility','begin_refund_workflow'},
 'product-consultation':{'search_products'},
 'complaint-handling':{'create_ticket'},
}
class Gateway:
    def __init__(self,business,config=settings,server=None):
        self.business,self.config,self.server=business,config,server
    async def call(self,a,name,args,skill=None,*,single_attempt=False,timeout=None,operation_id=None,attempt_no=None,ticket_operation=None,refund_preview_scope=None):
        start=time.perf_counter(); call_id=uid('gateway-')
        emit('tool_attempt',name=name,args=args,request_id=call_id,skill=skill,operation_id=operation_id,attempt_no=attempt_no)
        try:
            result=await self._call(a,name,args,skill,single_attempt=single_attempt,timeout=timeout,
                                    ticket_operation=ticket_operation,refund_preview_scope=refund_preview_scope)
        except Exception as error:
            code=error.code if isinstance(error,DomainError) else 'TOOL_EXECUTION_ERROR'
            emit('tool',name=name,args=args,request_id=call_id,status='blocked' if code=='FORBIDDEN' else 'error',error_code=code,ok=False,duration_ms=(time.perf_counter()-start)*1000,transport='inprocess' if self.server else 'http',layer='transport',operation_id=operation_id,attempt_no=attempt_no)
            raise
        emit('tool',name=name,args=args,request_id=call_id,result=result,ok=result['ok'],status='success' if result['ok'] else 'error',error_code=result['error']['code'] if result.get('error') else None,duration_ms=(time.perf_counter()-start)*1000,transport='inprocess' if self.server else 'http',layer='transport',operation_id=operation_id,attempt_no=attempt_no)
        return result
    async def _call(self,a,name,args,skill=None,*,single_attempt=False,timeout=None,ticket_operation=None,refund_preview_scope=None):
        if name not in ALLOW.get(skill,set()): raise DomainError('FORBIDDEN','工具不在白名单')
        token=sign_actor(a,self.config.mcp_service_secret,ticket_operation=ticket_operation,
                         refund_preview_scope=refund_preview_scope)
        for attempt in range(1 if single_attempt else 3 if name in {'get_orders','get_order','track_shipment','search_products','check_refund_eligibility'} else 1):
            marker=test_credential.set(token)
            try:
                transport=self.server or StreamableHttpTransport(self.config.mcp_base_url,headers={'X-Internal-Auth':'Bearer '+token})
                async with Client(transport) as client:
                    r=await asyncio.wait_for(client.call_tool(name,args),timeout=timeout or 15)
                result=decode_tool_result(r)
                return result
            except Exception as e:
                if single_attempt: raise
                if name=='begin_refund_workflow':
                    raise DomainError('UNKNOWN_OUTCOME','退款预览状态暂未确认，请核对原操作') from e
                if name=='create_ticket': raise DomainError('UNKNOWN_OUTCOME','工单写入结果未知，请先核对原会话记录，不会自动重试写入。') from e
                if attempt==2: raise DomainError('UPSTREAM_UNAVAILABLE') from e
                await asyncio.sleep(.1*2**attempt)
            finally: test_credential.reset(marker)

if __name__=='__main__':
    db=DB(settings.sqlite_url); db.initialize()
    create_mcp(Business(db),settings.mcp_service_secret).run(transport='http',host='127.0.0.1',port=8001)
