"""FastMCP boundary: actor credentials are transport metadata, never tool arguments."""
import base64
import hashlib
import hmac
import json
import logging
import time
import asyncio
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit
import httpx
from fastmcp import FastMCP, Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.dependencies import get_context, get_http_headers
from app.config import settings
from app.business import Actor, Business, DomainError, PriceCents
from app.db import DB, uid
from app.observability import active_trace, emit, evaluation_context
from pydantic import BaseModel,ConfigDict,Field,StrictBool,model_validator
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

class RetrievalDiagnostics(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    retrieval_candidate_count: int = Field(default=0, ge=0)
    retrieval_result_count: int = Field(default=0, ge=0)
    model_revision: str | dict[str, str | None] | None = None
    knowledge_snapshot: str | None = None
    stages: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0)
    invocation_id: str
    operation_id: str
    tool_call_id: str | None = None
    attempt_no: int = Field(ge=1)


class RetrievalResponse(ToolResponse):
    diagnostics: RetrievalDiagnostics | None = None


TOOL_SCHEMA_VERSION='v6'
BUSINESS_AUDIENCE = 'kefu-business'
RETRIEVAL_AUDIENCE = 'kefu-retrieval'
AUTH_METADATA = 'kefu.internal_auth'
logger = logging.getLogger(__name__)

def decode_tool_result(result):
    if getattr(result,'is_error',False) or getattr(result,'isError',False): raise DomainError('TOOL_UPSTREAM_ERROR','工具服务未返回有效业务结果')
    value=result.data
    if isinstance(value,BaseModel): value=value.model_dump(mode='json')
    if not isinstance(value,dict): value=json.loads(result.content[0].text)
    schema = RetrievalResponse if 'diagnostics' in value else ToolResponse
    return schema.model_validate(value, strict=True).model_dump(mode='json')

def sign_actor(a, secret, expires=None, *, ticket_operation=None, refund_preview_scope=None,
               audience=BUSINESS_AUDIENCE, scope=None):
    body=json.dumps(dict(user_id=a.user_id,tenant_id=a.tenant_id,exp=time.time()+60 if expires is None else expires,
                         aud=audience,nonce=uid(),ticket_operation=ticket_operation,
                         refund_preview_scope=refund_preview_scope,scope=scope),sort_keys=True).encode()
    payload=base64.urlsafe_b64encode(body).decode()
    signature=hmac.new(secret.encode(),payload.encode(),hashlib.sha256).hexdigest()
    return payload+"."+signature

def verify_request(token, secret, business, *, audience, tool=None):
    try:
        payload,sig=token.split('.')
        expected=hmac.new(secret.encode(),payload.encode(),hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected,sig): raise ValueError()
        d=json.loads(base64.urlsafe_b64decode(payload))
        now = time.time()
        if d['aud'] != audience or not now < d['exp'] <= now+65 or not d['nonce']: raise ValueError()
        call_scope = d.get('scope')
        if tool is not None and (not isinstance(call_scope, dict) or call_scope.get('tool') != tool):
            raise ValueError()
        scope=d.get('refund_preview_scope')
        if scope is not None and (
                not isinstance(scope,list) or len(scope)!=2 or any(not isinstance(value,str) or not value for value in scope)
        ): raise ValueError()
        from app.db import users
        with business.db.tx() as c:
            r=c.execute(users.select().where(users.c.id==d['user_id'],users.c.tenant_id==d['tenant_id'])).mappings().first()
            if not r: raise ValueError()
            actor = Actor(r['id'],r['tenant_id'],r['role'],d.get('ticket_operation'),tuple(scope) if scope else None)
            return actor, call_scope
    except (ValueError,KeyError,TypeError,AttributeError): raise DomainError("FORBIDDEN","内部身份凭据无效") from None


def verify_actor(token, secret, business):
    return verify_request(token, secret, business, audience=BUSINESS_AUDIENCE)[0]


def request_credential():
    context = get_context().request_context
    metadata = context.meta.model_extra if context and context.meta else None
    return (metadata or {}).get(AUTH_METADATA, '')

def create_mcp(business, secret, testing=False):
    m=FastMCP(BUSINESS_AUDIENCE)
    def execute(name,args):
        req=uid("tool-")
        try:
            token = request_credential()
            if not token and not testing:
                token = get_http_headers().get('x-internal-auth','').removeprefix('Bearer ')
            actor, _ = verify_request(token, secret, business, audience=BUSINESS_AUDIENCE, tool=name)
            return ToolResponse(ok=True,data=business.tool(actor,name,args),error=None,request_id=req)
        except DomainError as e:
            return ToolResponse(ok=False,data=None,error=ToolFailure(code=e.code,message=e.message,retryable=False),request_id=req)
    @m.tool
    def get_orders(limit:int=10)->ToolResponse:
        """List authenticated customer's orders."""
        return execute('get_orders',dict(limit=limit))
    @m.tool
    def get_order(order_id:str)->ToolResponse:
        """Read an owned order with items, payment, shipment summary and refund status."""
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
BUSINESS_TOOLS = frozenset().union(*ALLOW.values())
RETRIEVAL_TOOLS = frozenset({
    'search_knowledge', 'search_products_by_image', 'search_text_by_image',
    'search_images_by_text', 'search_products_by_image_and_text',
})
WRITE_TOOLS = {'begin_refund_workflow', 'create_ticket'}
TRANSIENT_READ_ERRORS = {'TOOL_TIMEOUT', 'TOOL_UPSTREAM_ERROR', 'UPSTREAM_UNAVAILABLE'}
RETRIEVAL_UNAVAILABLE_ERRORS = TRANSIENT_READ_ERRORS | {
    'MCP_DISABLED', 'MCP_TOOL_CONFLICT', 'MCP_UNKNOWN_TOOL', 'MCP_SCHEMA_MISMATCH',
    'TOOL_BAD_RESULT', 'KNOWLEDGE_NOT_READY',
}
ALLOW['product-consultation'].update(RETRIEVAL_TOOLS)


@dataclass(frozen=True)
class ToolRegistration:
    name: str
    server_id: str
    kind: str


@dataclass
class ServerRegistration:
    server_id: str
    endpoint: str
    audience: str
    secret: str = field(repr=False)
    enabled: bool = True
    timeout: float = 15
    read_retries: int = 2
    local_server: FastMCP | None = None
    client: Client | None = field(default=None, init=False, repr=False)
    schemas: dict = field(default_factory=dict, init=False, repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)


class Gateway:
    """Route to locally registered tools and hold one verified HTTP session per server."""

    def __init__(self, business, config=settings, server=None, *, retrieval_server=None):
        self.business, self.config = business, config
        self.servers = {
            'business': ServerRegistration(
                'business', config.business_endpoint, BUSINESS_AUDIENCE, config.business_secret,
                config.business_mcp_enabled, config.tool_timeout_seconds, config.tool_read_max_retries, server,
            ),
            'retrieval': ServerRegistration(
                'retrieval', config.retrieval_mcp_url, RETRIEVAL_AUDIENCE, config.retrieval_mcp_service_secret,
                config.retrieval_mcp_enabled, config.retrieval_mcp_timeout_seconds or config.tool_timeout_seconds,
                config.tool_read_max_retries, retrieval_server,
            ),
        }
        self.tools = {}
        for server_id, names in (('business', BUSINESS_TOOLS), ('retrieval', RETRIEVAL_TOOLS)):
            for name in names:
                if name in self.tools:
                    raise ValueError(f'MCP tool registered by multiple servers: {name}')
                kind = 'write' if name in WRITE_TOOLS else 'read'
                self.tools[name] = ToolRegistration(name, server_id, kind)

    def server_for(self, name):
        registered = self.tools.get(name)
        if not registered:
            raise DomainError('FORBIDDEN', '工具不在白名单')
        return self.servers[registered.server_id]

    async def _discover(self, server, client):
        if server.server_id == 'business':
            reference = create_mcp(None, '', testing=True)
        else:
            from app.retrieval_mcp import create_retrieval_mcp
            reference = create_retrieval_mcp(None, self.config)
        expected = await reference.get_tools()
        verified = {}
        for tool in await client.list_tools():
            registered = self.tools.get(tool.name)
            if not registered:
                raise DomainError('MCP_UNKNOWN_TOOL', f'{server.server_id} 声明了未注册工具')
            if registered.server_id != server.server_id or tool.name in verified:
                raise DomainError('MCP_TOOL_CONFLICT', f'{server.server_id} 工具名称冲突: {registered.name}')
            contract = expected[tool.name].to_mcp_tool()
            if (tool.inputSchema, tool.outputSchema) != (contract.inputSchema, contract.outputSchema):
                raise DomainError('MCP_SCHEMA_MISMATCH', f'{server.server_id} 工具定义不一致: {tool.name}')
            verified[tool.name] = (tool.inputSchema, tool.outputSchema)
        if verified.keys() != expected.keys():
            raise DomainError('MCP_SCHEMA_MISMATCH', f'{server.server_id} 缺少注册工具')
        return verified

    @staticmethod
    async def _close_client(client):
        try:
            await client.__aexit__(None, None, None)
        except Exception as error:
            # FastMCP raises the session task's original transport exception
            # when closing an already disconnected session.
            logger.debug('MCP session cleanup after %s', type(error).__name__)

    @staticmethod
    def _http_client(*, headers=None, auth=None, timeout=None, follow_redirects=False):
        return httpx.AsyncClient(headers=headers, auth=auth, trust_env=False,
                                 timeout=timeout or httpx.Timeout(5, read=300))

    async def _connect(self, server):
        async with server.lock:
            if server.client and server.client.is_connected():
                return server.client
            if server.client:
                stale, server.client = server.client, None
                server.schemas = {}
                await self._close_client(stale)
            client = Client(StreamableHttpTransport(server.endpoint, httpx_client_factory=self._http_client))
            # The SDK creates a long-lived task here. Do not retain the first
            # request's trace or evaluation context in the shared session.
            trace = active_trace.set(None)
            evaluation = evaluation_context.set({})
            try:
                await client.__aenter__()
            finally:
                evaluation_context.reset(evaluation)
                active_trace.reset(trace)
            try:
                server.schemas = await self._discover(server, client)
            except BaseException:
                await self._close_client(client)
                raise
            server.client = client
            return client

    async def discover(self, server_id):
        server = self.servers[server_id]
        if not server.enabled:
            raise DomainError('MCP_DISABLED', '工具服务已停用')
        async with asyncio.timeout(server.timeout):
            if server.local_server is not None:
                async with Client(server.local_server) as client:
                    return await self._discover(server, client)
            await self._connect(server)
            return dict(server.schemas)

    async def close(self):
        async def disconnect(server):
            async with server.lock:
                if server.client is not None:
                    client, server.client = server.client, None
                    server.schemas = {}
                    await self._close_client(client)
        await asyncio.gather(*(disconnect(server) for server in self.servers.values()))

    async def call(self, a, name, args, skill=None, *, single_attempt=False, timeout=None,
                   operation_id=None, attempt_no=None, ticket_operation=None, refund_preview_scope=None,
                   context=None, tool_call_id=None, access='agent', required_max_price=None, rewritten_query=None):
        server = self.server_for(name)
        start = time.perf_counter()
        call_id = uid('gateway-')
        fields = dict(server_id=server.server_id, name=name, tool_name=name, tool_call_id=tool_call_id,
                      request_id=call_id, skill=skill, operation_id=operation_id or call_id,
                      attempt_no=attempt_no or 1, layer='transport', args=args,
                      transport='inprocess' if server.local_server is not None else 'http')
        emit('tool_attempt', **fields)
        try:
            result = await self._call(
                a, name, args, skill, single_attempt=single_attempt, timeout=timeout,
                ticket_operation=ticket_operation, refund_preview_scope=refund_preview_scope,
                context=context, operation_id=fields['operation_id'], attempt_no=fields['attempt_no'],
                tool_call_id=tool_call_id, access=access, required_max_price=required_max_price,
                rewritten_query=rewritten_query,
            )
            diagnostics = result.get('diagnostics')
            if result['ok'] and server.server_id == 'retrieval' and not diagnostics:
                raise DomainError('TOOL_BAD_RESULT', '检索结果缺少执行证据')
            if diagnostics and (server.server_id != 'retrieval' or any(
                    diagnostics[key] != expected for key, expected in {
                        'invocation_id': context.invocation_id if context else fields['operation_id'],
                        'operation_id': fields['operation_id'], 'tool_call_id': tool_call_id,
                        'attempt_no': fields['attempt_no'],
                    }.items())):
                raise DomainError('TOOL_BAD_RESULT', '检索结果与请求不匹配')
        except (Exception, asyncio.CancelledError) as error:
            code = error.code if isinstance(error, DomainError) else (
                'TOOL_TIMEOUT' if isinstance(error, TimeoutError) else
                'RUN_CANCELLED' if isinstance(error, asyncio.CancelledError) else 'TOOL_UPSTREAM_ERROR')
            latency = (time.perf_counter()-start)*1000
            emit('tool', **fields, ok=False, status='blocked' if code == 'FORBIDDEN' else 'error',
                 result_status='error', error_code=code, duration_ms=latency, latency_ms=latency)
            if not single_attempt and isinstance(error, Exception) and name in WRITE_TOOLS and not isinstance(error, DomainError):
                raise DomainError('UNKNOWN_OUTCOME', '写入结果未知，请核对原操作') from error
            raise
        latency = (time.perf_counter()-start)*1000
        status = 'success' if result['ok'] else 'error'
        emit('tool', **fields, ok=result['ok'], status=status, result_status=status,
             error_code=result['error']['code'] if result['error'] else None,
             duration_ms=latency, latency_ms=latency, upstream_request_id=result['request_id'],
             result=result if server.server_id == 'business' else {})
        if diagnostics:
            attrs = {key: value for key, value in fields.items() if key != 'args'}
            emit('retrieval.execution', **(attrs | diagnostics), duration_ms=diagnostics['latency_ms'],
                 result_status=status, status=status, error_code=result['error']['code'] if result['error'] else None)
        return result

    async def _call(self, a, name, args, skill=None, *, single_attempt=False, timeout=None,
                    ticket_operation=None, refund_preview_scope=None, context=None,
                    operation_id=None, attempt_no=1, tool_call_id=None, access='agent',
                    required_max_price=None, rewritten_query=None):
        server = self.server_for(name)
        if server.server_id == 'business' and name not in ALLOW.get(skill, set()):
            raise DomainError('FORBIDDEN', '工具不在白名单')
        if server.server_id == 'retrieval' and context is None and access != 'direct_image':
            raise DomainError('FORBIDDEN', '检索请求缺少可信会话范围')
        if not server.enabled:
            raise DomainError('MCP_DISABLED', '工具服务已停用')
        scope = dict(tool=name, invocation_id=context.invocation_id if context else operation_id,
                     operation_id=operation_id, tool_call_id=tool_call_id, attempt_no=attempt_no,
                     conversation_id=context.conversation_id if context else None,
                     access=access, image_id=args.get('image_id'),
                     required_max_price=required_max_price, rewritten_query=rewritten_query)
        principal = context.principal if context is not None else a
        attempt_timeout = min(server.timeout, timeout) if timeout is not None else server.timeout
        token = sign_actor(principal, server.secret, expires=time.time()+min(60, attempt_timeout),
                           audience=server.audience, scope=scope,
                           ticket_operation=ticket_operation, refund_preview_scope=refund_preview_scope)
        async with asyncio.timeout(attempt_timeout):
            if server.local_server is not None:
                async with Client(server.local_server) as client:
                    if not server.schemas:
                        server.schemas = await self._discover(server, client)
                    result = await client.call_tool(name, args, meta={AUTH_METADATA: token}, raise_on_error=False)
            else:
                client = await self._connect(server)
                result = await client.call_tool(name, args, meta={AUTH_METADATA: token}, raise_on_error=False)
            return decode_tool_result(result)

    async def rebuild_knowledge(self, principal):
        from app.retrieval_mcp import REBUILD_SCOPE
        server = self.servers['retrieval']
        if not server.enabled:
            raise DomainError('KNOWLEDGE_NOT_READY', '知识服务已停用')
        url = urlsplit(server.endpoint)
        path = url.path.rstrip('/').rsplit('/', 1)[0] + '/internal/knowledge/rebuild'
        endpoint = urlunsplit((url.scheme, url.netloc, path, '', ''))
        token = sign_actor(principal, server.secret, audience=server.audience, scope={'tool': REBUILD_SCOPE})
        try:
            async with httpx.AsyncClient(timeout=self.config.agent_timeout_seconds, trust_env=False) as client:
                response = await client.post(endpoint, headers={'X-Internal-Auth': 'Bearer '+token})
                if response.status_code == 403:
                    raise DomainError('FORBIDDEN', '仅管理员可以重建知识索引')
                response.raise_for_status()
        except httpx.HTTPError:
            raise DomainError('KNOWLEDGE_NOT_READY', '知识索引重建暂不可用，请核对检索服务状态') from None
        return response.json()


if __name__=='__main__':
    db = DB(settings.sqlite_url)
    db.initialize()
    create_mcp(Business(db), settings.business_secret).run(
        transport='http', host=settings.business_mcp_host, port=settings.business_mcp_port,
        show_banner=False, stateless_http=True,
    )
