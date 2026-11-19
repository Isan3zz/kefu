"""The single OpenAI-compatible request site for decisions, answering and Judge."""
import asyncio,hashlib,json,time
from app.business import DomainError
from collections import deque
from app.db import uid
from pydantic import ValidationError
from app.harness.contracts import HarnessFailure,ModelResult,ModelUsage
from app.harness.recovery import backoff
from app.harness.tool_protocol import decision_from_response

def estimate_tokens(messages):
    # Conservative UTF-8 byte upper estimate; never advertised as provider token usage.
    return 8+sum(8+len((json.dumps(m, ensure_ascii=False) if 'tool_calls' in m else str(m.get('content',''))).encode('utf-8')) for m in messages)

class ModelGateway:
    def __init__(self,config,client_factory=None,image_adapter=None):
        self.config,self.client_factory=config,client_factory
        self.image_adapter = image_adapter
        self.client=None
    def get_client(self):
        if self.client_factory: return self.client_factory()
        if not all((self.config.llm_api_key,self.config.llm_base_url,self.config.llm_model or self.config.vlm_model)):
            raise HarnessFailure('CONFIG_ERROR','真实聊天需要配置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL',source='model')
        if self.client is None:
            from openai import AsyncOpenAI
            self.client=AsyncOpenAI(api_key=self.config.llm_api_key,base_url=self.config.llm_base_url,max_retries=0)
        return self.client
    async def close(self):
        if self.client: await self.client.close()
    async def complete_with_tools(self, *, ctx, messages, tool_schemas, budget, telemetry,
                                  image_ids=(), image_consent=False, **unused):
        return await self.complete(ctx=ctx, messages=messages, purpose='agent_loop', tool_schemas=tool_schemas,
                                   budget=budget, telemetry=telemetry, image_ids=image_ids, image_consent=image_consent)

    async def complete(self,*,ctx,messages,purpose,response_schema=None,temperature=None,budget,telemetry,model_id=None,tool_schemas=None,image_ids=(),image_consent=False,**unused):
        if image_ids:
            try:
                if self.image_adapter is None:
                    raise DomainError('VLM_NOT_CONFIGURED', 'VLM 图片适配器未配置')
                self.image_adapter.check_capabilities(image_consent, tools=tool_schemas is not None)
            except DomainError as error:
                telemetry.emit('vlm.image.rejected', status='blocked', error_code=error.code, count=len(image_ids))
                raise
            model_id = self.config.vlm_model
        else:
            model_id=model_id or self.config.llm_model
        ctx.remaining(); budget.model_calls+=1
        start=time.perf_counter(); attempt=0; repairs=0; transient=0; timeouts=0; operation=uid('model-op-')
        request=[dict(m) for m in messages]
        if response_schema:
            request[0]['content']+='\nReturn only JSON matching this schema: '+json.dumps(response_schema.model_json_schema())
        telemetry.emit('model.started',purpose=purpose,model_id=model_id,operation_id=operation,content_hash=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest())
        usages=[]
        while True:
            ctx.remaining()
            input_estimate=estimate_tokens(request)+(len(json.dumps(tool_schemas).encode()) if tool_schemas else 0)
            input_estimate += len(image_ids) * self.config.multimodal_image_token_estimate
            estimate=input_estimate+self.config.model_max_output_tokens
            if input_estimate>self.config.context_max_input_tokens:
                raise HarnessFailure('CONTEXT_BUDGET_EXCEEDED','模型 Schema 与必要上下文超出输入预算',source='model')
            if budget.token_estimate_spent+estimate>self.config.model_max_token_budget:
                raise HarnessFailure('CONTEXT_BUDGET_EXCEEDED','本次模型调用预算不足',source='model')
            budget.token_estimate_spent+=estimate; attempt+=1; budget.model_attempts+=1
            try:
                client=self.get_client()
                timeout=min(float(self.config.llm_timeout_seconds),ctx.remaining())
                options={'model':model_id,'messages':request,'temperature':0 if temperature is None else temperature,'max_tokens':self.config.model_max_output_tokens}
                if response_schema and self.config.model_json_mode: options['response_format']={'type':'json_object'}
                if tool_schemas is not None: options.update(tools=tool_schemas, tool_choice='auto')
                prepared = []
                async with asyncio.timeout(timeout):
                    if image_ids:
                        telemetry.emit('vlm.image.prepare_started', count=len(image_ids), model_id=model_id)
                        prepared = await asyncio.to_thread(self.image_adapter.prepare, ctx, image_ids, consent=image_consent)
                        options['messages'] = self.image_adapter.attach(request, prepared)
                        for image in prepared:
                            telemetry.emit('vlm.image.payload_created', image_id=image.image_id, byte_size=len(image.content),
                                           mime_type=image.mime_type, width=image.width, height=image.height,
                                           provider_transport='data_uri', model_id=model_id)
                    response=await client.with_options(max_retries=0,timeout=timeout).chat.completions.create(**options)
                message = response.choices[0].message
                if prepared:
                    for image in prepared:
                        telemetry.emit('vlm.image.sent', image_id=image.image_id, byte_size=len(image.content),
                                       mime_type=image.mime_type, provider_transport='data_uri', model_id=model_id)
                    request_bytes = len(json.dumps(options, ensure_ascii=False, separators=(',', ':')).encode())
                    telemetry.emit('vlm.request.completed', provider_request_bytes=request_bytes, count=len(prepared))
                    # Prevent a provider echo from reaching a graph checkpoint or a tool observation.
                    parts = next(m['content'] for m in options['messages'] if m['role']=='user')
                    encoded = [part['image_url']['url'].split(',', 1)[1] for part in parts if part['type']=='image_url']
                    def redact_echo(text):
                        for value in encoded:
                            text = text.replace('data:image/jpeg;base64,' + value, '[IMAGE_REDACTED]')
                            text = text.replace(value, '[IMAGE_REDACTED]')
                        return text
                    if message.content:
                        message.content = redact_echo(message.content)
                    for call in message.tool_calls or ():
                        if call.type == 'function':
                            call.function.arguments = redact_echo(call.function.arguments)
                content=message.content or ''
                raw=response.usage
                usage=ModelUsage(input_tokens=raw.prompt_tokens,output_tokens=raw.completion_tokens,total_tokens=raw.total_tokens,source='provider') if raw and raw.prompt_tokens is not None and raw.completion_tokens is not None else ModelUsage()
                usages.append(usage)
                if usage.source=='provider':
                    budget.prompt_tokens+=usage.input_tokens; budget.completion_tokens+=usage.output_tokens
                else: budget.provider_usage_complete=False
                parsed=response_schema.model_validate_json(content).model_dump() if response_schema else None
                if not response_schema and tool_schemas is None and not content.strip(): raise ValueError('Empty model result')
                total_usage=ModelUsage(input_tokens=sum(u.input_tokens or 0 for u in usages),output_tokens=sum(u.output_tokens or 0 for u in usages),total_tokens=sum(u.total_tokens if u.total_tokens is not None else (u.input_tokens or 0)+(u.output_tokens or 0) for u in usages),source='provider') if len(usages)==attempt and all(u.source=='provider' for u in usages) else ModelUsage()
                result=ModelResult(text=content,structured=parsed,model_id=model_id,usage=total_usage,attempts=attempt,latency_ms=(time.perf_counter()-start)*1000,finish_reason=response.choices[0].finish_reason)
                decision=decision_from_response(message,model_id,result.finish_reason) if tool_schemas is not None else result
                telemetry.emit('model.completed',purpose=purpose,model_id=result.model_id,operation_id=operation,attempts=attempt,duration_ms=result.latency_ms,usage_source=result.usage.source,input_tokens=result.usage.input_tokens,output_tokens=result.usage.output_tokens)
                from app.observability import emit
                emit('llm',stage=purpose,model=result.model_id,duration_ms=result.latency_ms,attempts=attempt,usage={'prompt_tokens':result.usage.input_tokens,'completion_tokens':result.usage.output_tokens},usage_source=result.usage.source)
                return decision
            except asyncio.CancelledError: raise
            except Exception as exc:
                if len(usages)<attempt: budget.provider_usage_complete=False
                if isinstance(exc,DomainError): code=exc.code
                elif isinstance(exc,(ValidationError,ValueError,IndexError,TypeError)): code='MODEL_BAD_OUTPUT'
                elif isinstance(exc,TimeoutError) or type(exc).__name__=='APITimeoutError': code='MODEL_TIMEOUT'
                else:
                    status=getattr(exc,'status_code',None)
                    code='CONFIG_ERROR' if status in (401,403) else 'MODEL_RATE_LIMIT' if status==429 else 'MODEL_UPSTREAM_ERROR'
                    if status in (400, 415, 422) and image_ids:
                        code = 'VLM_TOOL_CALLING_UNSUPPORTED' if 'tool' in str(exc).lower() else 'VLM_VISION_UNSUPPORTED'
                    elif tool_schemas is not None and status in (400,422) and 'tool' in str(exc).lower():
                        code='MODEL_TOOL_CALLING_UNSUPPORTED'
                if image_ids:
                    telemetry.emit('vlm.image.rejected', error_code=code, status='error', count=len(image_ids))
                if ctx.deadline_monotonic<=time.monotonic(): code='RUN_TIMEOUT'
                telemetry.emit('model.failed',purpose=purpose,model_id=model_id,operation_id=operation,attempts=attempt,attempt_no=attempt,error_code=code,status='error',error_type=type(exc).__name__)
                if code=='MODEL_BAD_OUTPUT' and repairs<1 and attempt<2+self.config.model_max_retries:
                    repairs+=1
                    request.append({'role':'user','content':'The previous output was invalid. Correct it once; return only the required JSON schema. Do not add actions or authorization.'})
                    continue
                if code in ('MODEL_RATE_LIMIT','MODEL_UPSTREAM_ERROR') and transient<self.config.model_max_retries and attempt<2+self.config.model_max_retries:
                    transient+=1; await backoff(ctx,attempt,telemetry,code); continue
                if code=='MODEL_TIMEOUT' and timeouts<min(1,self.config.model_max_retries) and attempt<2+self.config.model_max_retries:
                    timeouts+=1; await backoff(ctx,attempt,telemetry,code); continue
                from app.observability import emit
                emit('llm',stage=purpose,model=model_id,duration_ms=(time.perf_counter()-start)*1000,attempts=attempt,usage={},usage_source='unavailable',error_code=code,status='error')
                if isinstance(exc,DomainError): raise
                explanations = {
                    'MODEL_TOOL_CALLING_UNSUPPORTED': '当前模型或 API 不支持原生工具调用，请检查主模型配置',
                    'VLM_VISION_UNSUPPORTED': '外部 VLM 拒绝图片请求，请检查供应商的 data URI 支持和限制',
                    'VLM_TOOL_CALLING_UNSUPPORTED': '外部 VLM 拒绝图片与工具调用的组合，请检查模型能力',
                }
                raise HarnessFailure(code, explanations.get(code, '模型返回暂不可用，请重试或转人工'),
                                     source='model') from None

class FakeModelGateway:
    """Explicit offline adapter with a deterministic action policy."""
    def __init__(self,config=None): self.calls=deque(maxlen=512); self.config=config
    async def close(self): pass
    async def complete_with_tools(self, *, ctx, messages, tool_schemas, budget, telemetry, fake_message, **unused):
        result = await self.complete(ctx=ctx, messages=messages, purpose='agent_loop', budget=budget, telemetry=telemetry)
        return decision_from_response(fake_message(), result.model_id, 'stop')
    async def complete(self,*,ctx,messages,purpose,response_schema=None,budget,telemetry,task_state=None,**unused):
        ctx.remaining(); budget.model_calls+=1; budget.provider_usage_complete=False
        estimate=estimate_tokens(messages)+(self.config.model_max_output_tokens if self.config else 0)
        if self.config and budget.token_estimate_spent+estimate>self.config.model_max_token_budget:
            raise HarnessFailure('CONTEXT_BUDGET_EXCEEDED','本次离线模型调用预算不足',source='model')
        budget.model_attempts+=1
        budget.token_estimate_spent+=estimate
        self.calls.append({'invocation_id':ctx.invocation_id,'purpose':purpose})
        start=time.perf_counter(); state=task_state or {}; operation=uid('model-op-')
        telemetry.emit('model.started',purpose=purpose,model_id='FakeModelGateway',operation_id=operation)
        value=state.get('structured',{})
        parsed=response_schema.model_validate(value).model_dump() if response_schema else None
        text=json.dumps(parsed,ensure_ascii=False) if parsed is not None else state.get('offline_answer','显式离线演示，不代表真实模型回答。')
        result=ModelResult(text=text,structured=parsed,model_id='FakeModelGateway',attempts=1,latency_ms=(time.perf_counter()-start)*1000)
        telemetry.emit('model.completed',purpose=purpose,model_id=result.model_id,operation_id=operation,attempts=1,duration_ms=result.latency_ms,usage_source='unavailable',input_tokens=None,output_tokens=None)
        return result
