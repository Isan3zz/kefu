import asyncio,random,time
from app.harness.contracts import HarnessFailure

def error_class(code):
    if code in ('MODEL_TIMEOUT','MODEL_RATE_LIMIT','MODEL_UPSTREAM_ERROR'): return 'TRANSIENT_MODEL_FAILURE'
    if code in ('TOOL_TIMEOUT','TOOL_UPSTREAM_ERROR','UPSTREAM_UNAVAILABLE'): return 'TRANSIENT_READ_FAILURE'
    if code in ('FORBIDDEN','NOT_FOUND','POLICY_DENIED'): return 'PERMISSION_DENIED'
    if code in ('BUSINESS_RULE_FAILED','ALREADY_EXISTS'): return 'BUSINESS_RULE_DENIED'
    if code=='UNKNOWN_OUTCOME': return 'WRITTEN_BUT_RESPONSE_UNKNOWN'
    if code=='CONTEXT_BUDGET_EXCEEDED': return 'CONTEXT_EXHAUSTED'
    if code in ('MODEL_BAD_OUTPUT','TOOL_BAD_RESULT','INVALID_ARGUMENT'): return 'PERMANENT_VALIDATION_ERROR'
    return 'RUNTIME_FAILURE'

async def backoff(ctx,attempt,telemetry,code,sleep=asyncio.sleep):
    duration=.05*2**(attempt-1)+random.uniform(0,.02)
    if ctx.remaining()<=duration: raise HarnessFailure('RUN_TIMEOUT','剩余执行预算不足以重试')
    telemetry.emit('recovery.started',error_class=error_class(code),action='retry',attempt_no=attempt)
    await sleep(duration)
    ctx.remaining()
    telemetry.emit('recovery.completed',error_class=error_class(code),action='retry',result='ready')
