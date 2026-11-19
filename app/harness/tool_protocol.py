import json
import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage, ToolMessage

MAX_TOOL_BATCH = 24

@dataclass(frozen=True)
class ModelDecision:
    message: AIMessage
    model_id: str
    finish_reason: str | None = None


def decision_from_response(message, model_id, finish_reason):
    calls, invalid, sequence, seen = [], [], [], set()
    if finish_reason in ('length', 'content_filter'):
        raise ValueError('Incomplete tool decision')
    if len(message.tool_calls or ()) > MAX_TOOL_BATCH:
        raise ValueError('Tool batch exceeds protocol capacity')
    for call in message.tool_calls or ():
        if call.type != 'function' or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', call.id or '') or call.id in seen:
            raise ValueError('Invalid or duplicate tool call ID')
        seen.add(call.id)
        name, raw = call.function.name, call.function.arguments
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
            raise ValueError('Invalid tool name')
        sequence.append(call.id)
        try:
            if len(raw) > 8000:
                raise ValueError('Tool arguments too large')
            args = json.loads(raw)
            if not isinstance(args, dict):
                raise ValueError('Expected tool arguments object')
            json.dumps(args, allow_nan=False)
            calls.append({'id': call.id, 'name': name, 'args': args})
        except (ValueError, TypeError):
            invalid.append({'id': call.id, 'name': name, 'args': '{}', 'error': 'INVALID_TOOL_ARGUMENTS'})
    if finish_reason == 'tool_calls' and not sequence:
        raise ValueError('Provider omitted tool call objects')
    from app.db import uid
    return ModelDecision(AIMessage(id=uid('ai-'), content=message.content or '', tool_calls=calls, invalid_tool_calls=invalid,
                                   additional_kwargs={'call_order': sequence}), model_id, finish_reason)


def ordered_calls(message):
    indexed = {call['id']: call for call in [*message.tool_calls, *message.invalid_tool_calls]}
    return [indexed[ident] for ident in message.additional_kwargs.get('call_order', list(indexed))]


def protocol_messages(messages):
    result = []
    for message in messages:
        if isinstance(message, AIMessage):
            calls = [{'id': call['id'], 'type': 'function', 'function': {'name': call['name'],
                      'arguments': json.dumps(call['args'], ensure_ascii=False) if isinstance(call['args'], dict) else '{}'}}
                     for call in ordered_calls(message)]
            result.append({'role': 'assistant', 'content': None, 'tool_calls': calls} if calls else
                          {'role': 'assistant', 'content': message.content})
        elif isinstance(message, ToolMessage):
            result.append({'role': 'tool', 'tool_call_id': message.tool_call_id, 'content': message.content})
    return result
