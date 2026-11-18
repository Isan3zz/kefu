"""Compose user-facing business replies without trusting model-written facts."""
import re
from dataclasses import dataclass


FRAMING = (
    '为您整理如下', '为您汇总如下', '查询结果如下', '以下是查询结果', '给您汇总一下', '先看', '再看', '接着看',
    '首先', '其次', '另外', '同时', '也可以看一下', '如果还需要帮助，请告诉我',
)
FACT_MARKER = re.compile(r'\{\{([BK][1-9][0-9]*)\}\}')
NON_FACT = re.compile('|'.join(map(re.escape, FRAMING)) + r'|[\s，、。；：！？,:;.!?]+')
REPLY_INSTRUCTION = (
    '\n如果 current_task.reply_facts 存在，且不需要再调用工具，请用普通中文连接已核实事实：'
    '以 {{B1}} 这样的标签代表整个业务事实，所有 B 标签恰好出现一次；有 K 标签时至少选一个相关知识证据。'
    '只写现有标签与下列衔接语或标点，不要重写事实值、增加结论、使用 Markdown 或输出 JSON。'
    '可用衔接语：' + ' / '.join(FRAMING) + '。示例：为您整理如下：{{B1}}；另外，{{B2}}。'
)


@dataclass(frozen=True)
class ReplyFact:
    label: str
    text: str


def render_checked(draft: str, facts: dict[str, ReplyFact]) -> tuple[str | None, str]:
    if len(draft.encode('utf-8')) > 1800:
        return None, 'DRAFT_TOO_LONG'
    matches = list(FACT_MARKER.finditer(draft))
    keys = [match[1] for match in matches]
    if not keys:
        return None, 'NO_FACT_REFERENCES'
    if len(keys) != len(set(keys)) or not set(keys) <= facts.keys():
        return None, 'INVALID_FACT_REFERENCES'
    required = {key for key in facts if key.startswith('B')}
    knowledge = {key for key in facts if key.startswith('K')}
    if not required <= set(keys) or (knowledge and not knowledge.intersection(keys)):
        return None, 'MISSING_FACT_REFERENCES'
    offset = 0
    for match in matches:
        if NON_FACT.sub('', draft[offset:match.start()]):
            return None, 'UNVERIFIED_TEXT'
        offset = match.end()
    if NON_FACT.sub('', draft[offset:]):
        return None, 'UNVERIFIED_TEXT'
    rendered = FACT_MARKER.sub(lambda match: facts[match[1]].text.rstrip('。'), draft).strip()
    return rendered if rendered.endswith(('。', '！', '？')) else rendered + '。', 'VALIDATED'
