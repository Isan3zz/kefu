import re

from app.memory.contracts import TriggerDecision, TriggerDispatch
from app.memory.validator import REPORTED, normalized, prohibited

FORGET = re.compile(r'(?:忘记|删除|清空|抹除|别再保存|不要记住|关闭).{0,18}(?:记忆|偏好|你记|长期|保存|记住的)|(?:忘记|别记|不要记住)(?:我刚才说的|我的信息|这些|它)|(?i:forget (?:my|what)|delete (?:my )?memor)')
CHANGE = re.compile(r'不再|改成|改为|变成|现在.{0,16}(?:可以|接受|预算|更|改|不要)|(?:以后|从今以后).{0,10}(?:换|不选)|(?i:no longer|changed to|now prefer)')
URGENT = re.compile(r'记住|帮我记下|请记下|记一下我的|保存.{0,8}(?:偏好|设置)|(?i:remember (?:that|my))')
STABLE = re.compile(r'偏好|喜欢|讨厌|倾向|钟意|不爱|青睐|首选|优先|更(?:在意|看重|重视|想要)|预算|不能超过|'
                    r'平时|通常|一般|一贯|长期|习惯|惯用|经常|日常|常用|从来|只(?:用|买|选|考虑)|'
                    r'不(?:想要|太能接受|考虑|接受)|主要(?:想要|想买|用|看)|我会(?:选|买|用)|(?i:prefer|usually|budget|always|often)')
SAVE_COMMAND = re.compile(r'^\s*(?:请|麻烦)?(?:记住|帮我记下|请记下|记一下我的|保存我的)')
FURTHER_SERVICE_REQUEST = re.compile(r'[,，。；;\n].{0,12}(?:查|推荐|办理|申请|退货|转人工|告诉我)')
TRANSIENT = re.compile(r'今天临时|(?:只|仅)(?:这次|今天|本次)')


def management_request(text):
    return FORGET.search(normalized(text)) is not None


def save_request_only(text):
    return SAVE_COMMAND.search(text) is not None and FURTHER_SERVICE_REQUEST.search(text) is None


def recommendation_request(text):
    return any(word in text for word in ('推荐', '选购', '怎么选', '哪款适合', '适合我的'))


class MemoryTrigger:
    def check(self, source, *, opted_in, max_bytes):
        if not source or source['role'] != 'user':
            return TriggerDispatch(TriggerDecision.REJECT, 'SOURCE_NOT_OWNED_USER')
        if not opted_in:
            return TriggerDispatch(TriggerDecision.REJECT, 'OPT_IN_REQUIRED')
        content = normalized(source['content'] or '')
        reason = prohibited(content)
        if reason in ('SENSITIVE_CONTENT', 'INSTRUCTION_CONTENT'):
            return TriggerDispatch(TriggerDecision.REJECT, reason)
        if FORGET.search(content):
            return TriggerDispatch(TriggerDecision.REJECT, 'MANAGEMENT_REQUIRED')
        if reason or not content or REPORTED.search(content):
            return TriggerDispatch(TriggerDecision.NOOP, reason or 'NOT_USER_ASSERTED')
        if len(content.encode('utf-8')) > max_bytes:
            return TriggerDispatch(TriggerDecision.REJECT, 'SOURCE_TOO_LONG')
        if URGENT.search(content) or CHANGE.search(content):
            return TriggerDispatch(TriggerDecision.URGENT, 'EXPLICIT_SAVE_OR_CHANGE')
        if TRANSIENT.search(content) or (('?' in content or content.endswith('吗')) and '我' not in content):
            return TriggerDispatch(TriggerDecision.NOOP, 'NOT_STABLE_ASSERTION')
        if STABLE.search(content):
            return TriggerDispatch(TriggerDecision.CANDIDATE, 'POTENTIALLY_STABLE')
        return TriggerDispatch(TriggerDecision.NOOP, 'NO_STABLE_SIGNAL')
