import re
import unicodedata

SENSITIVE = re.compile(
    r'密码|口令|验证码|动态码|银行卡|信用卡|支付凭证|支付账号|身份证|护照|社保号|病历|病史|宗教信仰|'
    r'收货地址|家庭住址|详细地址|门牌号|手机号|电话号码|电子邮箱|'
    r'(?i:api[ _-]?key|password|passcode|private[ _-]?key|access[ _-]?token|secret[ _-]?key|bearer\s|sk-[\w-]{8,})|'
    r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b|\d(?:[\s-]*\d){10,}|'
    r'[^，。\n]{1,28}(?:路|街|巷|大道)\s*\d+\s*(?:号|栋|室)|\d+\s*(?:栋|单元)\s*\d+',
)
INSTRUCTION = re.compile(
    r'忽略|无视|绕过|无需确认|免确认|不用确认|跳过.{0,8}确认|系统提示|开发者指令|'
    r'直接退款|自动退款|代我确认|调用工具|替我下单|\[/?USER_MEMORY|'
    r'(?i:ignore.{0,24}instructions|system prompt|bypass|submit_refund_request|developer message)',
)
DYNAMIC = re.compile(
    r'订单号|订单状态|订单\s*[oO]\d|\b[Oo]\d{3,}\b|退款|物流状态|已发货|已送达|已到账|'
    r'当前价格|现价|运费|库存|退货政策|退款资格|跟踪号|(?i:tracking number|refund status|stock level)',
)
REPORTED = re.compile(r'(?:系统|客服|助手|机器人|工具|网页|文档|商家).{0,12}(?:说|认为|显示|写|推荐)|(?i:assistant said|system says)')


def normalized(text):
    return unicodedata.normalize('NFKC', text).strip()


def prohibited(text):
    text = normalized(text)
    if SENSITIVE.search(text):
        return 'SENSITIVE_CONTENT'
    if INSTRUCTION.search(text) or any(unicodedata.category(char) == 'Cf' for char in text):
        return 'INSTRUCTION_CONTENT'
    if DYNAMIC.search(text):
        return 'DYNAMIC_BUSINESS_FACT'
    return None
