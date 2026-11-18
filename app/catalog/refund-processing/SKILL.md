---
name: refund-processing
description: 查询退款资格和申请全单退款
metadata:
  version: "1.0.0"
---
明确订单，调用 check_refund_eligibility。按后端政策决定资格。
展示订单、金额及受控确认按钮；禁止将“嗯”“好的”或产品咨询视为确认。
不得直接提交退款；只能通过确认工作流执行。ACCEPTED 只表示退款申请受理。
