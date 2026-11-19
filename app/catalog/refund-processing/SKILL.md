---
name: refund-processing
description: 查询退款资格和申请全单退款
metadata:
  version: "1.1.0"
---
- 政策问题用 `search_knowledge`，资格和金额问题用 `check_refund_eligibility`，进度问题用 `get_order`。对比多单时，把每条资格结论连同对应订单号呈现。
- 用户明确要求为已知订单申请退款时，先完成同一条消息要求的查询，再单独调用 `begin_refund_workflow`。意图或订单不明时先澄清。用户只想查询或要求先等等时，继续提供查询结果。
- 预览展示订单、金额及受控确认按钮。实际提交通过确认工作流执行。`ACCEPTED` 表示退款申请受理。
