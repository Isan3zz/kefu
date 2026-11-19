---
id: kbv3_policy_refund_status
title: 退款状态术语与进度解释
version: '1'
tenant_id: public
status: published
visibility: public
effective_at: '2020-01-01T00:00:00+00:00'
---
# 退款状态术语与进度解释

## 四类容易混淆的状态
WAITING_CONFIRMATION 是退款预览待确认，不代表已经提交。CONFIRMED 表示确认条件已通过、提交操作可能继续进行，不代表资金状态。REQUEST_ACCEPTED 是工作流已完成申请受理。退款申请记录中的 ACCEPTED 只表示受理，不能说“已退款到账”。

## 用户问进度时应做什么
用户说“上一笔退款到账了吗”“O3001 的退款处理到哪了”时，应优先查询本人订单或实际申请记录，不要把进度问题误判断为新申请。没有退款记录时应明确指出未找到已受理申请。付款渠道的实际到账时间和支付回执不在模拟系统内，不能推断完成日期。

## 重复申请与异常结果
如果该订单已有退款申请，资格工具会返回 ALREADY_EXISTS。对于 UNKNOWN_OUTCOME，应查询原操作标识关联的业务结果；不能把未知状态直接当失败或成功。退款进度涉及本人订单，应遵循用户与租户权限隔离。
