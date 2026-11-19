---
id: kbv3_policy_refund_eligibility
title: 退款资格与七天边界规则
version: '1'
tenant_id: public
status: published
visibility: public
effective_at: '2020-01-01T00:00:00+00:00'
---
# 退款资格与七天边界规则（模拟 P-REFUND-v1）

## 已支付但未发货
订单状态 PAID，且没有异常运输记录时，系统可以进入退款申请预览。未创建成功退款申请之前，不得称为退款成功。若 PAID 订单的实际物流轨迹已经显示运输中或签收，须转人工复核，不能只看订单状态。

## 已发货和已签收
订单状态 SHIPPED：自动退款资格为 HUMAN_REVIEW，需人工审核，不允许直接走自动提交。订单状态 DELIVERED：以业务系统的实际 `delivered_at` 为准，从签收时间起七乘二十四小时以内含边界值可以申请；超出一秒即超期。缺少签收时间时为 MISSING_DELIVERY_TIME，需要人工核查，不能自行推断签收日期。

## 已取消和已有退款
订单状态 CANCELLED 不符合直接申请资格，系统返回 INVALID_STATUS。只要该订单已存在退款申请记录，则应返回 ALREADY_EXISTS，不得创建第二笔。实付金额小于等于零时为 INVALID_AMOUNT。判定优先级为实付金额、已有退款、运输和签收状态；最终结果以 `check_refund_eligibility` 的可信业务返回为准。

## 用户应如何咨询
用户只问“能否退款”时只检查资格；明确要求“申请退款”时可以展示订单、金额和确认预览。现有知识库内容仅解释规则，不能代替用户确认，不会执行任何写入。
