---
id: kbv3_logistics_shipment_states
title: 物流状态与订单状态的对应关系
version: '1'
tenant_id: public
status: published
visibility: public
effective_at: '2020-01-01T00:00:00+00:00'
---
# 物流状态与订单状态的对应关系

## 主要业务状态
PAID 表示订单已支付，但不一定已发货；SHIPPED 表示订单已发货；DELIVERED 表示订单已送达；CANCELLED 表示已取消。物流轨迹可能显示“待发货”“运输中”“已签收”等文字，但订单和物流记录不是同一张表。

## 需要防范的数据不一致
若 PAID 订单存在“运输中”轨迹，不能只根据 PAID 宣称仍未发货，退款业务需要人工复核。如果 DELIVERED 订单缺少 `delivered_at`，退款资格应为 MISSING_DELIVERY_TIME，不能用物流展示文本推断准确的签收时间。历史订单还可能存在多条物流记录或缺少物流记录。

## 客服回答优先级
当前订单状态用 `get_order` 获取；具体物流用 `track_shipment` 获取。知识库仅解释状态含义，不承诺订单实际位置和送达时间。
