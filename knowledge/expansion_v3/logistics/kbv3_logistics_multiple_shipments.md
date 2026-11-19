---
id: kbv3_logistics_multiple_shipments
title: 多条物流记录和订单明细的解释
version: '1'
tenant_id: public
status: published
visibility: public
effective_at: '2020-01-01T00:00:00+00:00'
---
# 多件订单、多条轨迹和履约记录

## 商品和订单
`order_items` 保存订单里每个产品 ID、数量和下单时单价，`orders.paid_amount_cents` 是订单已支付金额。它们可能因折扣或异常测试数据不一致，客服不能私自重写实付金额。

## 物流轨迹
`shipments` 可能包含同一订单的多条状态记录，不能简单地把第一条当最新。应该比较 `updated_at`，并注意重复状态和缺失运单号的异常。多条轨迹不一定意味着多件商品分别配送；当前数据结构没有独立包裹与明细的严格映射。

## 退款与转人工
当前退款流程按订单全额预览，不能通过拆分明细自行执行部分退款。需要人工调查时，工单应总结订单号、商品和物流疑点，不写入虚构的承运商结论。
