---
name: order-tracking
description: 查询本人订单及物流，结合当前上下文定位订单
metadata:
  version: "1.0.0"
---
先确定订单；多个候选时澄清。仅调用 get_orders、get_order、track_shipment。
只根据实际工具状态回答，不承诺送达时间，不修改订单。
