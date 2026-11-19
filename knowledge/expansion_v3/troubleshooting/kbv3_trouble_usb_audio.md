---
id: kbv3_trouble_usb_audio
title: Type-C 耳机无声及 USB-A 蓝牙适配器兼容性排查
version: '1'
tenant_id: public
status: published
visibility: public
effective_at: '2020-01-01T00:00:00+00:00'
---
# USB-C 数字音频与 USB-A 蓝牙音频适配器排查

## Type-C 有线耳机
YT-USBC 使用 USB-C 数字音频，不支持蓝牙。若插上后无声，应检查手机或电脑 USB-C 端口是否支持数字音频输出、系统默认音频设备、接口洁净程度及设备兼容性。USB-C 接口外形相同不代表音频协议一定兼容。

## 蓝牙适配器
YT-BT53 使用 USB-A 接口，依赖主机操作系统和驱动。连接失败时先确认系统是否识别适配器、是否具有相应驱动、是否选择了正确的蓝牙输出设备。不能保证所有电脑插入后立即工作。

## 转人工
无法排除的问题应收集设备型号、系统版本和具体错误提示。不得建议下载来源不明的驱动程序或关闭设备安全功能。
