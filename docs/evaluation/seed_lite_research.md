# Seed-Lite 对照实验说明

更新日期：2026-07-23。

## 当前可确认信息

- 火山方舟官方快速开始示例使用 `doubao-seed-2-0-lite-260215`，并以 Responses API 调用：[官方快速开始](https://www.volcengine.com/docs/82379/1795150)。
- 2026 年 6 月相关官方能力说明列出了 `doubao-seed-2-0-lite-260428`，并注明该版本可处理音频类数据：[能力更新](https://www.volcengine.com/docs/6492/2165228?lang=en)。
- 火山官方产品页把 Seed 2.0 Lite 定位为低成本模型，并列出按 token 计费，但价格会变化，实验报告应记录运行当日价格快照：[产品与定价页](https://www.volcengine.com/product/doubao/)。
- 官方开发者文章称新版 Seed 2.0 Lite 支持视频、图像、音频和文本联合理解：[全模态能力说明](https://developer.volcengine.com/articles/7636596381943070763)。

## 必须先验证的兼容性

本仓库当前使用自定义 `Chat Completions` 网关，而官方新模型示例主要使用 `Responses API`。因此不得只根据模型名假设以下能力可用：

1. 网关是否有对应模型路由权限。
2. `image_url`、`input_audio` 和函数调用的消息格式是否兼容。
3. 返回结果是否包含 usage。
4. 多轮 tool call 是否保持图片和音频上下文。

先运行 2 条 smoke test；通过后再跑 100 条。对照实验必须固定相同 prompt、参考图、视频帧、音频证据和工具结果。

## 报告要求

只展示人工复核结论、典型成功/失败案例、请求成功率、平均延迟、token 和当日估算成本，不做旧标签自动映射评分，也不自动决定是否替换 GPT。
