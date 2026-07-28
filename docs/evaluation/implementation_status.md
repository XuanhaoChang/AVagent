# 经典问题评测与 Harness 优化实施状态

更新日期：2026-07-28。

## 已完成

- Taxonomy：保留 13 个 Jing 类别和 3 个探索维度作为问题研究目录，不再自动映射旧标签。
- Pilot 审计：100/100 视频、258/258 参考图可解析；100 个视频已完成 ffprobe，其中 94 个含音轨。
- 真值审计：88 条符合 fenced JSON 规则；12 条进入 `gold_review_queue.csv`，未从 thinking 文本补造答案。
- 音频探针：已选择 3 条音频问题样本和 2 条对照样本，且导出队列不含标准答案。
- Harness：
  - Baseline A：2 fps、无音频、无局部裁剪；`audio_mode=none` 时提示词禁止
    音频问题或“缺少音频证据”占位条目，返回结果还会经过确定性过滤；没有
    可证实视觉问题时允许输出 `[]`。
  - Harness B：1 fps 粗筛、候选区间稠密采样、局部裁剪放大。
  - Harness C：B + 直接 WAV；无音轨样本自动退化为无音频证据。
  - GPT-D/Auralis：根目录 `call_ffmpeg_skill_gpt_d.py` 现为兼容入口；
    音频专家实现已迁入主目录 `agents/auralis/`，本地专家工具位于
    `tools/`。每条样本都进入 `AuralisAgent.analyze()`；无音轨返回
    `status=no_audio`，GPT-A 失败时也仍会尝试 Auralis。
  - Auralis 本地工具：Faster-Whisper `large-v3` 负责带时间戳 ASR，
    RapidOCR PP-OCRv6 负责视频帧字幕候选，确定性对齐工具负责错字、漏字、
    多字、语言不一致和时间不一致分类。ASR/OCR/对齐结果均作为候选证据，
    不能直接当作问题真值。
  - Auralis 最终 Judge：继续通过现有 Ark Chat Completions 网关调用
    `gemini-3.5-flash`。URL 和 Bearer 鉴权不变，
    Gemini 请求体采用网关实测兼容的
    `model + contents[].role/parts[]` 格式，将全部参考图、2 fps 全量 JPEG
    抽帧编码为 `inline_data`，并将同源 WAV 连续切为带 `time_range` 标签的
    1 秒 `inline_data` 片段，尾段不足 1 秒时保留；参考图用于辅助确认角色
    身份和性别，Gemini 重点检查语言、台词、声音与主体的绑定错误；字幕与
    实际语音不一致归为文字质量问题，纯视觉字幕存在问题交给 GPT-A；不向
    Gemini 发送用户反馈或标准答案；不修改 A/B/C。
    `sample_072` 已完成旧版真实双模型冒烟，日志记录 GPT-A 与 Gemini 各 1 次
    API 调用且总调用数为 2。实时双模型版本尚未进行 100 条全量重跑；
    当前正式 GPT-D 预测仍为此前的“历史 GPT-A 结果 + 实时 Gemini 音频结果”。
  - 本地工具真实冒烟：在 4.04 秒含音轨样本上，Faster-Whisper
    `large-v3` 已以 `cuda/int8_float16` 成功输出英文分段和词级时间戳，
    无 CPU 回退；RapidOCR 已以本地 PP-OCRv6 模型完成 4 帧推理。完整本地
    `AuralisAgent` 输出 `status=ok`、ASR/OCR/对齐证据。新版包含真实媒体的
    Gemini 端到端请求尚未执行，因为本轮没有取得将该样本音频和帧发送到
    内部网关的明确授权。
- 运行记录：逐样本记录成功状态、耗时、请求体字节数及网关返回的 usage。
- 30 秒探针：已生成 29.997 秒、含音轨、不含真值的合成容量探针；提供 8–80 张图片实验命令。
- 人工复核：已生成 20 条美感、动作协调性、动作连续性队列，未包含标准答案；
  文本复核读取 `input.json`、GT 与预测，但只向 API 发送 prompt、脱敏的材料
  存在性审计、GT 和预测，不发送用户反馈或资源路径；prompt 依赖但 input
  未提供的参考视频或参考音频会在调用 API 前对所有预测来源强制归为类别 5。
- 路由：已生成基于可观测信号的候选表；相对“4 类工具全调用”的静态参考，候选专家调用数由 400 降至 154。该 61.5% 仅是调用数估算，尚未经过准确率实验。
- 飞书：生成一问题一行的 UTF-8 CSV；实现两条 dry-run、批量写入、有限重试和本地 checkpoint PoC。

## 当前评价边界

- 已删除旧标签自动映射以及基于样本类别集合的 Precision、Recall、F1 评分链路。
- 现有预测结果只供逐问题人工复核，不根据旧标签映射判断结果对错。
- Taxonomy 只描述问题类型、所需证据和候选工具，不承担自动评分功能。

## 尚待外部执行

以下项目需要轮换后的有效 API key、网关模型权限或飞书凭据，当前环境未执行：

1. GPT A/B/C 真实样本冒烟，以及 GPT-D 实时双模型版本的 100 条全量实验。
2. 8/16/32/48/60/80 张图片真实请求测试。
3. Seed-Lite 准确模型 ID 与 GPT 对照。
4. 身份、姿态、声纹和精确音画同步等后续专家模型安装和运行。
5. 飞书两条真实写入。
6. 人工填写 20 条探索维度及 12 条真值格式复核。

执行命令统一见 [执行指南](execution_guide.md)。
