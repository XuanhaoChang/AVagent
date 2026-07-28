# 经典问题评测执行指南

## 安全前置

1. 撤销任何曾粘贴到聊天、工单或日志中的 API token。
2. 新 token 只在执行机器的 shell 中设置：`export ARK_API_KEY=...`。
3. 禁止把 `思考过程及标准答案`、`output/pred.csv` 或人工复核结果发送给推理模型。

## 已生成的离线产物

- `output/benchmark/audit/`：100 条媒体与真值格式审计。
- `output/benchmark/human_review_exploratory_20.csv`：不含标准答案的 20 条美感/动作人工复核队列。
- `output/benchmark/route_candidates.csv`：只基于 prompt、反馈和媒体元数据的候选路由。
- `output/benchmark/feishu_import.csv`：一问题一行的飞书导入表。

## 可交给低成本模型的任务

### 任务 A：20 条探索维度复核

只读取 `human_review_exploratory_20.csv` 中的允许字段和对应媒体，填写：

- 美感：好/一般/差/不可判断
- 动作协调性：正常/异常/不可判断
- 动作连续性：正常/异常/不可判断
- 评审证据：时间区间和可见事实
- 评审备注

不得查看 `思考过程及标准答案` 或任何人工复核结论。反馈只是优先核查线索。

### 任务 B：GPT 三配置冒烟

```bash
conda activate avagent
python scripts/run_experiment_matrix.py \
  --model gpt=gpt-5.5-2026-04-24 \
  --limit 2 \
  --execute
```

若 `harness_c` 报输入音频格式不支持，保留失败日志，不修改 A/B 配置。该结果即为当前网关的直接音频兼容性结论。

### 任务 C：Seed-Lite 冒烟

先向网关管理员确认实际模型 ID。确认后运行：

```bash
python scripts/run_experiment_matrix.py \
  --model seed_lite=<网关实际模型ID> \
  --limit 2 \
  --execute
```

不得自行把 Seedance 视频生成模型当作 Seed-Lite 评测模型。

### 任务 D：Auralis 本地专家工具 + Gemini 音视频联合检查

GPT-D 使用独立入口，不修改 GPT A/B/C。音频子智能体实现位于
`agents/auralis/`，可复用工具位于主目录 `tools/`。每条样本都无条件进入
`AuralisAgent.analyze()`；无音轨时返回 `status=no_audio`。GPT-A
baseline 完成参考图、用户反馈、动作、镜头、场景和文字等完整评测；GPT-A
在 `audio_mode=none` 时禁止输出任何音频问题，也不得输出“缺少音频证据”
一类占位条目。Auralis 先运行本地 Faster-Whisper ASR、RapidOCR 字幕提取和
确定性语音字幕对齐，再调用 `gemini-3.5-flash` 复核音频、
语言/台词/声音与主体绑定关系及粗粒度声画冲突，最后把
Gemini 问题追加到本次 GPT-A 结果。GPT-D 不读取历史 GPT-A `pred.csv`，
任一路调用失败时该条均不写成成功结果。

由于该网关返回
`unsupport Part Type: video_url` 和 `unsupport Part Type: input_audio`，
GPT-D 不使用 OpenAI 媒体 Part。请求 URL 和 Bearer 鉴权保持不变，请求体改为
网关实测兼容的 Gemini `model + contents[].role/parts[]` 格式：把当前 MP4
按 2 fps 全量抽帧，并将全部参考图、视频帧 JPEG 编码为 `inline_data`；
同源 WAV 按采样帧连续切成带明确 `time_range` 标签的 1 秒片段，每段作为
独立的 `inline_data` 发送，最后不足 1 秒的尾段保留。所有内容在一次请求中
与 `user_prompt` 联合发送。参考图用于辅助确认角色外观、身份和性别；
Gemini 不接收用户反馈或标准答案。

安装免费本地依赖并缓存模型：

```bash
conda activate avagent
python -m pip install -r requirements-audio-agent.txt
hf download Systran/faster-whisper-large-v3
```

先运行不发送任何媒体到 API 的本地工具冒烟：

```bash
python scripts/smoke_auralis_local.py /path/to/video.mp4
```

默认使用 `cuda/int8_float16`，CUDA失败时立即报错。只有显式设置
`AURALIS_ASR_ALLOW_CPU_FALLBACK=1` 才允许对 CUDA 初始化类错误退化为
`cpu/int8`；模型错误、模型损坏和显存不足仍然失败。成功输出必须包含
带时间戳 ASR、OCR 字幕候选和对齐候选。

先运行两条冒烟：

```bash
conda activate avagent
python call_ffmpeg_skill_gpt_d.py \
  --gpt-a-model gpt-5.5-2026-04-24 \
  --gemini-model gemini-3.5-flash \
  --limit 2 \
  --output-csv output/benchmark/runs/gpt/gpt_d/pred.csv \
  --run-log output/benchmark/runs/gpt/gpt_d/run.jsonl
```

最终 GPT-D 结果为“本次实时 GPT-A 问题数组 + 本次实时 Auralis 问题数组”。Gemini
分支接收参考图，但不接收参考音频，因此可以结合参考图判断角色身份、性别和
明显的男女声冲突，不能判断具体人物的参考声纹一致性；其他视觉检查由保留的
GPT-A 结果承担。
2 fps 抽帧可以辅助绑定可见说话人和检查粗粒度声画冲突，但不能支持精确口型
同步测量。Gemini 只能按音频片段边界定位音频事件，不得生成比分片边界更精细
的音频时间。单个 WAV `inline_data` 兼容格式已实测返回 HTTP 200；多 WAV
分片联合请求仍需用真实样本冒烟确认网关兼容性。

### 任务 E：图片数量容量测试

仓库已由样本 39 循环构造一个不含真值的约 30 秒探针。重新生成：

```bash
python scripts/prepare_30s_probe.py
```

然后运行 8/16/32/48/60/80 张图片实验：

```bash
python scripts/run_capacity_matrix.py \
  --model gpt-5.5-2026-04-24 \
  --sample-index 1 \
  --input-csv output/benchmark/capacity_30s/probe_gt.csv \
  --output-root output/benchmark/capacity_30s/runs \
  --execute
```

该任务验证请求成功率、请求大小和延迟。探针后半段来自循环素材，只适合测接口容量；真实 30 秒内容理解质量仍需在取得原生 30 秒样本后补测。

## 全量实验

两条冒烟全部成功后才运行：

```bash
python scripts/run_experiment_matrix.py \
  --model gpt=gpt-5.5-2026-04-24 \
  --model seed_lite=<网关实际模型ID> \
  --limit 100 \
  --execute
```

每组输出独立写入 `output/benchmark/runs/<model>/<profile>/`，不会覆盖原始 `output/pred.csv`。运行只生成预测与日志，不自动评分。

## 专家工具实验

将每个工具的结构化 JSON 放入独立目录，随后运行：

```bash
python call_ffmpeg_skill.py \
  --profile harness_b \
  --expert-evidence-dir output/benchmark/expert_evidence/paddleocr \
  --output-csv output/benchmark/runs/gpt_ocr/pred.csv \
  --run-log output/benchmark/runs/gpt_ocr/run.jsonl
```

不同实验结果通过人工复核包逐问题比较，不再做旧标签自动映射或 Precision、Recall、F1 评分。

## 飞书验证

先 dry-run：

```bash
python scripts/upload_feishu_bitable.py --limit 2
```

配置飞书环境变量并确认表字段后再执行：

```bash
python scripts/upload_feishu_bitable.py --limit 2 --execute
```

确认两条记录无乱码、JSON 截断或字段错位后，使用 `--limit 0 --execute` 扩展到全量。
