# Auralis 音视取证子 Agent

Auralis 是项目中的专用音频子智能体。程序处理每条视频时都必须调用
`AuralisAgent.analyze()`；无音轨不是“跳过调用”，而是成功返回
`status="no_audio"`。

如果由主 GPT 自主选择工具，使用以下强制路由规则：

> 只要存在一点点可能与音频、语音、台词、字幕、音色、音效、背景音乐、
> 环境声、说话人绑定、口型或声画同步有关，就必须调用 Auralis。
> 在 Auralis 返回前，主 Agent 不得输出或排除任何音频相关结论。

## 目录职责

- `agent.py`：编排本地工具与最终 Judge。
- `schemas.py`：稳定的输入、证据和输出结构。
- `gemini_backend.py`：Gemini 网关 Judge；不承担 ASR 或 OCR。
- `runner.py`：当前 GPT-A + Auralis CSV 执行入口。
- `/tools/speech_transcription`：Faster-Whisper 本地 ASR。
- `/tools/subtitle_extraction`：RapidOCR 本地字幕提取。
- `/tools/speech_subtitle_alignment`：确定性字幕差异分类。
- `/tools/media`：ffmpeg/ffprobe 基础能力。

本地工具只提供候选证据。Gemini必须回到原始音频和画面复核，不能把
ASR、OCR 或编辑距离结果直接当作问题真值。

## 安装

```bash
conda activate avagent
python -m pip install -r requirements-audio-agent.txt
hf download Systran/faster-whisper-large-v3
```

这些依赖和模型均在本机运行，不调用付费 ASR/OCR API。Linux GPU 运行所需
的 cuBLAS/cuDNN 路径由 Auralis CLI 自动加入子进程环境。默认要求 GPU
成功初始化并在失败时立即报错；只有明确设置
`AURALIS_ASR_ALLOW_CPU_FALLBACK=1` 时才允许针对 CUDA 初始化错误退化为
`cpu/int8`，模型名错误、模型损坏和显存不足不会被静默降级。

## 离线冒烟

```bash
conda activate avagent
python scripts/smoke_auralis_local.py /path/to/video.mp4
```

成功输出必须包含：

- `status` 为 `ok` 或无音轨时的 `no_audio`；
- ASR 的 `backend/model/device`；
- 带起止时间的 ASR segments；
- OCR 字幕 segments；
- 字幕对齐候选问题。

## 完整评测

```bash
conda activate avagent
python call_ffmpeg_skill_gpt_d.py \
  --limit 2 \
  --output-csv output/benchmark/runs/gpt/auralis/pred.csv \
  --run-log output/benchmark/runs/gpt/auralis/run.jsonl
```

该入口先运行 GPT-A，再对每条样本无条件调用 Auralis。`用户反馈`和
`思考过程及标准答案`都不会发送给 Auralis。
