# Auralis 音频与字幕取证子 Agent

Auralis 是项目中的专用音频与字幕子智能体。程序处理每条视频时都必须调用
`AuralisAgent.analyze()`；无音轨不是“跳过调用”，而是成功返回
`status="no_audio"`。

如果由主 GPT 自主选择工具，使用以下强制路由规则：

> 只要存在一点点可能与音频、语音、台词、字幕、音色、音效、背景音乐、
> 环境声或说话人绑定有关，就必须调用 Auralis。
> 在 Auralis 返回前，主 Agent 不得输出或排除任何音频相关结论。

精确音画同步和口型延迟不属于 Auralis。此类判断交给
`third_party/AVBench/evaluation/evaluate_syncnet.py`；Auralis 的视频帧只用于
辅助确认字幕和可见说话人，不用于同步测量。

## 目录职责

- `agent.py`：编排本地工具与最终 Judge。
- `constrained_asr.py`：从自由格式 prompt 锚定参考文本、生成 ASR 候选并解释
  SenseVoice CTC 成对分数；不依赖引号、角色冒号或固定字段。
- `schemas.py`：稳定的输入、证据和输出结构。
- `gemini_backend.py`：Gemini 网关 Judge；不承担 ASR、OCR 或音画同步。
- `runner.py`：当前 GPT-A + Auralis + AVBench + 最终 GPT 汇总的 CSV 执行入口。
- `../avbench_sync.py`：固定调用 AVBench SyncNet 的单视频适配器；它在 `.conda-envs/avbench` 子进程中复用模型，不把 PyTorch 依赖带入 ASR/OCR 环境。
- `/tools/speech_transcription`：SenseVoice-Small + CAM++ 本地 ASR/说话人分离。
- `/tools/subtitle_extraction`：RapidOCR 本地字幕提取。
- `/tools/speech_subtitle_alignment`：确定性字幕差异分类。
- `/tools/media`：ffmpeg/ffprobe 基础能力。

台词参考文本不是用正则“解析字段”：系统把完整 ASR 片段与 prompt 全文做
格式无关的字符级半全局对齐，并要求每个结果都能回溯到 prompt 的原始字符区间。
无可靠锚点时返回 `no_reference_dialogue`，不会猜测标准台词。存在差异时，复用同一个
SenseVoice worker 对局部 WAV 中的“ASR 实际候选”和“prompt 预期候选”计算 CTC
似然；`observed_preferred` 会直接形成确定性 Auralis 音频问题，Gemini 不需要回听
完整 WAV。`expected_preferred` 用于拦截自由 ASR 假阳性；拼音（含声调）一致的候选
标记为 `orthographic_homophone`，避免把“棠棠/糖糖”这类字符模型偏好误报为读音错误；
`ambiguous` 不会被升级成确定错误。OCR 和说话人绑定仍由各自证据分支处理。
Gemini 若仍输出与这些本地否决分支直接冲突的读音问题，Agent 会按候选文本与时间重叠
做窄范围过滤，并把被过滤对象保存在 `diagnostics`，不会影响同时间段的角色绑定问题。

## 安装

```bash
conda activate avagent
python -m pip install -r requirements-audio-agent.txt

# SenseVoice/CAM++ worker uses the isolated avbench environment.
/data/changxuanhao/AVagent/.conda-envs/avbench/bin/python -m pip install -r requirements-sensevoice.txt
```

这些依赖和模型均在本机运行，不调用付费 ASR/OCR API。首次运行会通过
ModelScope 下载 SenseVoiceSmall、FSMN-VAD、CT-Punc 和 CAM++ 权重。
SenseVoice worker 保持为长驻子进程，避免每条视频重复加载模型；默认使用
`merge_vad=False`，保留说话轮次供 CAM++ 分离；短片段使用去除 FunASR 小样本
单说话人捷径的频谱聚类。CAM++ 没有参考声纹时只输出匿名 `spk` 标签；没有参考声纹时
不能把标签绝对命名为具体角色，但仍必须用相同/不同 `spk` 与 prompt 台词轮次做相对
绑定一致性检查，例如同一 `spk` 跨越不同角色台词或同一角色轮次被拆给多个 `spk`。

SenseVoice 模型如需覆盖，使用专用变量 `AURALIS_SENSEVOICE_MODEL`。旧版
Faster-Whisper 使用的 `AURALIS_ASR_MODEL` 不会再覆盖 SenseVoice，避免把 Whisper
模型路径误传给 FunASR worker。

## 离线冒烟

```bash
conda activate avagent
python scripts/smoke_auralis_local.py /path/to/video.mp4
```

成功输出必须包含：

- `status` 为 `ok` 或无音轨时的 `no_audio`；
- ASR 的 `backend/model/device`，其中 backend 应为 `funasr-sensevoice-campp`；
- 带起止时间的 ASR segments；
- 每段的 `speaker` 标签和 `speaker_diarization` 元数据；
- `constrained_asr` 的原始 prompt 字符锚点、候选分数和判定分支；若 prompt 没有
  可验证参考文本，则状态为 `no_reference_dialogue`；
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

该入口先运行 GPT-A，再对每条样本无条件调用 Auralis 和 AVBench，最后用一次纯文本 GPT
对三路候选问题去重和整理。`用户反馈`和`思考过程及标准答案`都不会发送给
Auralis、AVBench 或最终汇总器。
