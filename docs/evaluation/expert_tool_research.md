# 专家工具调研与实验接入建议

更新日期：2026-07-23。

## 结论

专家工具应输出“候选证据”，再由统一评测器结合 prompt、参考图和视频复核。不能把 OCR、ASR、姿态或相似度分数直接当作问题真值。

| 维度 | 首选 Pilot | 适用证据 | 主要限制 |
|---|---|---|---|
| 台词、语种、漏读 | Whisper + VAD | 带时间戳转写、语言、静音区间 | 转写不能证明音色、杂音或口型同步；词级时间戳有近似误差 |
| 口型/音画同步 | AV-sync/SyncNet 类模型 | 音画 offset、置信度、有效人脸区间 | 侧脸、遮挡、无清晰口部或非说话声时可能失效 |
| 音色 | 说话人表征 | 参考与生成音频的 embedding 相似度 | 需要同说话人参考；短语音、混响、变调会影响结果 |
| 字幕、Logo、水印 | PaddleOCR | 文本、置信度、像素框、时间 | 小字、艺术字和运动模糊需要原分辨率或局部放大 |
| 人物身份 | InsightFace/ArcFace 类 | 人脸检测框、embedding、跨帧轨迹 | 预训练模型授权需单独核查；侧脸、遮挡和风格化人物不稳定 |
| 重复人物 | 人体/人脸检测 + 跟踪 | 同帧实例数、轨迹 ID | 镜面、海报、屏幕内人物会导致假阳性 |
| 动作协调/连续性 | MMPose/RTMPose + 光流 | 关键点轨迹、速度突变、关节置信度、光流异常 | 关键点异常不等于物理错误；必须由视频语义复核 |
| 画质/美感/闪烁 | 无参考 IQA + 时序统计 | 清晰度、噪声、帧间质量波动 | 美感主观性强；VMAF 等全参考指标不能直接用于无参考生成视频 |
| 空间关系 | 检测 + 跟踪 + 几何规则 | 目标框、中心点、左右/前后关系 | 语义角色绑定仍需 MLLM |

## 候选实现依据

- [OpenAI Whisper](https://github.com/openai/whisper)支持多语种识别、翻译和语言识别，并可输出分段信息；官方实现以 30 秒滑窗处理音频。它适合生成带时间戳的台词证据，但不是音色、音频质量或口型工具。
- Whisper 官方实现提供 `word_timestamps`，但其[官方讨论](https://github.com/openai/whisper/discussions/1855)明确指出词级时间来自推理技巧，停顿等场景并非完全准确，因此只能作候选时间证据。
- [Audio-Visual Synchronisation in the Wild](https://arxiv.org/abs/2112.04432)说明开放域音画同步需要专门测试集和度量；同步工具应单独报告有效人脸区间和置信度。
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)提供场景文字检测、识别和坐标输出，适合字幕、Logo、水印和屏幕文字候选定位。
- [InsightFace](https://github.com/deepinsight/insightface)提供人脸检测、对齐和识别组件；仓库同时提示代码与预训练模型的授权条件不同，正式产品接入前必须核查模型许可。
- [MMPose](https://github.com/open-mmlab/mmpose)覆盖人体、手部、面部和 whole-body 关键点；其 RTMPose 系列适合对稠密帧生成姿态轨迹。
- [RAFT](https://arxiv.org/abs/2003.12039)提供稠密光流估计，可用于找运动突变和局部漂移，但光流本身不能判断动作是否符合 prompt。
- [MUSIQ](https://arxiv.org/abs/2108.05997)是多尺度无参考图像质量方法，可作为画质特征候选；不能替代人类美感判断。
- [VMAF](https://github.com/Netflix/vmaf)是全参考感知视频质量指标，需要参考视频；本数据只有参考图时不应直接使用。若样本未来提供原始视频，它可用于压缩/画质退化对照。

## 统一专家证据格式

每个视频对应一个 `<video_stem>.json`，放到独立 evidence 目录：

```json
{
  "tool": "paddleocr",
  "version": "record-the-exact-version",
  "applicable": true,
  "observations": [
    {
      "start_sec": 1.2,
      "end_sec": 1.8,
      "bbox": [0.1, 0.8, 0.9, 0.96],
      "value": "识别文字",
      "confidence": 0.91
    }
  ],
  "limitations": []
}
```

`call_ffmpeg_skill.py --expert-evidence-dir <目录>` 会将其标记为待验证证据。证据文件如包含标准答案字段会被拒绝。

## 实验顺序

1. GPT-only：不传专家证据。
2. Expert-only：将工具观测转换为同一预测 JSON，但不得读取真值。
3. GPT+Expert：传入完全相同的视频帧和专家证据。
4. 将三组结果导出到人工复核包，逐问题检查证据、时间和定位是否成立。
5. 报告人工复核结论、失败案例、延迟和 token；不做旧标签自动映射评分。
