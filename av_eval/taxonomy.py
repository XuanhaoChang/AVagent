"""Canonical issue taxonomy definitions."""

from __future__ import annotations

from dataclasses import dataclass
@dataclass(frozen=True)
class Category:
    key: str
    name: str
    jing_count: int
    evidence: str
    candidate_tools: tuple[str, ...]


JING_TAXONOMY = (
    Category("character_identity_error", "人物身份错误", 230, "参考人物与生成帧中的身份对应关系", ("人脸识别", "视觉表征", "跨帧跟踪")),
    Category("reference_consistency_error", "参考一致性错误", 208, "提示词绑定的参考图与生成画面的可比属性", ("视觉表征", "目标跟踪")),
    Category("dialogue_speech_error", "台词/口型错误", 158, "真实音轨、带时间戳转写及可见口部运动", ("ASR", "VAD", "说话人分离", "音画同步")),
    Category("duplicate_character", "重复人物", 105, "同一时刻的人体或人脸实例及跨帧轨迹", ("人体检测", "人脸检测", "跨帧跟踪")),
    Category("instruction_noncompliance", "指令不遵循", 101, "提示词硬约束与生成音视频中的直接事实", ("MLLM",)),
    Category("motion_physics_error", "动作与物理错误", 97, "相邻稠密帧中的姿态、交互和物体轨迹", ("姿态估计", "光流", "视频模型")),
    Category("voice_tone_error", "音色错误", 88, "参考音频与生成语音中同一说话人的声学表征", ("声纹表征", "说话人分离")),
    Category("subtitle_overlay_error", "字幕/水印/文字错误", 87, "原分辨率帧中的文字区域及识别结果", ("OCR", "文本区域检测")),
    Category("spatial_layout_error", "空间布局错误", 65, "目标位置、朝向、左右关系和跨帧轨迹", ("目标检测", "姿态估计", "跟踪")),
    Category("visual_quality_artifact", "画质与视觉伪影", 49, "原分辨率帧和相邻帧中的清晰度、闪烁与结构变化", ("IQA", "视频美学", "时序稳定性")),
    Category("audio_artifact", "音频伪影", 19, "真实音轨的波形、频谱或可听证据", ("音频质量检测", "VAD")),
    Category("system_task_failure", "系统任务失败", 7, "任务状态或系统日志，不由生成视频内容判断", ("系统日志",)),
    Category("junk_test", "测试/垃圾数据", 51, "数据内容和标注规则，不由生成视频内容判断", ("数据清洗",)),
)

EXPLORATORY_DIMENSIONS = ("美感", "动作协调性", "动作连续性")
