"""Observable-signal routing; population frequencies are never sample evidence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    experts: tuple[str, ...]
    dense_sampling: bool
    local_crop: bool
    model_tier_candidate: str
    reasons: tuple[str, ...]


def route_observations(
    prompt: str,
    feedback: str,
    has_audio: bool,
    reference_count: int = 0,
    duration_sec: float = 0.0,
) -> RouteDecision:
    text = f"{prompt or ''}\n{feedback or ''}".lower()
    experts: list[str] = []
    reasons: list[str] = []

    speech = any(word in text for word in ("台词", "说：", "说话", "念", "语种", "发音", "口型"))
    if has_audio and speech:
        experts.append("asr")
        reasons.append("文本或反馈包含语音/台词可观测信号")
    if has_audio and any(word in text for word in ("口型", "对嘴", "唇", "音画同步")):
        experts.append("av_sync")
        reasons.append("文本或反馈包含口型或音画同步信号")
    if any(word in text for word in ("字幕", "文字", "logo", "水印", "屏幕")):
        experts.append("ocr")
        reasons.append("文本或反馈包含画面文字信号")
    if any(word in text for word in ("身份", "换脸", "变脸", "撞脸", "人不对", "参考图")):
        experts.append("identity")
        reasons.append("文本或反馈包含身份或参考一致性信号")

    dense = any(
        word in text
        for word in ("快速", "一闪", "连续", "跳切", "穿模", "动作", "变形", "闪烁", "掉帧")
    )
    local_crop = any(
        word in text
        for word in ("手", "脸", "口型", "字幕", "文字", "logo", "水印", "小物体")
    )
    complex_case = (
        (has_audio and speech)
        or dense
        or reference_count > 2
        or duration_sec > 15
        or len(prompt or "") > 500
    )
    tier = "gpt_candidate" if complex_case else "seed_lite_candidate"
    return RouteDecision(tuple(experts), dense, local_crop, tier, tuple(reasons))
