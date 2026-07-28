"""Gemini gateway judge used by the Auralis specialist agent.

ASR, OCR and deterministic alignment run locally. Gemini receives their outputs
as candidate evidence together with the original audio/video evidence and must
verify every reported problem.
"""

from __future__ import annotations

import base64
import io
import json
import re
import tempfile
import time
import urllib.error
import urllib.request
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from agents.auralis.schemas import AuralisEvidence, AuralisInput
from tools.media.ffmpeg import (
    extract_audio_wav,
    extract_video_frames,
    image_data_url,
    prepare_image_jpeg,
    resolve_local_media,
)


DEFAULT_MODEL = "gemini-3.5-flash"
AUDIO_SEGMENT_SECONDS = 1.0
VIDEO_FRAME_FPS = 2.0
VIDEO_FRAME_WIDTH = 384
OUTPUT_KEYS = (
    "可定位性",
    "置信度",
    "问题说明",
    "问题类型",
    "时间区间",
    "关键帧秒",
    "BBox",
)
TIME_RANGE_PATTERN = re.compile(
    r"^\s*(?:(?P<start_minutes>\d+):)?"
    r"(?P<start_seconds>\d+(?:\.\d+)?)s\s*-\s*"
    r"(?:(?P<end_minutes>\d+):)?"
    r"(?P<end_seconds>\d+(?:\.\d+)?)s\s*$"
)


SYSTEM_MESSAGE = """你是 Auralis 音视取证专家。

你将收到用户 prompt、参考图、带时间戳的视频帧、连续 WAV 音频片段，以及
本地 ASR、OCR 和语音字幕对齐工具输出。请联合复核，只报告明确、可验证的
音频错误、音频参与才能判断的字幕错误或声画冲突。

安全与证据约束：
1. 本地工具结果只是候选证据，不是问题真值，必须回到原始音频和画面复核。
2. prompt 描述预期，不代表实际内容，不得根据 prompt 猜测声音。
3. 每个问题必须写清预期依据、实际证据、具体差异和发生时间。
4. 不输出低置信度猜测、纯主观审美、一般建议或正常内容摘要。
5. 没有明确错误时返回空 JSON 数组。
6. 最终只能返回 JSON 数组，不要 Markdown，不要额外解释。"""


def build_prompt(user_prompt: str, evidence_json: str = "{}") -> str:
    return f"""用户 prompt：
{user_prompt.strip()}

本地专家工具候选证据：
{evidence_json}

不得把 ASR、OCR 或对齐结果直接当作问题真值。请联合分析参考图、视频帧、
音频片段和本地工具证据，回听音频并查看对应时间的视频帧，排除 ASR/OCR
误识别后再下结论。

请检查：
1. 台词内容、语言、发音、说话人数、说话顺序和说话人绑定是否符合 prompt。
   结合嘴部运动、镜头焦点和对话轮次，重点检查语言、台词、声音与主体的绑定关系，
   例如角色 A 的台词由角色 B 发出。旁白或画外音错误绑定到
   可见角色时应报告；证据不足时不得强行绑定。
2. 字幕与实际语音之间是否存在错别字、漏字、多字、语言或说话人对应错误。
   即使 prompt 明确禁止字幕，也要继续核对字幕与实际语音是否吻合。
   “出现了不该出现的字幕”等纯视觉问题留给主视觉 Agent，不在这里重复。
3. 音色、音调、年龄/性别特征、情绪、背景音乐、环境声和动作音效是否
   明显冲突。没有参考音频时不能判断具体人物的声纹，只能判断明显特征。
4. 是否存在明显杂音、爆音、削波、断音、卡顿、异常静音、音量突变、
   重复声音或不自然拼接。
5. 是否存在可由连续相邻帧支持的粗粒度声画冲突。2 fps 画面不能支持
   毫秒级口型判断；音频时间只能按提供的 WAV time_range 定位，
   不得自行估算比分片边界更精细的音频时间。

只输出明确错误。每个问题严格包含：
- 可定位性：统一填写“否”
- 置信度：只能填写“高”或“中”
- 问题说明：包含预期、实际证据和差异
- 问题类型：只能是“音频质量问题”或“文字质量问题”
- 时间区间：“开始秒s - 结束秒s”
- 关键帧秒：空字符串
- BBox：空字符串

没有明确错误时输出 []。"""


def evidence_json(evidence: AuralisEvidence) -> str:
    return json.dumps(
        {
            "asr": asdict(evidence.transcript),
            "subtitles": asdict(evidence.subtitles),
            "speech_subtitle_alignment": asdict(evidence.alignment),
        },
        ensure_ascii=False,
    )


def split_wav_bytes(
    wav_bytes: bytes,
    segment_seconds: float = AUDIO_SEGMENT_SECONDS,
) -> List[Dict[str, Any]]:
    if not wav_bytes:
        return []
    if segment_seconds <= 0:
        raise ValueError("音频分片时长必须大于 0")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            frame_rate = source.getframerate()
            total_frames = source.getnframes()
            compression_type = source.getcomptype()
            compression_name = source.getcompname()
            frames_per_segment = max(1, int(round(frame_rate * segment_seconds)))
            bytes_per_frame = channels * sample_width
            segments: List[Dict[str, Any]] = []
            start_frame = 0
            while start_frame < total_frames:
                requested_frames = min(
                    frames_per_segment,
                    total_frames - start_frame,
                )
                frame_data = source.readframes(requested_frames)
                actual_frames = len(frame_data) // bytes_per_frame
                if actual_frames <= 0:
                    break
                output = io.BytesIO()
                with wave.open(output, "wb") as chunk:
                    chunk.setparams(
                        (
                            channels,
                            sample_width,
                            frame_rate,
                            0,
                            compression_type,
                            compression_name,
                        )
                    )
                    chunk.writeframes(frame_data)
                end_frame = start_frame + actual_frames
                segments.append(
                    {
                        "start_sec": start_frame / frame_rate,
                        "end_sec": end_frame / frame_rate,
                        "wav_bytes": output.getvalue(),
                    }
                )
                start_frame = end_frame
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"无法解析待分片 WAV：{exc}") from exc
    if not segments:
        raise ValueError("待分片 WAV 不含有效音频帧")
    return segments


def build_user_content(
    *,
    reference_images: Iterable[str],
    video_frames: Iterable[Mapping[str, Any]],
    audio_segments: Iterable[Mapping[str, Any]],
    user_prompt: str,
    local_evidence_json: str = "{}",
) -> List[Dict[str, Any]]:
    references = list(reference_images)
    frames = list(video_frames)
    audio = list(audio_segments)
    content: List[Dict[str, Any]] = [
        {"text": build_prompt(user_prompt, local_evidence_json)}
    ]
    if references:
        content.append(
            {
                "text": (
                    f"以下为 {len(references)} 张用户参考图，仅用于确认角色"
                    "外观、身份和明显性别特征。"
                )
            }
        )
        for index, image_url in enumerate(references, start=1):
            content.extend(
                [
                    {"text": f"参考图 {index:02d}/{len(references):02d}"},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_url.split(",", 1)[-1],
                        }
                    },
                ]
            )
    content.append(
        {
            "text": (
                f"以下为待评估视频按 {VIDEO_FRAME_FPS:g} fps 抽取的 "
                f"{len(frames)} 张画面。"
            )
        }
    )
    for index, frame in enumerate(frames, start=1):
        timestamp = float(frame["timestamp_sec"])
        content.extend(
            [
                {
                    "text": (
                        f"视频帧 {index:03d}/{len(frames):03d}，"
                        f"timestamp={timestamp:.2f}s"
                    )
                },
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": str(frame["data_url"]).split(",", 1)[-1],
                    }
                },
            ]
        )
    for index, segment in enumerate(audio, start=1):
        content.extend(
            [
                {
                    "text": (
                        f"音频片段 {index:03d}/{len(audio):03d}，"
                        f"time_range={float(segment['start_sec']):.2f}s - "
                        f"{float(segment['end_sec']):.2f}s。"
                    )
                },
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": base64.b64encode(
                            bytes(segment["wav_bytes"])
                        ).decode("ascii"),
                    }
                },
            ]
        )
    if not audio:
        content.append(
            {
                "text": (
                    "ffprobe 未检测到音轨；不得编造台词、音色、音乐或音效内容。"
                )
            }
        )
    return content


def build_chat_payload(
    model: str,
    parts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "model": model,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": SYSTEM_MESSAGE}, *parts],
            }
        ],
    }


def parse_prediction(
    text: str,
    *,
    duration_sec: float | None = None,
    segment_seconds: float = AUDIO_SEGMENT_SECONDS,
    allowed_boundaries: Iterable[float] | None = None,
) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Gemini 未返回 JSON 数组")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("Gemini 预测结果必须是 JSON 数组")
    explicit_allowed_boundaries = (
        tuple(float(value) for value in allowed_boundaries)
        if allowed_boundaries is not None
        else ()
    )
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError("Gemini 预测数组元素必须是对象")
        explanation = item.get("问题说明")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"Gemini 第 {index} 个问题缺少有效问题说明")
        confidence = item.get("置信度")
        if confidence not in {"高", "中"}:
            raise ValueError(f"Gemini 第 {index} 个问题的置信度必须为高或中")
        problem_type = item.get("问题类型") or "音频质量问题"
        if problem_type not in {"音频质量问题", "文字质量问题"}:
            raise ValueError(
                f"Gemini 第 {index} 个问题的问题类型必须为音频质量问题或文字质量问题"
            )
        time_range = item.get("时间区间")
        if not isinstance(time_range, str):
            raise ValueError(f"Gemini 第 {index} 个问题缺少有效时间区间")
        match = TIME_RANGE_PATTERN.fullmatch(time_range)
        if match is None:
            raise ValueError(
                f"Gemini 第 {index} 个问题的时间区间格式无效：{time_range!r}"
            )
        start_seconds = float(match["start_seconds"])
        end_seconds = float(match["end_seconds"])
        if (
            match["start_minutes"] is not None
            and start_seconds >= 60
        ) or (
            match["end_minutes"] is not None
            and end_seconds >= 60
        ):
            raise ValueError(
                f"Gemini 第 {index} 个问题的时间区间格式无效：{time_range!r}"
            )
        start_total = 60 * float(match["start_minutes"] or 0) + start_seconds
        end_total = 60 * float(match["end_minutes"] or 0) + end_seconds
        if start_total >= end_total:
            raise ValueError(
                f"Gemini 第 {index} 个问题的时间区间格式无效：{time_range!r}"
            )
        duration = float(duration_sec) if duration_sec is not None else None
        if duration_sec is not None:
            assert duration is not None
            if start_total < 0 or end_total > duration + 0.02:
                raise ValueError(
                    f"Gemini 第 {index} 个问题的时间区间超过视频时长 "
                    f"{duration:g}s：{time_range!r}"
                )
        if duration is not None or explicit_allowed_boundaries:
            boundaries = (start_total, end_total)
            for boundary in boundaries:
                on_explicit_boundary = any(
                    abs(boundary - candidate) <= 0.02
                    for candidate in explicit_allowed_boundaries
                )
                on_regular_boundary = (
                    duration is not None
                    and not explicit_allowed_boundaries
                    and abs(
                        boundary / segment_seconds
                        - round(boundary / segment_seconds)
                    )
                    <= 0.02
                )
                on_final_boundary = (
                    duration is not None
                    and not explicit_allowed_boundaries
                    and abs(boundary - duration) <= 0.02
                )
                if (
                    not on_explicit_boundary
                    and not on_regular_boundary
                    and not on_final_boundary
                ):
                    raise ValueError(
                        f"Gemini 第 {index} 个问题的时间端点必须来自音频分片边界："
                        f"{time_range!r}"
                    )
        issue = {key: item.get(key, "") for key in OUTPUT_KEYS}
        issue["可定位性"] = "否"
        issue["问题类型"] = problem_type
        if (
            match["start_minutes"] is not None
            or match["end_minutes"] is not None
        ):
            issue["时间区间"] = f"{start_total:.2f}s - {end_total:.2f}s"
        else:
            issue["时间区间"] = time_range
        issue["关键帧秒"] = ""
        issue["BBox"] = ""
        normalized.append(issue)
    return json.dumps(normalized, ensure_ascii=False)


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        )
    return ""


class GeminiGateway:
    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: int = 900,
        max_attempts: int = 3,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.last_usage: Mapping[str, Any] = {}
        self.last_request_bytes = 0
        self.last_attempts = 0

    def reset_stats(self) -> None:
        self.last_usage = {}
        self.last_request_bytes = 0
        self.last_attempts = 0

    def complete(self, parts: List[Dict[str, Any]]) -> str:
        self.reset_stats()
        body = json.dumps(
            build_chat_payload(self.model, parts),
            ensure_ascii=False,
        ).encode("utf-8")
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(
                self.api_url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout,
                ) as response:
                    result = json.loads(response.read().decode("utf-8"))
                message = result["choices"][0]["message"]
                text = _response_text(message.get("content"))
                if not text:
                    raise ValueError("Gemini 未返回文本结果")
                self.last_usage = result.get("usage", {})
                self.last_request_bytes = len(body) * attempt
                self.last_attempts = attempt
                return text
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Gemini Chat Completions HTTP {exc.code}: {detail[-2000:]}"
                )
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_attempts:
                    raise last_error from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
            ) as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
            time.sleep(min(8, 2 ** (attempt - 1)))
        raise RuntimeError(
            f"Gemini Chat Completions 请求失败：{last_error}"
        ) from last_error


def chat_completion(
    api_url: str,
    api_key: str,
    model: str,
    parts: List[Dict[str, Any]],
    timeout: int,
    max_attempts: int,
) -> Dict[str, Any]:
    """Compatibility API for the former monolithic GPT-D script."""
    gateway = GeminiGateway(
        api_url=api_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        max_attempts=max_attempts,
    )
    text = gateway.complete(parts)
    return {
        "role": "assistant",
        "content": text,
        "_response_usage": gateway.last_usage,
        "_api_attempts": gateway.last_attempts,
        "_request_bytes": gateway.last_request_bytes,
    }


class GeminiAuralisJudge:
    """Verify local tool evidence against raw audio and sampled video."""

    def __init__(
        self,
        gateway: GeminiGateway,
        *,
        input_dir: Path,
    ) -> None:
        self.gateway = gateway
        self.input_dir = input_dir

    def __call__(
        self,
        agent_input: AuralisInput,
        evidence: AuralisEvidence,
    ) -> List[Mapping[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="auralis_gemini_") as temp_text:
            temp_dir = Path(temp_text)
            references = []
            for index, reference in enumerate(
                agent_input.reference_images,
                start=1,
            ):
                source = resolve_local_media(
                    reference,
                    base_dir=self.input_dir,
                    label=f"参考图 {index}",
                )
                proxy = prepare_image_jpeg(
                    source,
                    temp_dir / f"reference_{index:02d}.jpg",
                    max_width=1024,
                )
                references.append(image_data_url(proxy))
            frames = extract_video_frames(
                agent_input.video_path,
                temp_dir / "frames",
                fps=VIDEO_FRAME_FPS,
                max_width=VIDEO_FRAME_WIDTH,
            )
            video_frames = [
                {
                    "timestamp_sec": timestamp,
                    "data_url": image_data_url(path),
                }
                for timestamp, path in frames
            ]
            audio_path = extract_audio_wav(
                agent_input.video_path,
                temp_dir / "audio.wav",
            )
            audio_segments = split_wav_bytes(audio_path.read_bytes())
            parts = build_user_content(
                reference_images=references,
                video_frames=video_frames,
                audio_segments=audio_segments,
                user_prompt=agent_input.user_prompt,
                local_evidence_json=evidence_json(evidence),
            )
            allowed_boundaries = tuple(
                sorted(
                    {
                        float(segment["start_sec"])
                        for segment in audio_segments
                    }
                    | {
                        float(segment["end_sec"])
                        for segment in audio_segments
                    }
                )
            )
            audio_duration = max(allowed_boundaries, default=0.0)
            prediction = parse_prediction(
                self.gateway.complete(parts),
                duration_sec=max(
                    float(evidence.media_metadata["duration_sec"]),
                    audio_duration,
                ),
                segment_seconds=AUDIO_SEGMENT_SECONDS,
                allowed_boundaries=allowed_boundaries,
            )
        return json.loads(prediction)
