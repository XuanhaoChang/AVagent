"""Gemini gateway judge used by the Auralis specialist agent.

ASR, OCR and deterministic alignment run locally. Gemini uses the local ASR/OCR
results as the primary decision evidence; any media parts are auxiliary context.
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


OUTPUT_FORMAT = """输出格式（必须严格遵守）：
- 顶层必须是 JSON 数组；没有明确错误时输出 []。
- 数组中的每个对象必须包含以下 7 个键，不能省略、不能改名、不能输出 null：
  可定位性、置信度、问题说明、问题类型、时间区间、关键帧秒、BBox。
- 可定位性固定为“否”；置信度只能是“高”或“中”。
- 问题说明必须是非空字符串，并同时写出预期、实际证据、具体差异和发生时间。
- 问题类型只能是“音频质量问题”或“文字质量问题”。
- 时间区间必须写成“开始秒s - 结束秒s”，例如“1.20s - 2.35s”。
  开始和结束可以是原始证据支持的任意小数，不要求取整或对齐到预先切分边界。
- 关键帧秒和 BBox 固定为空字符串。
- 只输出 JSON 数组本身，不要 Markdown 代码围栏、解释文字或注释。

唯一允许的对象形状示例：
[{"可定位性":"否","置信度":"高","问题说明":"预期为……，实际听到……，差异是……，发生在……。","问题类型":"音频质量问题","时间区间":"1.20s - 2.35s","关键帧秒":"","BBox":""}]"""


SYSTEM_MESSAGE = f"""你是 Auralis 音频与字幕取证专家。

你将收到用户 prompt、参考图、带时间戳的视频帧、本地 ASR、OCR 和语音字幕对齐工具输出；
请求可能附带 WAV 作为辅助上下文，但台词内容和读音问题不要求回听完整 WAV，依据本地 ASR 及其受约束候选评分判定。请只报告明确、可验证的音频错误或音频参与才能判断的字幕错误。

精确音画同步、口型延迟和音频与画面动作的同步判断由 AVBench 负责。本 Agent不需判断或报告音画同步问题；视频帧只用于确认字幕和可见说话人，不用于口型同步测量。

安全与证据约束：
1. ASR 是本流程对实际台词内容和读音差异的检测与判定依据；OCR 是字幕文字的判定依据。不得因为没有回听 WAV，就否定受约束 ASR 已经确认的台词差异。
2. prompt 描述预期，不代表实际内容，不得根据 prompt 猜测声音。
3. 每个问题必须写清预期依据、实际证据、具体差异和发生时间。
4. 不输出低置信度猜测、纯主观审美、一般建议或正常内容摘要。
5. 没有明确错误时返回空 JSON 数组。

字幕和台词的判断规则：
- 用户 prompt 是自由格式，可能没有台词，也可能用任意格式写台词。不得自行依赖引号、角色冒号、字段名或 Markdown 重新猜测台词。`constrained_asr` 已把完整 ASR 片段与 prompt全文做格式无关的字符级局部对齐；只有其中带 `prompt_start`、`prompt_end` 和`prompt_source_text` 的原文锚点才是可用于受约束评分的 prompt 参考文本。
- `constrained_asr.status=no_reference_dialogue` 表示没有找到可靠原文锚点，不等于 prompt
  一定没有任何语音要求，但你不得据此编造标准台词。此时只能按 ASR、OCR 和画面已有证据
  检查其他问题。
- 对每个 `candidate_scores` 项严格按本地判定分支处理：`observed_preferred` 表示同一局部
  音频的 CTC 似然明确支持 ASR 实际候选，必须报告 prompt 预期与实际读音/台词差异；
  `expected_preferred` 表示自由 ASR 差异更可能是假阳性，不得报告为确定台词错误；
  `orthographic_homophone` 表示两端拼音（含声调）一致，字符 CTC 偏好不能构成读音错误；
  `ambiguous`、`pronunciation_unverified` 或 `scoring_failed` 只能保留为审计证据，不得升级
  为明确错误。
- OCR 与有原文锚点的 prompt 参考文本不一致，说明字幕文字可能不符合要求；ASR 与 OCR
  不一致，说明字幕与实际语音不一致。每一种有独立证据支持的差异应分别处理，不能因为
  另一种比较通过而忽略。受约束评分只裁决音频候选，不替代 OCR 字幕判断。
- 比如原文锚点预期“发霉”、ASR 为“发膜”，且 `decision=observed_preferred` 时，必须报告
  “发霉”被读成“发膜”，不要求回听完整 WAV；若判定为其他分支则不得照搬该结论。
- 比较字幕内容时必须同时参考 OCR 和原始字幕画面：如果 OCR 显示“美缝”，而 ASR
  单独写成“每份/每分”，只能据此报告 ASR 与字幕/台词的不一致；不得把 ASR 差异改写
  成字幕错误。查看原始字幕画面只用于核对 OCR，不要求回听音频排除 ASR。
- “prompt 要求无字幕但视频出现字幕”属于字幕是否存在的要求违例；如果该字幕同时有
  错字、漏字、多字或与台词不一致，还必须另报“字幕内容错误”。这两类问题不能合并，
  也不能用其中一类替代另一类。仅凭 OCR 误检的 logo、衣服文字或水印，不得当作字幕。
- 单个采样帧中孤立出现的单字符 OCR 不足以证明画面存在文字或字幕，必须有相邻帧重复 OCR、
  可信 ASR-OCR 对齐或其他明确文字证据才能报告。即使某一帧的人脸、衣纹或背景轮廓看起来像
  单字，也不得自行将其升级为字幕缺陷；本地工具已从 `subtitles.segments` 中排除这类未验证候选。

说话人绑定和音色规则：
- 本地 ASR metadata 中的 `prompt_speech_plan` 只描述 prompt 明确锚定的预期台词轮次：
  `scope=closed` 表示这些台词构成封闭脚本，未分配语音可作为多余台词候选；`scope=partial`
  表示只允许检查已锚定轮次，其他语音保持 unknown/unassigned，不能擅自绑定；`scope=none`
  时不得仅凭 prompt 做角色绑定。`expected_speaker_count` 是评测目标，不是实际声纹簇 GT，
  不能用它否定 CAM++ 的声学结果或强行把声音拆成该数量。
- 优先使用 `speaker_diarization.speaker_turns` 和按这些声纹边界拆分后的 ASR segments。
  `clustering.granularity_conflict=true` 表示旧的标点句级 speaker 数少于细粒度声纹数；此时严禁
  根据整句单一标签声称“全程同一声纹”或据此输出高置信度绑定错误。
- 必须检查 prompt 中指定的角色是否真的说出了对应台词，尤其是角色 A 的台词是否由角色 B
  发出、一个角色的台词是否被拆给多个声音、旁白/画外音是否被错误绑定到可见角色。
- 如果本地证据包含 CAM++ 的 `speaker`/`spk` 标签，必须把它作为语音绑定检测证据，而不只是
  参考信息：相同标签表示这些片段被 CAM++ 视为同一匿名声纹候选，不同标签表示候选声音发生
  变化。先按 ASR 时间戳和 prompt 的台词轮次建立“预期角色 -> 实际 spk”的对照，再检查：
  (a) 同一个 prompt 台词轮次是否被多个 spk 拆开；(b) 同一个 spk 是否跨越了 prompt 中明确属于
  不同角色的台词轮次；(c) 相邻台词的 spk 顺序是否与角色轮次冲突。
- 不要把 `spk=0/1` 当作角色姓名，但必须先建立一个临时的“人物 -> 匿名 spk”映射再判断绑定。
  映射只能来自证据中的锚定轮次：prompt 明确指定角色的台词、该台词与 ASR 文本/时间重合，
  并且同一人物在其他轮次使用相同 spk，或画面明确显示该人物正在说话。先在内部列出例如
  `贺雨棠 -> spk0`、`李莲 -> spk1`、`林建军 -> spk2` 的证据和置信度；没有足够锚点的角色
  才保持未映射。
- 建立映射后，逐段检查“prompt 预期角色 -> 实际 ASR spk”，不能只检查同一句台词内部是否
  使用了同一个 spk。若预期李莲的整句台词全部由 spk0 说出，而锚定映射显示 spk0 属于贺雨棠、
  spk1 属于李莲，整句仍然是角色绑定错误；若该句前半段为 spk0、后半段为 spk1，则应报告
  前半段的错误角色绑定，并指出后半段与李莲映射一致。也就是说，“台词内部连续”不能证明
  “角色绑定正确”。
- 不要要求先有外部参考声纹才能做上述映射；这里不是声纹识别或绝对身份认证，而是利用 prompt
  台词轮次、ASR 时间戳、文本和 CAM++ spk 建立本视频内部的相对人物-声纹对应关系。只要映射
  锚点和冲突明确，就报告角色绑定错误，在问题说明中写明映射依据、预期角色、实际 spk、台词
  和时间区间；不要因为 spk 是匿名标签或没有参考声纹而静默跳过。
- 例如：如果 prompt 先要求角色甲说台词1、再要求角色乙说台词2、之后又要求角色甲说台词3，
  且锚定轮次得到 `角色甲 -> spk0`、`角色乙 -> spk1`，那么台词2全部由 spk0 说出也必须报告
  为角色乙台词被角色甲声纹发出；如果台词2前半段为 spk0、后半段为 spk1，则只把前半段
  作为错误绑定，不能笼统写成“整句应由 spk1 完整发出”。
- CAM++ 是候选声纹分离工具，不是角色 GT；只有在上述相对冲突不明确时，才降级为不报告，不能
  把“匿名”误解成“不可用于绑定检查”。
- `speaker_binding_evidence.role_to_speakers` 和 `speaker_to_roles` 是按台词文本锚定得到的相对
  映射；必须逐项核查 `split_role_candidates`、`shared_speaker_candidates` 和
  `unassigned_segments`。这些字段是待核查证据，不是用 prompt 人数强制重聚类的结果；
  `scope=closed` 时未分配片段可作为多余语音候选，`scope=partial/none` 时不能据此报错。
- 检查角色的明显年龄/性别声学特征是否与 prompt、参考材料和画面身份冲突，例如
  男性角色明显发女声或女性角色明显发男声。参考材料若包含参考音频或明确音色描述，
  才能进一步核查音色、音调、年龄、音质、口音和其他细节；只有参考图时不得声称
  具体声纹或详细音色已经匹配。

{OUTPUT_FORMAT}"""


def build_prompt(user_prompt: str, evidence_json: str = "{}") -> str:
    return f"""用户 prompt：
{user_prompt.strip()}

本地专家工具候选证据：
{evidence_json}

ASR 是台词内容和读音问题的判定依据，OCR 是字幕文字问题的判定依据。请结合
prompt 原文锚点、ASR 时间戳、受约束候选评分、OCR/字幕画面和对齐结果；不要求回听完整 WAV，
也不能因为没有音频复核就删除 `observed_preferred` 已确认的台词差异。

精确音画同步和口型延迟由 AVBench 负责，本次只检查音频内容、音频质量和
音频参与才能判断的字幕问题，不得输出音画同步问题。

请检查：
1. 台词内容、语言、发音、说话人数、说话顺序和说话人绑定是否符合 prompt。
   结合 ASR 时间戳、对话轮次和画面中可见主体，重点检查语言、台词、声音与主体的绑定关系，
   例如角色 A 的台词由角色 B 发出。旁白或画外音错误绑定到
   可见角色时应报告；如果证据中有 CAM++ 的 speaker 标签，必须纳入轮次和角色绑定判断，
   但不得把 spk 编号直接当作角色名；证据不足时不得强行绑定。
2. 按 SYSTEM_MESSAGE 中的受约束评分分支核对台词；只把带原始 prompt 字符区间的文本
   当作参考台词。没有可靠原文锚点时核对 ASR-OCR。报告错别字、漏字、多字、语言或
   说话人对应错误，并明确写出比较的两端。报告字幕与实际语音之间的错误；即使 prompt 明确禁止字幕，也要继续核对
   已出现字幕的内容是否与实际语音吻合；纯粹的字幕存在/不存在问题留给主视觉 Agent，
   但字幕内容错误仍需在本 Agent 报告。
3. 音色、音调、年龄/性别特征、情绪、背景音乐、环境声和动作音效是否
   明显冲突。没有参考音频时不能判断具体人物的声纹，只能判断明显特征。
4. 是否存在明显杂音、爆音、削波、断音、卡顿、异常静音、音量突变、
   重复声音或不自然拼接。

只输出明确错误。

{OUTPUT_FORMAT}
"""


def evidence_json(evidence: AuralisEvidence) -> str:
    return json.dumps(
        {
            "asr": asdict(evidence.transcript),
            "subtitles": asdict(evidence.subtitles),
            "speech_subtitle_alignment": asdict(evidence.alignment),
            "constrained_asr": dict(evidence.constrained_asr),
            "subtitle_evidence_policy": {
                "isolated_single_frame_single_character_is_unverified": True,
                "requires_temporal_recurrence_or_alignment_support": True,
            },
        },
        ensure_ascii=False,
    )


def wav_duration_sec(wav_bytes: bytes) -> float:
    """Return the duration of one complete WAV without slicing its payload."""
    if not wav_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as source:
            frame_rate = source.getframerate()
            if frame_rate <= 0:
                raise ValueError("WAV 采样率必须大于 0")
            return source.getnframes() / frame_rate
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"无法解析 WAV：{exc}") from exc


def build_user_content(
    *,
    reference_images: Iterable[str],
    video_frames: Iterable[Mapping[str, Any]],
    audio_wav: bytes | None,
    audio_duration_sec: float | None = None,
    user_prompt: str,
    local_evidence_json: str = "{}",
) -> List[Dict[str, Any]]:
    references = list(reference_images)
    frames = list(video_frames)
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
    if audio_wav:
        duration_label = ""
        if audio_duration_sec is not None:
            duration_label = f"，总时长约 {float(audio_duration_sec):.2f}s"
        content.extend(
            [
                {
                    "text": (
                        "以下为可选的原始 WAV 辅助输入，"
                        f"时间轴从 0.00s 开始{duration_label}；台词内容和读音问题以本地 ASR "
                        "受约束候选判定为准，不要求回听完整 WAV，也不要用音频输入否定 "
                        "`observed_preferred` 已确认的差异。"
                    )
                },
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": base64.b64encode(bytes(audio_wav)).decode("ascii"),
                    }
                },
            ]
        )
    else:
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
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError("Gemini 预测数组元素必须是对象")
        missing_keys = [key for key in OUTPUT_KEYS if key not in item]
        if missing_keys:
            raise ValueError(
                f"Gemini 第 {index} 个问题缺少必填字段：{', '.join(missing_keys)}"
            )
        explanation = item.get("问题说明")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"Gemini 第 {index} 个问题缺少有效问题说明")
        confidence = item.get("置信度")
        if confidence not in {"高", "中"}:
            raise ValueError(f"Gemini 第 {index} 个问题的置信度必须为高或中")
        problem_type = item.get("问题类型")
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
    """Review local ASR/OCR/alignment evidence against the prompt and video."""

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
            audio_wav = audio_path.read_bytes()
            audio_duration = wav_duration_sec(audio_wav)
            parts = build_user_content(
                reference_images=references,
                video_frames=video_frames,
                audio_wav=audio_wav,
                audio_duration_sec=audio_duration,
                user_prompt=agent_input.user_prompt,
                local_evidence_json=evidence_json(evidence),
            )
            prediction = parse_prediction(
                self.gateway.complete(parts),
                duration_sec=max(
                    float(evidence.media_metadata["duration_sec"]),
                    audio_duration,
                ),
            )
        return json.loads(prediction)
