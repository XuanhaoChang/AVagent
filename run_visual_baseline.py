#!/usr/bin/env python3
"""Evaluate every gt.csv row without exposing gold answers to the model."""

import argparse
import base64
import csv
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from av_eval.project_env import load_project_env


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
INPUT_CSV = INPUT_DIR / "gt.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_CSV = OUTPUT_DIR / "pred.csv"
SKILL_PATH = BASE_DIR / "configs" / "evaluator_prompt.md"
PREDICTION_COLUMN = "GPT预测结果"
SOURCE_COLUMNS = [
    "序号",
    "user_prompt",
    "reference_image_urls",
    "generated_video_url",
    "用户反馈",
    "思考过程及标准答案",
]
INFERENCE_COLUMNS = [
    "序号",
    "user_prompt",
    "reference_image_urls",
    "generated_video_url",
    "用户反馈",
]
DEFAULT_API_URL = ""
DEFAULT_MODEL = ""
API_KEY_ENV = "AVAGENT_API_KEY"
LEGACY_API_KEY_ENV = "ARK_API_KEY"
MAX_VISUAL_TOOL_CALLS = 4
DEFAULT_VIDEO_FRAME_FPS = 2.0
DEFAULT_VIDEO_FRAME_WIDTH = 384
TEXT_VISUAL_MARKERS = re.compile(
    r"文字|字幕|字样|错别字|拼写|文案|标题|招牌|海报|报价单|合同|"
    r"单据|票据|屏幕|水印|logo|写着|写成|写为|显示|展示",
    re.IGNORECASE,
)
QUOTED_TEXT_SPAN = re.compile(r"[\"“]([^\"”]{1,80})[\"”]")
PROFILE_DEFAULTS = {
    "baseline_a": {
        "video_frame_fps": 2.0,
        "max_video_frames": 0,
        "audio_mode": "none",
        "enable_local_crop": False,
    },
    "harness_b": {
        "video_frame_fps": 1.0,
        "max_video_frames": 48,
        "audio_mode": "none",
        "enable_local_crop": True,
    },
    "harness_c": {
        "video_frame_fps": 1.0,
        "max_video_frames": 48,
        "audio_mode": "direct",
        "enable_local_crop": True,
    },
}


def needs_text_visual_verification(input_data: Dict[str, Any]) -> bool:
    """Return whether this row explicitly calls for visible-text inspection."""

    text = "\n".join(
        str(input_data.get(name) or "")
        for name in ("user_prompt", "用户反馈")
    )
    return bool(TEXT_VISUAL_MARKERS.search(text))


def extract_visual_text_candidates(input_data: Dict[str, Any]) -> List[str]:
    """Extract quoted text near visible-text instructions as review hints."""

    candidates: List[str] = []
    for field in ("user_prompt", "用户反馈"):
        text = str(input_data.get(field) or "")
        for match in QUOTED_TEXT_SPAN.finditer(text):
            context = text[max(0, match.start() - 24):match.start()]
            context = re.split(r"[，。；！？—\n\"”]", context)[-1]
            candidate = match.group(1).strip()
            if (
                candidate
                and TEXT_VISUAL_MARKERS.search(context)
                and candidate not in candidates
            ):
                candidates.append(candidate)
    return candidates

OUTPUT_KEYS = [
    "可定位性",
    "置信度",
    "问题说明",
    "问题类型",
    "时间区间",
    "关键帧秒",
    "BBox",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ffprobe_video",
            "description": "读取当前生成视频的时长、帧率和分辨率。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_contact_sheet",
            "description": (
                "生成指定时间范围的接触表并查看。格子按从左到右、从上到下排列；"
                "用于 coarse 或 refine 定位，最多包含 48 帧。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number", "description": "开始秒数，包含。"},
                    "end_sec": {"type": "number", "description": "结束秒数，不超过视频时长。"},
                    "fps": {"type": "number", "description": "抽样帧率；粗定位建议 1-2，细定位建议 6-12。"},
                    "width": {"type": "integer", "description": "每格宽度，建议 256-384。"},
                    "columns": {"type": "integer", "description": "接触表列数，建议 6。"},
                },
                "required": ["start_sec", "end_sec", "fps"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_frame",
            "description": "抽取并查看指定时间的高清原始帧，用于最终关键帧和 BBox 复核。",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp_sec": {"type": "number", "description": "要抽取的秒数。"},
                },
                "required": ["timestamp_sec"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_crop",
            "description": "抽取指定时间的原始帧并按归一化 BBox 裁剪放大，用于脸、手、文字和小物体复核。",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp_sec": {"type": "number"},
                    "x1": {"type": "number"},
                    "y1": {"type": "number"},
                    "x2": {"type": "number"},
                    "y2": {"type": "number"},
                },
                "required": ["timestamp_sec", "x1", "y1", "x2", "y2"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "完成标注并提交最终 JSON 数组。",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "array",
                        "minItems": 0,
                        "items": {
                            "type": "object",
                            "properties": {
                                "可定位性": {"type": "string"},
                                "置信度": {"type": "string"},
                                "问题说明": {"type": "string"},
                                "问题类型": {"type": "string"},
                                "时间区间": {"type": "string"},
                                "关键帧秒": {"type": "string"},
                                "BBox": {"type": "string"},
                            },
                            "required": OUTPUT_KEYS,
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["result"],
                "additionalProperties": False,
            },
        },
    },
]


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = PROFILE_DEFAULTS[args.profile]
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 configs/evaluator_prompt.md 评测输入 CSV。"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=INPUT_CSV,
        help="输入 CSV；默认 input/gt.csv，必须保持相同 schema。",
    )
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条；0 表示全部。")
    parser.add_argument("--start", type=int, default=1, help="从第几条数据开始，默认 1。")
    parser.add_argument(
        "--api-url",
        default=os.getenv(
            "AVAGENT_API_URL",
            os.getenv("VIDEO_EVAL_API_URL", DEFAULT_API_URL),
        ),
        help="OpenAI 兼容的 /chat/completions 完整地址。",
    )
    parser.add_argument(
        "--model",
        default=os.getenv(
            "AVAGENT_VISUAL_MODEL",
            os.getenv("VIDEO_EVAL_MODEL", DEFAULT_MODEL),
        ),
        help="远程 Chat Completions 模型名。",
    )
    parser.add_argument("--timeout", type=int, default=900, help="单次 HTTP 请求超时秒数。")
    parser.add_argument("--api-retries", type=int, default=3, help="HTTP 请求最大尝试次数。")
    parser.add_argument("--max-agent-steps", type=int, default=10, help="每条视频最多远程 agent 轮数。")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_DEFAULTS),
        default="baseline_a",
        help="锁定消融变量的实验配置。",
    )
    parser.add_argument(
        "--video-frame-fps",
        type=float,
        default=None,
        help="覆盖 profile 的初始抽帧 fps。",
    )
    parser.add_argument(
        "--video-frame-width",
        type=int,
        default=DEFAULT_VIDEO_FRAME_WIDTH,
        help="初始视频帧输入的最长边宽度，默认 384，用于控制请求体大小。",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=None,
        help="覆盖 profile 的初始视频帧上限；0 表示按 fps 全量发送。",
    )
    parser.add_argument(
        "--audio-mode",
        choices=("none", "direct", "transcript"),
        default=None,
        help="none=不提供音频；direct=附 WAV；transcript=附带时间戳的 ASR 工具证据。",
    )
    parser.add_argument(
        "--asr-transcript-dir",
        type=Path,
        default=None,
        help="transcript 模式下读取 <视频文件名>.txt 的目录。",
    )
    parser.add_argument(
        "--expert-evidence-dir",
        type=Path,
        default=None,
        help="可选：读取 <视频文件名>.json，并作为待验证的专家工具证据。",
    )
    crop = parser.add_mutually_exclusive_group()
    crop.add_argument("--enable-local-crop", dest="enable_local_crop", action="store_true")
    crop.add_argument("--disable-local-crop", dest="enable_local_crop", action="store_false")
    parser.set_defaults(enable_local_crop=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="独立实验输出，默认仍为 output/pred.csv。",
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        default=None,
        help="可选 JSONL 运行日志，不含标准答案。",
    )
    parser.add_argument("--resume", action="store_true", help="保留 pred.csv 中已有预测，跳过已完成行。")
    return apply_profile_defaults(parser.parse_args())


def read_csv(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.reader(file))


def row_value(header: List[str], row: List[str], name: str) -> str:
    try:
        index = header.index(name)
    except ValueError:
        return ""
    return row[index] if index < len(row) else ""


def read_existing_predictions(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}

    rows = read_csv(path)
    if not rows:
        return {}

    predictions: Dict[int, str] = {}
    for index, row in enumerate(rows[1:], start=1):
        if row and row[-1].strip():
            predictions[index] = row[-1]
    return predictions


def parse_reference_image_urls(raw_value: str) -> List[str]:
    if not raw_value.strip():
        return []
    value = json.loads(raw_value)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("reference_image_urls 必须是本地路径字符串组成的 JSON 数组")
    return [item.strip() for item in value if item.strip()]


def inference_input(header: List[str], row: List[str], row_number: int) -> Dict[str, Any]:
    """Build the only row data that may be sent to the remote model."""
    value: Dict[str, Any] = {
        name: row_value(header, row, name) for name in INFERENCE_COLUMNS
    }
    value["序号"] = value["序号"] or f"#{row_number}"
    value["reference_image_urls"] = parse_reference_image_urls(
        value["reference_image_urls"]
    )
    return value


def build_prompt(input_data: Dict[str, Any], video_path: Path, video_frame_fps: float) -> str:
    return "\n".join(
        [
            "严格执行 system message 中的多图参考文生音视频评测 Skill。",
            f"参考图已按输入字段顺序附在本消息中；生成视频已按 {video_frame_fps:g}fps 手动抽帧为多张图片并按时间顺序附在本消息中。",
            "视频帧文本中的 timestamp 是该帧在原视频中的秒数；如需更精准定位，可继续调用工具查看接触表或高清关键帧。",
            "本次推理输入包含用户反馈列，已排除标准答案列。",
            "用户反馈是高优先级核查线索：反馈中提到的问题要尽最大可能定位和验证；反馈之外的明确问题也要按 Skill 正常检查。",
            "禁止读取或索取 gt.csv、pred.csv、思考过程及标准答案，以及任何真值派生信息。",
            f"生成视频的本地等价副本：{video_path.name}",
            "本行允许使用的输入：",
            json.dumps(input_data, ensure_ascii=False),
            "完成后调用 finish；也可以直接回复符合 Skill 最终格式的 JSON 数组。不要 markdown，不要解释。",
        ]
    )


def build_audio_parts(
    audio_mode: str,
    wav_bytes: bytes,
    transcript: str,
) -> List[Dict[str, Any]]:
    if audio_mode == "none":
        return [
            {
                "type": "text",
                "text": "本次配置未提供音频内容；不得判断台词、音色、杂音或音画同步。",
            }
        ]
    if audio_mode == "direct":
        if not wav_bytes:
            raise ValueError("direct 音频模式缺少 WAV 数据")
        return [
            {
                "type": "text",
                "text": "以下 WAV 是待评估生成视频的真实音轨，可作为音频内容证据。",
            },
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(wav_bytes).decode("ascii"),
                    "format": "wav",
                },
            },
        ]
    if audio_mode == "transcript":
        if not transcript.strip():
            raise ValueError("transcript 音频模式缺少带时间戳转写")
        return [
            {
                "type": "text",
                "text": (
                    "以下内容是外部 ASR 工具证据，不代表模型直接听到了音频；"
                    "只能支持台词内容和时间相关判断，不能单独证明音色、杂音或口型同步。\n"
                    + transcript.strip()
                ),
            }
        ]
    raise ValueError(f"未知 audio_mode：{audio_mode}")


def audio_evidence_instruction(audio_mode: str) -> str:
    if audio_mode == "none":
        return (
            "本次音频证据模式为 none，模型没有收到任何可听音频或 ASR 证据。"
            "不得输出任何音频相关问题，包括台词内容、语言或发音、音色或音调、"
            "说话人声音性别、背景音乐、环境声、音效、杂音、静音、音画或声画同步。"
            "不得输出“缺少音频证据”“无法核实音频”或类似占位问题；"
            "即使用户反馈提到音频，也只能忽略该音频线索并继续检查有视觉证据的问题。"
        )
    if audio_mode in {"direct", "transcript"}:
        return (
            f"本次音频证据模式为 {audio_mode}。"
            "只有获得与问题直接相关且可靠的音频或转写证据时，才可输出音频问题。"
        )
    raise ValueError(f"未知 audio_mode：{audio_mode}")


def build_expert_evidence_parts(evidence: str) -> List[Dict[str, Any]]:
    if not evidence.strip():
        return []
    return [
        {
            "type": "text",
            "text": (
                "以下是专家工具候选证据，不是问题必然存在的证明；"
                "必须结合 prompt、参考图和生成音视频复核：\n" + evidence.strip()
            ),
        }
    ]


def build_user_content(
    input_data: Dict[str, Any],
    video_path: Path,
    video_frame_fps: float,
    model_image_urls: Optional[List[str]] = None,
    model_video_frames: Optional[List[Dict[str, Any]]] = None,
    audio_mode: str = "none",
    wav_bytes: bytes = b"",
    transcript: str = "",
    expert_evidence: str = "",
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": build_prompt(input_data, video_path, video_frame_fps)}
    ]
    references = input_data["reference_image_urls"]
    image_urls = model_image_urls or [media_reference(value, "image") for value in references]
    if len(image_urls) != len(references):
        raise ValueError("参考图代理文件数量与 reference_image_urls 不一致")
    for index, image_url in enumerate(image_urls, start=1):
        content.extend(
            [
                {"type": "text", "text": f"参考图 {index}"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ]
        )
    if str(input_data["generated_video_url"]).strip():
        frames = model_video_frames or []
        content.append(
            {
                "type": "text",
                "text": (
                    f"待评估生成音视频 {video_frame_fps:g}fps 抽帧序列，共 {len(frames)} 帧；"
                    "以下图片按时间顺序排列，用于模拟 video_url 视频输入。"
                ),
            }
        )
        for index, frame in enumerate(frames, start=1):
            timestamp = float(frame["timestamp_sec"])
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"生成视频帧 {index:03d}/{len(frames):03d}，timestamp={timestamp:.2f}s",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(frame["path"]), "detail": "high"},
                    },
                ]
            )
    content.extend(build_audio_parts(audio_mode, wav_bytes, transcript))
    content.extend(build_expert_evidence_parts(expert_evidence))
    return content


def parse_prediction(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start < 0 or end < start:
        raise ValueError("agent 未返回 JSON 数组")

    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("预测结果必须是 JSON 数组")
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("预测结果数组元素必须是对象")
        normalized.append({key: item.get(key, "") for key in OUTPUT_KEYS})
    return json.dumps(normalized, ensure_ascii=False)


def filter_prediction_for_audio_mode(
    prediction: str,
    audio_mode: str,
) -> str:
    if audio_mode not in {"none", "direct", "transcript"}:
        raise ValueError(f"未知 audio_mode：{audio_mode}")
    if audio_mode != "none":
        return prediction
    value = json.loads(prediction)
    audio_type_markers = (
        "音频",
        "人声",
        "声音",
        "音色",
        "音调",
        "配音",
        "背景音乐",
        "BGM",
        "音效",
        "声画同步",
        "音画同步",
    )
    strong_audio_description_markers = (
        "音频",
        "人声",
        "声音",
        "音色",
        "音调",
        "配音",
        "男声",
        "女声",
        "童声",
        "语音",
        "发音",
        "背景音乐",
        "BGM",
        "音效",
        "杂音",
        "静音",
        "声画同步",
        "音画同步",
        "ASR",
        "听到",
        "咳嗽声",
    )
    ambiguous_audio_markers = (
        "台词",
        "对白",
        "语言",
        "口音",
        "说话",
        "旁白",
        "画外音",
        "音乐",
    )
    placeholder_pattern = re.compile(
        r"(?:缺少|没有|未提供|不足|无法).{0,20}"
        r"(?:证据|核实|确认|判断|验证)|"
        r"(?:证据|材料).{0,12}(?:不足|缺失|不充分)"
    )
    filtered = []
    for item in value:
        problem_type = str(item.get("问题类型", ""))
        description = str(item.get("问题说明", ""))
        problem_type_lower = problem_type.lower()
        description_lower = description.lower()
        is_audio_issue = (
            placeholder_pattern.search(description)
            or any(
                marker.lower() in problem_type_lower
                for marker in audio_type_markers
            )
            or any(
                marker.lower() in description_lower
                for marker in strong_audio_description_markers
            )
            or (
                problem_type == "其他"
                and any(
                    marker.lower() in description_lower
                    for marker in ambiguous_audio_markers
                )
            )
        )
        if not is_audio_issue:
            filtered.append(item)
    return json.dumps(filtered, ensure_ascii=False)


def require_api_key() -> str:
    api_key = (
        os.getenv(API_KEY_ENV, "").strip()
        or os.getenv(LEGACY_API_KEY_ENV, "").strip()
    )
    if not api_key:
        raise ValueError(f"缺少环境变量 {API_KEY_ENV}；请设置兼容网关的 Bearer token。")
    return api_key


def response_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                value = part.get("text")
                if isinstance(value, str):
                    texts.append(value)
        return "\n".join(texts).strip()
    return ""


def http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "no structured error detail"
    error = payload.get("error", payload) if isinstance(payload, dict) else {}
    if not isinstance(error, dict):
        return "no structured error detail"
    parts = []
    for key in ("code", "type", "message"):
        value = error.get(key)
        if value is not None:
            parts.append(f"{key}={str(value)[:300]}")
    return ", ".join(parts) or "no structured error detail"


def accumulate_usage(
    stats: Dict[str, Any],
    usage: Dict[str, Any],
    request_bytes: int,
) -> None:
    stats["api_calls"] = int(stats.get("api_calls", 0)) + 1
    stats["request_bytes"] = int(stats.get("request_bytes", 0)) + int(request_bytes)
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            stats[key] = stats.get(key, 0) + value


def chat_completion(
    api_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout: int,
    max_attempts: int,
) -> Dict[str, Any]:
    payload = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto"}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Optional[BaseException] = None
    attempts = max(1, max_attempts)
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            api_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            message = result["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("API 响应 choices[0].message 不是对象")
            message["_response_usage"] = result.get("usage", {})
            message["_request_bytes"] = len(body)
            return message
        except urllib.error.HTTPError as exc:
            detail = http_error_detail(exc)
            last_error = RuntimeError(f"Chat Completions HTTP {exc.code}: {detail}")
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt >= attempts:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
        time.sleep(min(8, 2 ** (attempt - 1)))
    raise RuntimeError(f"Chat Completions 请求失败：{last_error}") from last_error


def run_command(command: List[str], timeout: int = 120) -> str:
    executable = command[0]
    if not Path(executable).is_absolute() and shutil.which(executable) is None:
        conda_candidate = Path.home() / "miniconda3/envs/avagent/bin" / executable
        if conda_candidate.is_file():
            command = [str(conda_candidate), *command[1:]]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"缺少本地命令：{command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"{command[0]} 执行失败：{detail[-1000:]}") from exc
    return result.stdout


def ensure_video(
    generated_video_path: str,
) -> Path:
    """Validate and return the local video path from generated_video_url."""
    video_reference = generated_video_path.strip()
    if not video_reference:
        raise FileNotFoundError("generated_video_url 为空")
    return local_media_path(video_reference, "本地视频")


def probe_video(video_path: Path) -> Dict[str, Any]:
    output = run_command(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video_path)]
    )
    probe = json.loads(output)
    stream = next((item for item in probe.get("streams", []) if item.get("codec_type") == "video"), None)
    if not stream:
        raise ValueError("ffprobe 未找到视频流")
    rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    try:
        numerator, denominator = rate_text.split("/", 1)
        fps = float(numerator) / float(denominator) if float(denominator) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(stream.get("duration") or probe.get("format", {}).get("duration") or 0.0)
    audio = next((item for item in probe.get("streams", []) if item.get("codec_type") == "audio"), None)
    return {
        "duration_sec": round(duration, 3),
        "fps": round(fps, 3),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "has_audio": audio is not None,
        "audio_codec": (audio or {}).get("codec_name", ""),
        "audio_sample_rate": int((audio or {}).get("sample_rate") or 0),
        "audio_channels": int((audio or {}).get("channels") or 0),
    }


def image_data_url(path: Path) -> str:
    return media_data_url(path, "image")


def local_media_path(value: str, label: str) -> Path:
    if value.lower().startswith(("http://", "https://", "data:")):
        raise ValueError(f"{label}必须是本地路径，不支持 URL：{value[:120]}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = INPUT_DIR / path
    path = path.resolve()
    if not path.is_file():
        try:
            fallback = (INPUT_DIR / path.relative_to(path.anchor)).resolve()
        except ValueError:
            fallback = path
        if fallback != path and fallback.is_file():
            path = fallback
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在：{path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label}为空文件：{path}")
    return path


def media_mime_type(path: Path, kind: str) -> str:
    mime_type = mimetypes.guess_type(str(path))[0]
    if mime_type and mime_type.startswith(f"{kind}/"):
        return mime_type

    with path.open("rb") as file:
        header = file.read(16)
    if kind == "image":
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return "image/webp"
        raise ValueError(f"无法识别本地图片格式：{path}")
    if kind == "video":
        if header[4:8] == b"ftyp":
            return "video/mp4"
        raise ValueError(f"无法识别本地视频格式：{path}")
    raise ValueError(f"不支持的媒体类型：{kind}")


def media_data_url(path: Path, kind: str) -> str:
    mime_type = media_mime_type(path, kind)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def media_reference(value: str, kind: str) -> str:
    reference = value.strip()
    if not reference:
        raise ValueError(f"{kind} 媒体引用为空")
    return media_data_url(local_media_path(reference, f"本地{kind}"), kind)


def prepare_model_image(source: Path, output: Path) -> Path:
    """Create a bounded JPEG attachment while retaining enough reference detail."""
    run_command(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(source),
            "-vf", "scale='trunc(min(1024,iw)/2)*2':-2",
            "-frames:v", "1", "-q:v", "5", str(output),
        ]
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg 未生成参考图代理文件：{source}")
    return output


def prepare_model_video_frames(
    source: Path,
    output_dir: Path,
    fps: float,
    width: int,
    max_frames: int,
) -> List[Dict[str, Any]]:
    """Extract a low-rate frame sequence for image_url-only video simulation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_fps = max(0.1, min(12.0, float(fps)))
    safe_width = max(160, min(768, int(width)))
    output_pattern = output_dir / "video_frame_%04d.jpg"
    video_filter = (
        f"fps={safe_fps:.6f},"
        f"scale='trunc(min({safe_width},iw)/2)*2':-2"
    )
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(source),
        "-vf", video_filter, "-q:v", "6",
    ]
    if max_frames > 0:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(output_pattern))
    run_command(command, timeout=300)

    frame_paths = [
        path for path in sorted(output_dir.glob("video_frame_*.jpg"))
        if path.is_file() and path.stat().st_size > 0
    ]
    if not frame_paths:
        raise RuntimeError(f"ffmpeg 未生成视频抽帧：{source}")
    return [
        {"path": path, "timestamp_sec": index / safe_fps}
        for index, path in enumerate(frame_paths)
    ]


def make_contact_sheet(
    video_path: Path,
    output_path: Path,
    meta: Dict[str, Any],
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    duration = float(meta["duration_sec"])
    start = max(0.0, min(duration, float(arguments.get("start_sec", 0.0))))
    end = max(start, min(duration, float(arguments.get("end_sec", duration))))
    if end - start < 0.05:
        raise ValueError("接触表时间范围过短")
    requested_fps = max(0.1, min(12.0, float(arguments.get("fps", 2.0))))
    fps = min(requested_fps, 48.0 / (end - start))
    frame_count = max(1, min(48, int(math.ceil((end - start) * fps))))
    columns = max(1, min(8, int(arguments.get("columns", 6))))
    rows = int(math.ceil(frame_count / columns))
    width = max(160, min(480, int(arguments.get("width", 320))))
    video_filter = (
        f"fps={fps:.6f},scale={width}:-2,"
        f"tile={columns}x{rows}:padding=2:margin=2:color=black"
    )
    run_command(
        [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
            "-i", str(video_path), "-vf", video_filter, "-frames:v", "1", "-q:v", "5",
            str(output_path),
        ]
    )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg 未生成接触表")
    return {
        "image_path": output_path,
        "description": (
            f"接触表范围 {start:.3f}s-{end:.3f}s，共约 {frame_count} 帧，"
            f"{columns} 列，按从左到右、从上到下排列；第 n 格时间约为 "
            f"{start:.3f} + (n-1)/{fps:.6f} 秒。"
        ),
    }


def extract_frame(
    video_path: Path,
    output_path: Path,
    meta: Dict[str, Any],
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    duration = float(meta["duration_sec"])
    timestamp = max(0.0, min(max(0.0, duration - 0.001), float(arguments["timestamp_sec"])))
    run_command(
        [
            "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(video_path),
            "-vf", "scale='min(1600,iw)':-2", "-frames:v", "1", "-q:v", "3", str(output_path),
        ]
    )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg 未生成关键帧")
    return {"image_path": output_path, "description": f"高清关键帧，timestamp={timestamp:.3f}s。"}


def extract_crop(
    video_path: Path,
    output_path: Path,
    meta: Dict[str, Any],
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    duration = float(meta["duration_sec"])
    timestamp = max(0.0, min(max(0.0, duration - 0.001), float(arguments["timestamp_sec"])))
    x1, y1, x2, y2 = (float(arguments[name]) for name in ("x1", "y1", "x2", "y2"))
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("裁剪 BBox 必须满足 0<=x1<x2<=1 且 0<=y1<y2<=1")
    width = int(meta["width"])
    height = int(meta["height"])
    crop_x = max(0, min(width - 1, int(x1 * width)))
    crop_y = max(0, min(height - 1, int(y1 * height)))
    crop_w = max(2, min(width - crop_x, int(math.ceil((x2 - x1) * width))))
    crop_h = max(2, min(height - crop_y, int(math.ceil((y2 - y1) * height))))
    run_command(
        [
            "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(video_path),
            "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale='min(1600,iw*3)':-2",
            "-frames:v", "1", "-q:v", "3", str(output_path),
        ]
    )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg 未生成局部放大图")
    return {
        "image_path": output_path,
        "description": (
            f"局部放大图，timestamp={timestamp:.3f}s，"
            f"BBox=<bbox>{x1:g},{y1:g},{x2:g},{y2:g}</bbox>。"
        ),
    }


def extract_audio_wav(video_path: Path, output_path: Path) -> bytes:
    run_command(
        [
            "ffmpeg", "-v", "error", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output_path),
        ],
        timeout=300,
    )
    if not output_path.is_file() or output_path.stat().st_size <= 44:
        raise RuntimeError("ffmpeg 未生成有效 WAV 音轨")
    return output_path.read_bytes()


def load_asr_transcript(video_path: Path, transcript_dir: Optional[Path]) -> str:
    if transcript_dir is None:
        raise ValueError("transcript 模式必须提供 --asr-transcript-dir")
    transcript_path = transcript_dir / f"{video_path.stem}.txt"
    if not transcript_path.is_file():
        raise FileNotFoundError(f"缺少 ASR 转写：{transcript_path}")
    text = transcript_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"ASR 转写为空：{transcript_path}")
    return text


def load_expert_evidence(video_path: Path, evidence_dir: Optional[Path]) -> str:
    if evidence_dir is None:
        return ""
    evidence_path = evidence_dir / f"{video_path.stem}.json"
    if not evidence_path.is_file():
        return ""
    value = json.loads(evidence_path.read_text(encoding="utf-8"))
    serialized = json.dumps(value, ensure_ascii=False)
    banned = ("思考过程及标准答案", "<thinking>", "标准答案")
    if any(marker in serialized for marker in banned):
        raise ValueError(f"专家证据文件疑似包含真值字段：{evidence_path}")
    return serialized


def tool_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    raw = tool_call.get("function", {}).get("arguments", "{}")
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    return value


def run_agent(
    input_data: Dict[str, Any],
    api_url: str,
    api_key: str,
    model: str,
    timeout: int,
    api_retries: int,
    max_agent_steps: int,
    video_frame_fps: float,
    video_frame_width: int,
    max_video_frames: int,
    audio_mode: str,
    asr_transcript_dir: Optional[Path],
    expert_evidence_dir: Optional[Path],
    enable_local_crop: bool,
    run_stats: Optional[Dict[str, Any]] = None,
    skill_text_override: Optional[str] = None,
    runtime_instruction: str = "",
) -> str:
    video_path = ensure_video(str(input_data.get("generated_video_url", "")))
    text_visual_required = needs_text_visual_verification(input_data)
    text_candidates = extract_visual_text_candidates(input_data)

    skill_text = (
        skill_text_override.strip()
        if skill_text_override is not None
        else SKILL_PATH.read_text(encoding="utf-8").strip()
    )
    text_visual_instruction = ""
    if text_visual_required:
        candidate_note = (
            "执行器从自由格式输入中提取到的画面文字候选为："
            + json.dumps(text_candidates, ensure_ascii=False)
            + "。这些只是待核查线索，不是问题存在的证明。"
            if text_candidates
            else "执行器未可靠提取出固定格式文字候选，仍需按自由格式 prompt 自行识别文字要求。"
        )
        text_visual_instruction = (
            "本行 prompt 或用户反馈明确涉及画面文字。"
            + candidate_note
            + "必须逐项核对相关屏幕文字或单据文字，并且在提交结果前至少调用一次 "
            "extract_frame 或 extract_crop 查看相关时间点的原分辨率证据；"
            "不得只依据 384px 初始概览帧完成文字判断。"
        )
    system_message = "\n\n".join(
        [
            skill_text,
            (
                "运行约束：你通过远程 Chat Completions 工作，不能直接访问本地路径。"
                f"本行的多张参考图已作为多模态内容提供；生成音视频已由执行器按 {video_frame_fps:g}fps 抽帧为按时间顺序排列的多张图片。"
                "本行允许使用用户反馈字段；用户反馈是待验证的高优先级线索，不是问题存在的直接证据。"
                "请优先围绕用户反馈逐项排查并尽最大可能找到可证实问题，同时继续依据 user_prompt、reference_image_urls 和 generated_video_url 正常检查其他明确指令与参考图约束。"
                "必须先结合初始视频帧序列做全片检查；如需精确时间、关键帧或 BBox，再通过 ffprobe_video、make_contact_sheet、extract_frame 工具复核。"
                + ("小区域候选问题可以调用 extract_crop 做局部放大复核。" if enable_local_crop else "")
                +
                "客观画面问题在 finish 前必须查看视觉证据；纯主观评价或建议类反馈不能直接当作可定位问题，但仍需正常检查其他明确问题。"
                + text_visual_instruction
                + audio_evidence_instruction(audio_mode)
                + runtime_instruction.strip()
                +
                f"视觉工具总调用次数不得超过 {MAX_VISUAL_TOOL_CALLS}。"
            ),
        ]
    )
    with tempfile.TemporaryDirectory(prefix="ffmpeg_skill_") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        model_image_urls: List[str] = []
        for index, reference in enumerate(input_data["reference_image_urls"], start=1):
            source = local_media_path(reference, f"本地参考图 {index}")
            proxy = prepare_model_image(source, temp_dir / f"reference_{index:02d}.jpg")
            model_image_urls.append(image_data_url(proxy))
        model_video_frames = prepare_model_video_frames(
            video_path,
            temp_dir / "video_frames",
            video_frame_fps,
            video_frame_width,
            max_video_frames,
        )
        meta: Optional[Dict[str, Any]] = probe_video(video_path)
        effective_audio_mode = audio_mode
        wav_bytes = b""
        transcript = ""
        if audio_mode != "none" and not meta["has_audio"]:
            effective_audio_mode = "none"
        elif audio_mode == "direct":
            wav_bytes = extract_audio_wav(video_path, temp_dir / "audio.wav")
        elif audio_mode == "transcript":
            transcript = load_asr_transcript(video_path, asr_transcript_dir)
        expert_evidence = load_expert_evidence(video_path, expert_evidence_dir)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": build_user_content(
                    input_data,
                    video_path,
                    video_frame_fps,
                    model_image_urls,
                    model_video_frames,
                    effective_audio_mode,
                    wav_bytes,
                    transcript,
                    expert_evidence,
                ),
            },
        ]
        visual_calls = 0
        text_visual_verified = False
        if run_stats is not None:
            run_stats["text_visual_verification_required"] = text_visual_required
            run_stats["text_visual_candidates"] = list(text_candidates)
            run_stats["text_visual_verified"] = False
            run_stats["tool_calls"] = []

        for step in range(1, max(1, max_agent_steps) + 1):
            message = chat_completion(api_url, api_key, model, messages, timeout, api_retries)
            usage = message.pop("_response_usage", {})
            request_bytes = int(message.pop("_request_bytes", 0))
            if run_stats is not None:
                accumulate_usage(
                    run_stats,
                    usage if isinstance(usage, dict) else {},
                    request_bytes,
                )
            text = response_text(message.get("content"))
            calls = message.get("tool_calls") or []
            if not calls:
                if text:
                    if text_visual_required and not text_visual_verified:
                        messages.extend(
                            [
                                {
                                    "role": "assistant",
                                    "content": message.get("content"),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        "尚未完成强制的画面文字高清复核。请先调用 "
                                        "extract_frame 或 extract_crop 查看最相关文字区域，"
                                        "再提交最终 JSON。"
                                    ),
                                },
                            ]
                        )
                        if run_stats is not None:
                            run_stats["forced_text_visual_retries"] = int(
                                run_stats.get("forced_text_visual_retries", 0)
                            ) + 1
                        continue
                    return filter_prediction_for_audio_mode(
                        parse_prediction(text),
                        audio_mode,
                    )
                raise ValueError("远程模型既未返回内容，也未调用工具")

            messages.append(
                {"role": "assistant", "content": message.get("content"), "tool_calls": calls}
            )
            image_parts: List[Dict[str, Any]] = []

            for call_index, call in enumerate(calls, start=1):
                call_id = str(call.get("id") or f"step_{step}_call_{call_index}")
                name = call.get("function", {}).get("name", "")
                arguments: Dict[str, Any] = {}
                try:
                    arguments = tool_arguments(call)
                    if name == "finish":
                        if text_visual_required and not text_visual_verified:
                            raise ValueError(
                                "画面文字任务提交前必须先调用 extract_frame 或 extract_crop"
                            )
                        if run_stats is not None:
                            run_stats["tool_calls"].append(
                                {
                                    "step": step,
                                    "name": name,
                                    "arguments": arguments,
                                    "ok": True,
                                }
                            )
                        return filter_prediction_for_audio_mode(
                            parse_prediction(
                                json.dumps(
                                    arguments.get("result"),
                                    ensure_ascii=False,
                                )
                            ),
                            audio_mode,
                        )
                    if name == "ffprobe_video":
                        meta = meta or probe_video(video_path)
                        tool_result: Dict[str, Any] = {"ok": True, **meta}
                    elif name in {"make_contact_sheet", "extract_frame", "extract_crop"}:
                        if visual_calls >= MAX_VISUAL_TOOL_CALLS:
                            raise ValueError(f"视觉工具最多调用 {MAX_VISUAL_TOOL_CALLS} 次")
                        if name == "extract_crop" and not enable_local_crop:
                            raise ValueError("当前实验 profile 禁用局部裁剪")
                        meta = meta or probe_video(video_path)
                        visual_calls += 1
                        output_path = temp_dir / f"step_{step:02d}_{call_index:02d}.jpg"
                        if name == "make_contact_sheet":
                            visual = make_contact_sheet(video_path, output_path, meta, arguments)
                        elif name == "extract_frame":
                            visual = extract_frame(video_path, output_path, meta, arguments)
                        else:
                            visual = extract_crop(video_path, output_path, meta, arguments)
                        tool_result = {"ok": True, "description": visual["description"]}
                        if name in {"extract_frame", "extract_crop"}:
                            text_visual_verified = True
                            if run_stats is not None:
                                run_stats["text_visual_verified"] = True
                        image_parts.extend(
                            [
                                {"type": "text", "text": f"tool_call_id={call_id}: {visual['description']}"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_url(visual["image_path"]),
                                        "detail": "high",
                                    },
                                },
                            ]
                        )
                    else:
                        raise ValueError(f"未知工具：{name}")
                except Exception as exc:
                    tool_result = {"ok": False, "error": str(exc)}
                if run_stats is not None:
                    trace = {
                        "step": step,
                        "name": name,
                        "arguments": arguments,
                        "ok": bool(tool_result.get("ok")),
                    }
                    if "description" in tool_result:
                        trace["description"] = tool_result["description"]
                    if "error" in tool_result:
                        trace["error"] = tool_result["error"]
                    run_stats["tool_calls"].append(trace)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

            if image_parts:
                messages.append({"role": "user", "content": image_parts})

    raise RuntimeError(f"远程 agent 超过最大轮数 {max_agent_steps}，仍未提交最终结果")


def main() -> int:
    load_project_env(BASE_DIR / ".env.local")
    args = parse_args()
    api_key = require_api_key()
    if not args.api_url:
        raise ValueError("缺少 --api-url 或 AVAGENT_API_URL。")
    if not args.model:
        raise ValueError("缺少 --model 或 AVAGENT_VISUAL_MODEL。")
    table = read_csv(args.input_csv)
    if not table:
        raise ValueError("gt.csv 为空")

    header, source_rows = table[0], table[1:]
    if header != SOURCE_COLUMNS:
        raise ValueError(
            "gt.csv 列不符合预期；必须严格为：" + ",".join(SOURCE_COLUMNS)
        )
    if any(len(row) != len(SOURCE_COLUMNS) for row in source_rows):
        raise ValueError("gt.csv 存在列数不一致的数据行")
    start_index = max(0, args.start - 1)
    end_index = len(source_rows) if args.limit <= 0 else min(len(source_rows), start_index + args.limit)
    output_csv = args.output_csv
    existing_predictions = read_existing_predictions(output_csv) if args.resume else {}
    if existing_predictions:
        print(f"resume: loaded {len(existing_predictions)} existing predictions", flush=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.run_log is not None:
        args.run_log.parent.mkdir(parents=True, exist_ok=True)
    failed_rows = 0
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SOURCE_COLUMNS + [PREDICTION_COLUMN])
        for index, row in enumerate(source_rows, start=1):
            prediction = existing_predictions.get(index, "")
            if prediction:
                print(f"[{index:03d}/{len(source_rows)}] skip existing row_{index:04d}.mp4", flush=True)
            elif start_index <= index - 1 < end_index:
                print(f"[{index:03d}/{len(source_rows)}] processing row_{index:04d}.mp4", flush=True)
                started = time.monotonic()
                error_text = ""
                run_stats: Dict[str, Any] = {}
                try:
                    prediction = run_agent(
                        inference_input(header, row, index),
                        args.api_url,
                        api_key,
                        args.model,
                        args.timeout,
                        args.api_retries,
                        args.max_agent_steps,
                        args.video_frame_fps,
                        args.video_frame_width,
                        args.max_video_frames,
                        args.audio_mode,
                        args.asr_transcript_dir,
                        args.expert_evidence_dir,
                        args.enable_local_crop,
                        run_stats,
                    )
                except Exception as exc:
                    failed_rows += 1
                    error_text = str(exc)
                    print(f"  failed: {exc}", flush=True)
                if args.run_log is not None:
                    record = {
                        "row_index": index,
                        "序号": row_value(header, row, "序号"),
                        "profile": args.profile,
                        "model": args.model,
                        "audio_mode": args.audio_mode,
                        "success": bool(prediction),
                        "elapsed_sec": round(time.monotonic() - started, 3),
                        "error": error_text,
                        **run_stats,
                    }
                    with args.run_log.open("a", encoding="utf-8") as log_file:
                        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.writerow([row[header.index(name)] for name in SOURCE_COLUMNS] + [prediction])
            file.flush()

    print(f"done: {output_csv}; failed_rows={failed_rows}", flush=True)
    return 1 if failed_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
