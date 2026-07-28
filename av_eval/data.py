"""Dataset, media, gold parsing, and ffprobe helpers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldParseResult:
    status: str
    items: list[dict[str, Any]]
    reason: str = ""


_FENCE_RE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_gold_array(text: str) -> GoldParseResult:
    """Parse only fenced JSON arrays and never infer gold from thinking text."""

    blocks = _FENCE_RE.findall(text or "")
    if not blocks:
        return GoldParseResult("needs_review", [], "missing_fenced_json")
    last_error = "no_array_block"
    for block in reversed(blocks):
        try:
            value = json.loads(block.strip())
        except json.JSONDecodeError as exc:
            last_error = f"invalid_json:{exc.msg}"
            continue
        if not isinstance(value, list):
            last_error = "top_level_not_array"
            continue
        if not all(isinstance(item, dict) for item in value):
            return GoldParseResult("needs_review", [], "array_item_not_object")
        return GoldParseResult("valid", value)
    return GoldParseResult("needs_review", [], last_error)


def resolve_legacy_media_path(value: str, media_root: Path) -> Path:
    """Resolve legacy absolute `/data02/...` paths beneath a local media root."""

    raw = (value or "").strip()
    if not raw:
        raise FileNotFoundError("媒体路径为空")
    if raw.lower().startswith(("http://", "https://", "data:")):
        raise ValueError(f"媒体必须是本地路径：{raw[:120]}")
    original = Path(raw).expanduser()
    candidates = [original]
    if original.is_absolute():
        candidates.append(media_root / str(original).lstrip("/"))
    else:
        candidates.append(media_root / original)
    root = media_root.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            if candidate != original and not resolved.is_relative_to(root):
                continue
            return resolved
    raise FileNotFoundError(f"媒体不存在：{raw}")


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract a tar archive after rejecting traversal and link entries."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"压缩包包含越界路径：{member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"压缩包包含链接：{member.name}")
        tar.extractall(destination, members=members, filter="data")


def find_binary(name: str) -> str:
    direct = shutil.which(name)
    if direct:
        return direct
    candidate = Path.home() / "miniconda3/envs/avagent/bin" / name
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f"缺少命令：{name}")


def probe_media(path: Path) -> dict[str, Any]:
    output = subprocess.run(
        [
            find_binary("ffprobe"),
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(output.stdout)
    video = next((s for s in payload.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in payload.get("streams", []) if s.get("codec_type") == "audio"), None)
    duration = video.get("duration") or payload.get("format", {}).get("duration") or 0
    return {
        "duration_sec": round(float(duration), 3),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "video_codec": video.get("codec_name", ""),
        "has_audio": audio is not None,
        "audio_codec": (audio or {}).get("codec_name", ""),
        "audio_sample_rate": int((audio or {}).get("sample_rate") or 0),
        "audio_channels": int((audio or {}).get("channels") or 0),
    }
