"""Small ffmpeg/ffprobe primitives shared by specialist agents."""

from __future__ import annotations

import base64
import json
import mimetypes
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Sequence


def _run(command: Sequence[str], timeout: int = 300) -> str:
    executable = shutil.which(command[0])
    if executable is None:
        raise RuntimeError(f"缺少本地命令：{command[0]}")
    result = subprocess.run(
        [executable, *command[1:]],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{command[0]} 执行失败：{detail[-1000:]}")
    return result.stdout


def probe_video(video_path: Path) -> Dict[str, Any]:
    payload = json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(video_path),
            ]
        )
    )
    video = next(
        (
            stream
            for stream in payload.get("streams", ())
            if stream.get("codec_type") == "video"
        ),
        None,
    )
    if video is None:
        raise ValueError("ffprobe 未找到视频流")
    audio = next(
        (
            stream
            for stream in payload.get("streams", ())
            if stream.get("codec_type") == "audio"
        ),
        None,
    )
    duration = float(
        video.get("duration")
        or payload.get("format", {}).get("duration")
        or 0.0
    )
    return {
        "duration_sec": round(duration, 3),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "has_audio": audio is not None,
        "audio_codec": (audio or {}).get("codec_name", ""),
        "audio_sample_rate": int((audio or {}).get("sample_rate") or 0),
        "audio_channels": int((audio or {}).get("channels") or 0),
    }


def extract_audio_wav(video_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    if not output_path.is_file() or output_path.stat().st_size <= 44:
        raise RuntimeError("ffmpeg 未生成有效 WAV 音轨")
    return output_path


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    fps: float = 2.0,
    max_width: int = 0,
) -> list[tuple[float, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_fps = max(0.1, min(12.0, float(fps)))
    output_pattern = output_dir / "frame_%06d.jpg"
    filters = [f"fps={safe_fps:.6f}"]
    if max_width > 0:
        filters.append(f"scale='trunc(min({int(max_width)},iw)/2)*2':-2")
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            ",".join(filters),
            "-q:v",
            "4",
            str(output_pattern),
        ]
    )
    paths = sorted(output_dir.glob("frame_*.jpg"))
    if not paths:
        raise RuntimeError("ffmpeg 未生成视频帧")
    return [(index / safe_fps, path) for index, path in enumerate(paths)]


def resolve_local_media(
    value: str,
    *,
    base_dir: Path,
    label: str,
) -> Path:
    reference = value.strip()
    if reference.lower().startswith(("http://", "https://", "data:")):
        raise ValueError(f"{label}必须是本地路径：{reference[:120]}")
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        try:
            fallback = (base_dir / path.relative_to(path.anchor)).resolve()
        except ValueError:
            fallback = path
        if fallback.is_file():
            path = fallback
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label}不存在或为空：{path}")
    return path


def prepare_image_jpeg(
    source: Path,
    output: Path,
    *,
    max_width: int,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"scale='trunc(min({int(max_width)},iw)/2)*2':-2",
            "-frames:v",
            "1",
            "-q:v",
            "5",
            str(output),
        ]
    )
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg 未生成图片代理：{source}")
    return output


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return (
        f"data:{mime_type};base64,"
        + base64.b64encode(path.read_bytes()).decode("ascii")
    )
