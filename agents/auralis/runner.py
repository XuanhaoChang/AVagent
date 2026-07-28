"""CLI orchestration for GPT visual review plus the Auralis audio agent."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import call_ffmpeg_skill as gpt_a
from agents.auralis.agent import AuralisAgent
from agents.auralis.gemini_backend import (
    DEFAULT_MODEL,
    GeminiAuralisJudge,
    GeminiGateway,
)
from agents.auralis.schemas import AuralisInput
from av_eval.project_env import load_project_env
from tools.speech_subtitle_alignment.tool import check_speech_subtitle_alignment
from tools.speech_transcription.backends.faster_whisper import (
    FasterWhisperBackend,
)
from tools.speech_transcription.cuda import (
    cuda_library_dirs,
    cuda_process_environment,
)
from tools.speech_transcription.tool import transcribe_speech
from tools.subtitle_extraction.backends.rapidocr import RapidOCRBackend
from tools.subtitle_extraction.tool import extract_subtitles


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "input"
INPUT_CSV = INPUT_DIR / "gt.csv"
OUTPUT_CSV = BASE_DIR / "output" / "pred_gpt_d.csv"
DEFAULT_API_URL = gpt_a.DEFAULT_API_URL
DEFAULT_GPT_A_MODEL = gpt_a.DEFAULT_MODEL
API_KEY_ENV = "ARK_API_KEY"
PREDICTION_COLUMN = gpt_a.PREDICTION_COLUMN
SOURCE_COLUMNS = gpt_a.SOURCE_COLUMNS
INFERENCE_COLUMNS = (
    "序号",
    "user_prompt",
    "reference_image_urls",
    "generated_video_url",
)
VIDEO_FRAME_FPS = 2.0
VIDEO_FRAME_WIDTH = 384


def inference_input(
    header: List[str],
    row: List[str],
    row_number: int,
) -> Dict[str, Any]:
    value = {name: gpt_a.row_value(header, row, name) for name in INFERENCE_COLUMNS}
    value["序号"] = value["序号"] or f"#{row_number}"
    value["reference_image_urls"] = gpt_a.parse_reference_image_urls(
        value["reference_image_urls"]
    )
    return value


def _prediction_array(text: str, source: str) -> List[Dict[str, Any]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} 预测不是合法 JSON 数组") from exc
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(f"{source} 预测必须是 JSON 对象数组")
    return value


def merge_predictions(
    gpt_a_prediction: str,
    audio_prediction: str,
) -> str:
    return json.dumps(
        _prediction_array(gpt_a_prediction, "GPT-A")
        + _prediction_array(audio_prediction, "Auralis 音频"),
        ensure_ascii=False,
    )


def read_matching_predictions(
    path: Path,
    source_header: List[str],
    source_rows: List[List[str]],
) -> Dict[int, str]:
    if not path.exists():
        return {}
    rows = gpt_a.read_csv(path)
    if not rows:
        return {}
    expected_header = source_header + [PREDICTION_COLUMN]
    if rows[0] != expected_header:
        raise ValueError(f"resume 输出表头不匹配：{path}")
    if len(rows) - 1 > len(source_rows):
        raise ValueError(f"resume 输出行数超过当前输入：{path}")
    predictions: Dict[int, str] = {}
    for index, output_row in enumerate(rows[1:], start=1):
        if len(output_row) != len(expected_header):
            raise ValueError(f"resume 第 {index} 行列数不一致：{path}")
        if output_row[: len(source_header)] != source_rows[index - 1]:
            raise ValueError(f"resume 第 {index} 行源字段不一致：{path}")
        if output_row[-1].strip():
            predictions[index] = output_row[-1]
    return predictions


def build_auralis_agent(
    *,
    api_url: str,
    api_key: str,
    gemini_model: str,
    timeout: int,
    api_retries: int,
) -> tuple[AuralisAgent, GeminiGateway, FasterWhisperBackend]:
    gateway = GeminiGateway(
        api_url=api_url,
        api_key=api_key,
        model=gemini_model,
        timeout=timeout,
        max_attempts=api_retries,
    )
    asr_backend = FasterWhisperBackend()
    ocr_backend = RapidOCRBackend()
    agent = AuralisAgent(
        transcribe_speech=lambda path: transcribe_speech(
            path,
            backend=asr_backend,
        ),
        extract_subtitles=lambda path: extract_subtitles(
            path,
            backend=ocr_backend,
        ),
        align_speech_subtitles=check_speech_subtitle_alignment,
        judge=GeminiAuralisJudge(gateway, input_dir=INPUT_DIR),
    )
    return agent, gateway, asr_backend


def run_audio_row(
    input_data: Dict[str, Any],
    *,
    api_url: str,
    api_key: str,
    model: str,
    timeout: int,
    api_retries: int,
    run_stats: Dict[str, Any] | None = None,
    auralis_agent: AuralisAgent | None = None,
    gateway: GeminiGateway | None = None,
) -> str:
    local_gateway = gateway
    if auralis_agent is None:
        auralis_agent, local_gateway, _ = build_auralis_agent(
            api_url=api_url,
            api_key=api_key,
            gemini_model=model,
            timeout=timeout,
            api_retries=api_retries,
        )
    if local_gateway is not None:
        local_gateway.reset_stats()
    result = auralis_agent.analyze(
        AuralisInput(
            video_path=gpt_a.ensure_video(input_data["generated_video_url"]),
            user_prompt=input_data["user_prompt"],
            reference_images=tuple(input_data["reference_image_urls"]),
            sample_id=str(input_data.get("序号") or ""),
        )
    )
    if run_stats is not None:
        run_stats["status"] = result.status
        if result.evidence is not None:
            run_stats["asr_backend"] = result.evidence.transcript.backend
            run_stats["asr_model"] = result.evidence.transcript.model
            run_stats["asr_device"] = result.evidence.transcript.device
            run_stats["subtitle_backend"] = result.evidence.subtitles.backend
            run_stats["alignment_issue_count"] = len(
                result.evidence.alignment.issues
            )
        if local_gateway is not None and local_gateway.last_attempts:
            gpt_a.accumulate_usage(
                run_stats,
                dict(local_gateway.last_usage),
                local_gateway.last_request_bytes,
            )
            run_stats["api_calls"] = local_gateway.last_attempts
    return json.dumps(list(result.issues), ensure_ascii=False)


def run_combined_row(
    gpt_a_input: Dict[str, Any],
    audio_input: Dict[str, Any],
    *,
    api_url: str,
    api_key: str,
    gpt_a_model: str,
    gemini_model: str,
    timeout: int,
    api_retries: int,
    max_gpt_a_agent_steps: int,
    run_stats: Dict[str, Any] | None = None,
    auralis_agent: AuralisAgent | None = None,
    gateway: GeminiGateway | None = None,
) -> str:
    gpt_a_stats: Dict[str, Any] = {}
    gpt_a_prediction = ""
    gpt_a_error: Exception | None = None
    try:
        gpt_a_prediction = gpt_a.run_agent(
            gpt_a_input,
            api_url,
            api_key,
            gpt_a_model,
            timeout,
            api_retries,
            max_gpt_a_agent_steps,
            VIDEO_FRAME_FPS,
            VIDEO_FRAME_WIDTH,
            0,
            "none",
            None,
            None,
            False,
            gpt_a_stats,
        )
    except Exception as exc:
        gpt_a_error = exc
    if run_stats is not None:
        run_stats["gpt_a"] = gpt_a_stats
    auralis_stats: Dict[str, Any] = {}
    if run_stats is not None:
        run_stats["auralis_audio"] = auralis_stats
        # Retain the old key for existing log consumers.
        run_stats["gemini_audio"] = auralis_stats
    audio_prediction = ""
    auralis_error: Exception | None = None
    try:
        audio_prediction = run_audio_row(
            audio_input,
            api_url=api_url,
            api_key=api_key,
            model=gemini_model,
            timeout=timeout,
            api_retries=api_retries,
            run_stats=auralis_stats,
            auralis_agent=auralis_agent,
            gateway=gateway,
        )
    except Exception as exc:
        auralis_error = exc
    if run_stats is not None:
        run_stats["api_calls"] = int(gpt_a_stats.get("api_calls", 0)) + int(
            auralis_stats.get("api_calls", 0)
        )
        run_stats["request_bytes"] = int(
            gpt_a_stats.get("request_bytes", 0)
        ) + int(auralis_stats.get("request_bytes", 0))
    if gpt_a_error is not None and auralis_error is not None:
        raise RuntimeError(
            f"GPT-A 失败：{gpt_a_error}；Auralis 失败：{auralis_error}"
        ) from gpt_a_error
    if gpt_a_error is not None:
        raise gpt_a_error
    if auralis_error is not None:
        raise auralis_error
    return merge_predictions(gpt_a_prediction, audio_prediction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GPT-D：先运行 GPT-A，再无条件调用 Auralis 音视取证专家。"
        )
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument(
        "--api-url",
        default=os.getenv("VIDEO_EVAL_API_URL", DEFAULT_API_URL),
    )
    parser.add_argument(
        "--gpt-a-model",
        default=os.getenv("VIDEO_EVAL_MODEL", DEFAULT_GPT_A_MODEL),
    )
    parser.add_argument(
        "--gemini-model",
        "--model",
        dest="gemini_model",
        default=os.getenv("VIDEO_EVAL_GPT_D_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument("--max-gpt-a-agent-steps", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--run-log", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _restart_with_cuda_libraries_if_needed() -> None:
    if os.getenv("AURALIS_ASR_DEVICE", "cuda") != "cuda":
        return
    if os.getenv("AURALIS_CUDA_BOOTSTRAPPED") == "1":
        return
    if not cuda_library_dirs():
        return
    os.execve(
        sys.executable,
        [sys.executable, *sys.argv],
        cuda_process_environment(),
    )


def main() -> int:
    _restart_with_cuda_libraries_if_needed()
    load_project_env(BASE_DIR / ".env.local")
    args = parse_args()
    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise ValueError(f"缺少环境变量 {API_KEY_ENV}；请设置 Ark 网关 token。")
    table = gpt_a.read_csv(args.input_csv)
    if not table:
        raise ValueError("gt.csv 为空")
    header, source_rows = table[0], table[1:]
    if header != SOURCE_COLUMNS:
        raise ValueError("gt.csv 列不符合预期；必须严格为：" + ",".join(SOURCE_COLUMNS))
    if any(len(row) != len(SOURCE_COLUMNS) for row in source_rows):
        raise ValueError("gt.csv 存在列数不一致的数据行")
    start_index = max(0, args.start - 1)
    end_index = (
        len(source_rows)
        if args.limit <= 0
        else min(len(source_rows), start_index + args.limit)
    )
    existing = (
        read_matching_predictions(args.output_csv, header, source_rows)
        if args.resume
        else {}
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.run_log is not None:
        args.run_log.parent.mkdir(parents=True, exist_ok=True)
    agent, gateway, asr_backend = build_auralis_agent(
        api_url=args.api_url,
        api_key=api_key,
        gemini_model=args.gemini_model,
        timeout=args.timeout,
        api_retries=args.api_retries,
    )
    failed_rows = 0
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(SOURCE_COLUMNS + [PREDICTION_COLUMN])
        for index, row in enumerate(source_rows, start=1):
            prediction = existing.get(index, "")
            if prediction:
                print(f"[{index:03d}/{len(source_rows)}] skip existing", flush=True)
            elif start_index <= index - 1 < end_index:
                print(
                    f"[{index:03d}/{len(source_rows)}] GPT-A + Auralis",
                    flush=True,
                )
                started = time.monotonic()
                error_text = ""
                run_stats: Dict[str, Any] = {}
                try:
                    prediction = run_combined_row(
                        gpt_a.inference_input(header, row, index),
                        inference_input(header, row, index),
                        api_url=args.api_url,
                        api_key=api_key,
                        gpt_a_model=args.gpt_a_model,
                        gemini_model=args.gemini_model,
                        timeout=args.timeout,
                        api_retries=args.api_retries,
                        max_gpt_a_agent_steps=args.max_gpt_a_agent_steps,
                        run_stats=run_stats,
                        auralis_agent=agent,
                        gateway=gateway,
                    )
                except Exception as exc:
                    failed_rows += 1
                    error_text = str(exc)
                    print(f"  failed: {exc}", flush=True)
                if asr_backend.fallback_reason:
                    run_stats["asr_cuda_fallback_reason"] = (
                        asr_backend.fallback_reason
                    )
                if args.run_log is not None:
                    record = {
                        "row_index": index,
                        "序号": gpt_a.row_value(header, row, "序号"),
                        "profile": "gpt_d_auralis",
                        "gpt_a_model": args.gpt_a_model,
                        "gemini_model": args.gemini_model,
                        "input_mode": "live_gpt_a_then_auralis",
                        "success": prediction != "",
                        "elapsed_sec": round(time.monotonic() - started, 3),
                        "error": error_text,
                        **run_stats,
                    }
                    with args.run_log.open("a", encoding="utf-8") as log_file:
                        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            writer.writerow(
                [row[header.index(name)] for name in SOURCE_COLUMNS] + [prediction]
            )
            file.flush()
    print(f"done: {args.output_csv}; failed_rows={failed_rows}", flush=True)
    return 1 if failed_rows else 0
