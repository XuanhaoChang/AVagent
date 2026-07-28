#!/usr/bin/env python3
"""Create a truth-free 30-second capacity probe from the longest pilot video."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from av_eval.data import find_binary, resolve_legacy_media_path


COLUMNS = [
    "序号",
    "user_prompt",
    "reference_image_urls",
    "generated_video_url",
    "用户反馈",
    "思考过程及标准答案",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="准备 30 秒图片容量探针")
    parser.add_argument("--source-row", type=int, default=39)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("output/benchmark/capacity_30s"))
    args = parser.parse_args()
    with Path("input/gt.csv").open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    source = rows[args.source_row - 1]
    video = resolve_legacy_media_path(source["generated_video_url"], Path("input"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_video = (args.output_dir / "synthetic_30s.mp4").resolve()
    subprocess.run(
        [
            find_binary("ffmpeg"),
            "-v",
            "error",
            "-y",
            "-stream_loop",
            "1",
            "-i",
            str(video),
            "-t",
            str(args.duration_sec),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
        check=True,
        timeout=600,
    )
    probe = {
        "序号": "#capacity-30s",
        "user_prompt": source["user_prompt"],
        "reference_image_urls": source["reference_image_urls"],
        "generated_video_url": str(output_video),
        "用户反馈": source["用户反馈"],
        "思考过程及标准答案": "",
    }
    probe_csv = args.output_dir / "probe_gt.csv"
    with probe_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(probe)
    (args.output_dir / "probe_manifest.json").write_text(
        json.dumps(
            {
                "source_row": args.source_row,
                "duration_sec": args.duration_sec,
                "video": str(output_video),
                "input_csv": str(probe_csv.resolve()),
                "contains_gold": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(probe_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
