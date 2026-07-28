#!/usr/bin/env python3
"""Print or execute the GPT/Seed-Lite × A/B/C experiment matrix."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from av_eval.runner import build_experiment_commands


def parse_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("模型必须写成 label=model_id")
    label, model = value.split("=", 1)
    if not label.strip() or not model.strip():
        raise argparse.ArgumentTypeError("模型 label 和 model_id 均不能为空")
    return label.strip(), model.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="运行消融和模型对照实验")
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("baseline_a", "harness_b", "harness_c"),
        default=("baseline_a", "harness_b", "harness_c"),
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("output/benchmark/runs"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    models = dict(args.model)
    commands = build_experiment_commands(
        python=sys.executable,
        script=BASE_DIR / "call_ffmpeg_skill.py",
        models=models,
        profiles=tuple(args.profiles),
        limit=args.limit,
        start=args.start,
        output_root=args.output_root,
    )
    if not args.execute:
        for command in commands:
            print(shlex.join(command))
        return 0
    if not os.getenv("ARK_API_KEY", "").strip():
        raise SystemExit("缺少 ARK_API_KEY；请只在本机 shell 中设置，不要写入脚本。")

    failures = 0
    for command in commands:
        print(shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=BASE_DIR, check=False)
        failures += int(result.returncode != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
