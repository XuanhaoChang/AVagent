#!/usr/bin/env python3
"""Print or execute the required 8–80 image capacity probes."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from av_eval.runner import build_capacity_commands


def main() -> int:
    parser = argparse.ArgumentParser(description="运行图片数量容量实验")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-index", type=int, default=39)
    parser.add_argument("--input-csv", type=Path, default=Path("input/gt.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("output/benchmark/capacity"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    counts = (8, 16, 32, 48, 60, 80)
    commands = build_capacity_commands(
        python=sys.executable,
        script=BASE_DIR / "run_visual_baseline.py",
        model=args.model,
        sample_index=args.sample_index,
        image_counts=counts,
        output_root=args.output_root,
        input_csv=args.input_csv,
    )
    if not args.execute:
        for command in commands:
            print(shlex.join(command))
        return 0
    if not (
        os.getenv("AVAGENT_API_KEY", "").strip()
        or os.getenv("ARK_API_KEY", "").strip()
    ):
        raise SystemExit("缺少 AVAGENT_API_KEY；请只在本机环境中设置。")

    results = []
    for count, command in zip(counts, commands):
        result = subprocess.run(command, cwd=BASE_DIR, check=False)
        run_log = args.output_root / f"images_{count:03d}" / "run.jsonl"
        record = {}
        if run_log.is_file():
            lines = [line for line in run_log.read_text(encoding="utf-8").splitlines() if line]
            if lines:
                record = json.loads(lines[-1])
        results.append(
            {
                "image_count": count,
                "process_exit_code": result.returncode,
                **record,
            }
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "capacity_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return int(any(item["process_exit_code"] != 0 for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
