#!/usr/bin/env python3
"""Compare AVBench SyncNet on an aligned video and a known delayed copy.

The script records raw/per-face evidence and the calibrated decision fields;
it does not alter model weights or thresholds.  Use the resulting JSON when
revisiting the calibration set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from agents.avbench_sync import AVBenchSyncRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-video", required=True)
    parser.add_argument("--delayed-video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latentsync-root", default=None)
    parser.add_argument("--syncnet-ckpt", default=None)
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--vshift", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = AVBenchSyncRunner(
        latentsync_root=args.latentsync_root,
        syncnet_ckpt=args.syncnet_ckpt,
        python_executable=args.python_executable,
        device=args.device,
        batch_size=args.batch_size,
        vshift=args.vshift,
    )
    try:
        results = {
            "normal": runner.evaluate(Path(args.normal_video)),
            "delayed": runner.evaluate(Path(args.delayed_video)),
        }
    finally:
        runner.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"Calibration written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
