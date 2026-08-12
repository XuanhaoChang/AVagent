"""Long-lived AVBench SyncNet worker used by :mod:`agents.avbench_sync`."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parents[1]


def _load_evaluator(args: argparse.Namespace):
    evaluation_dir = BASE_DIR / "third_party" / "AVBench" / "evaluation"
    latentsync_root = Path(args.latentsync_root).resolve()
    # LatentSync's official S3FD loader resolves checkpoints/auxiliary/
    # sfd_face.pth relative to the LatentSync repository.
    os.chdir(latentsync_root)
    sys.path.insert(0, str(evaluation_dir))
    sys.path.insert(0, str(latentsync_root))
    from av_eval.syncnet import evaluate_lip_sync
    from eval.syncnet.syncnet_eval import SyncNetEval
    from eval.syncnet_detect import SyncNetDetector
    import torch

    requested_device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    syncnet_eval = SyncNetEval(device=requested_device)
    checkpoint = torch.load(args.syncnet_ckpt, map_location=requested_device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    syncnet_eval.__S__.load_state_dict(state_dict, strict=False)
    syncnet_eval.__S__.eval()
    syncnet_detector = SyncNetDetector(
        device=requested_device,
        detect_results_dir=str(BASE_DIR / ".tmp" / "avbench_detector"),
    )
    return evaluate_lip_sync, syncnet_eval, syncnet_detector


def _evaluate(
    evaluate_lip_sync: Any,
    syncnet_eval: Any,
    syncnet_detector: Any,
    video_path: Path,
    *,
    batch_size: int,
    vshift: int,
) -> Dict[str, Any]:
    temp_root = os.getenv("TMPDIR") or None
    temp_dir = Path(tempfile.mkdtemp(prefix="avbench_syncnet_", dir=temp_root))
    try:
        return dict(
            evaluate_lip_sync(
                str(video_path),
                syncnet_eval,
                str(temp_dir),
                batch_size=batch_size,
                vshift=vshift,
                syncnet_detector=syncnet_detector,
            )
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latentsync-root", required=True)
    parser.add_argument("--syncnet-ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--vshift", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        # Keep imports and any third-party diagnostics off the JSON protocol.
        with contextlib.redirect_stdout(sys.stderr):
            evaluate_lip_sync, syncnet_eval, syncnet_detector = _load_evaluator(args)
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc), "traceback": traceback.format_exc()},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            video_path = Path(request["video_path"])
            with contextlib.redirect_stdout(sys.stderr):
                result = _evaluate(
                    evaluate_lip_sync,
                    syncnet_eval,
                    syncnet_detector,
                    video_path,
                    batch_size=max(1, args.batch_size),
                    vshift=max(1, args.vshift),
                )
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"ok": False, "error": str(exc), "traceback": traceback.format_exc()},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
