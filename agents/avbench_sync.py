"""Process-isolated single-video adapter around AVBench's SyncNet evaluator.

AVAgent's ASR/OCR stack and AVBench intentionally use separate environments:
the former does not need PyTorch, while the latter uses a CUDA PyTorch build.
This adapter keeps a small JSON-lines worker alive so the SyncNet model is
loaded once and reused for every row in a run.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LATENTSYNC_ROOT = str(BASE_DIR / ".external" / "LatentSync")
DEFAULT_AVBENCH_PYTHON = BASE_DIR / ".conda-envs" / "avbench" / "bin" / "python"
WORKER_MODULE = "agents.avbench_worker"


class AVBenchSyncRunner:
    """Evaluate one local video at a time with AVBench SyncNet."""

    def __init__(
        self,
        *,
        latentsync_root: str | Path | None = None,
        syncnet_ckpt: str | Path | None = None,
        python_executable: str | Path | None = None,
        device: str = "cuda",
        batch_size: int = 20,
        vshift: int = 15,
    ) -> None:
        self.latentsync_root = Path(
            latentsync_root
            or os.getenv("LATENTSYNC_ROOT", DEFAULT_LATENTSYNC_ROOT)
        ).expanduser()
        self.syncnet_ckpt = Path(
            syncnet_ckpt
            or os.getenv(
                "AVBENCH_SYNCNET_CKPT",
                str(self.latentsync_root / "checkpoints/auxiliary/syncnet_v2.model"),
            )
        ).expanduser()
        self.sfd_face_ckpt = Path(
            os.getenv(
                "AVBENCH_S3FD_CKPT",
                str(self.latentsync_root / "checkpoints/auxiliary/sfd_face.pth"),
            )
        ).expanduser()
        configured_python = python_executable or os.getenv("AVBENCH_PYTHON", "")
        self.python_executable = Path(
            configured_python or (DEFAULT_AVBENCH_PYTHON if DEFAULT_AVBENCH_PYTHON.is_file() else sys.executable)
        ).expanduser()
        self.device_name = device
        self.batch_size = max(1, int(batch_size))
        self.vshift = max(1, int(vshift))
        self._worker: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _validate_configuration(self) -> None:
        evaluation_dir = BASE_DIR / "third_party" / "AVBench" / "evaluation"
        if not evaluation_dir.is_dir():
            raise FileNotFoundError(f"AVBench evaluation 目录不存在：{evaluation_dir}")
        if not self.latentsync_root.is_dir():
            raise FileNotFoundError(
                f"LatentSync 目录不存在：{self.latentsync_root}；请设置 LATENTSYNC_ROOT。"
            )
        if not self.syncnet_ckpt.is_file():
            raise FileNotFoundError(
                f"SyncNet checkpoint 不存在：{self.syncnet_ckpt}；请设置 AVBENCH_SYNCNET_CKPT。"
            )
        if not self.sfd_face_ckpt.is_file():
            raise FileNotFoundError(
                f"S3FD 人脸检测 checkpoint 不存在：{self.sfd_face_ckpt}；"
                "请设置 AVBENCH_S3FD_CKPT 或补齐 LatentSync 官方 auxiliary/sfd_face.pth。"
            )
        if not self.python_executable.is_file():
            raise FileNotFoundError(
                f"AVBench Python 不存在：{self.python_executable}；请设置 AVBENCH_PYTHON。"
            )

    def _worker_environment(self) -> Dict[str, str]:
        runtime_dir = Path(
            os.getenv("AVBENCH_TMPDIR", str(BASE_DIR / ".tmp" / "avbench_runtime"))
        ).expanduser()
        mpl_dir = Path(
            os.getenv("MPLCONFIGDIR", str(BASE_DIR / ".tmp" / "matplotlib"))
        ).expanduser()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        mpl_dir.mkdir(parents=True, exist_ok=True)

        environment = os.environ.copy()
        environment["TMPDIR"] = str(runtime_dir)
        environment["MPLCONFIGDIR"] = str(mpl_dir)
        environment["PYTHONUNBUFFERED"] = "1"
        path_entries = [
            str(BASE_DIR),
            str(BASE_DIR / "third_party" / "AVBench" / "evaluation"),
            str(self.latentsync_root),
        ]
        existing = environment.get("PYTHONPATH", "")
        if existing:
            path_entries.append(existing)
        environment["PYTHONPATH"] = os.pathsep.join(path_entries)
        return environment

    def _start_worker(self) -> None:
        self._validate_configuration()
        command = [
            str(self.python_executable),
            "-m",
            WORKER_MODULE,
            "--latentsync-root",
            str(self.latentsync_root.resolve()),
            "--syncnet-ckpt",
            str(self.syncnet_ckpt.resolve()),
            "--device",
            self.device_name,
            "--batch-size",
            str(self.batch_size),
            "--vshift",
            str(self.vshift),
        ]
        self._worker = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            env=self._worker_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )

    def _request(self, video_path: Path) -> Dict[str, Any]:
        if self._worker is None or self._worker.poll() is not None:
            self.close()
            self._start_worker()
        assert self._worker is not None
        assert self._worker.stdin is not None
        assert self._worker.stdout is not None
        request = json.dumps(
            {"video_path": str(video_path.resolve())}, ensure_ascii=False
        )
        try:
            self._worker.stdin.write(request + "\n")
            self._worker.stdin.flush()
            response_line = self._worker.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise RuntimeError(f"AVBench worker 通信失败：{exc}") from exc
        if not response_line:
            return_code = self._worker.poll()
            self.close()
            raise RuntimeError(
                f"AVBench worker 意外退出，returncode={return_code}。"
            )
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "AVBench worker 返回了非 JSON 响应：" + response_line[:500]
            ) from exc
        if not isinstance(response, dict):
            raise RuntimeError("AVBench worker 返回值不是 JSON 对象")
        if not response.get("ok"):
            raise RuntimeError(
                "AVBench worker 评估失败："
                + str(response.get("error") or "unknown error")
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("AVBench worker 结果不是 JSON 对象")
        return result

    def evaluate(self, video_path: Path) -> Dict[str, Any]:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"视频不存在：{video_path}")
        with self._lock:
            result = self._request(video_path)
        return {
            "source": "AVBench evaluate_syncnet.py",
            "video_name": video_path.name,
            **result,
            "status": "ok" if bool(result.get("success")) else "failed",
        }

    def close(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        if worker.stdin is not None:
            try:
                worker.stdin.close()
            except OSError:
                pass
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:
                worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
