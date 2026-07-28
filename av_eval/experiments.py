"""Locked experiment configurations and capacity-test matrix."""

from __future__ import annotations

import math
from typing import Any


def capacity_matrix(duration_sec: float = 30.0) -> dict[str, Any]:
    rates = (0.5, 1.0, 2.0)
    return {
        "image_counts": [8, 16, 32, 48, 60, 80],
        "long_video_duration_sec": duration_sec,
        "long_video_frames": {
            f"{rate:.1f}": int(math.ceil(duration_sec * rate)) for rate in rates
        },
        "quality_checks": [
            "request_success",
            "request_bytes",
            "latency_sec",
            "usage_tokens",
            "image_order_recall",
            "issue_detection_quality",
        ],
        "interpretation": {
            "hard_limit": "最后一个请求成功的图片数",
            "effective_limit": "图片顺序和问题检出质量未出现明显退化的最大图片数",
        },
    }


def experiment_profiles() -> dict[str, dict[str, object]]:
    return {
        "baseline_a": {
            "initial_fps": 2.0,
            "max_initial_frames": 0,
            "dense_sampling": False,
            "local_crop": False,
            "audio_mode": "none",
        },
        "harness_b": {
            "initial_fps": 1.0,
            "max_initial_frames": 48,
            "dense_sampling": True,
            "local_crop": True,
            "audio_mode": "none",
        },
        "harness_c": {
            "initial_fps": 1.0,
            "max_initial_frames": 48,
            "dense_sampling": True,
            "local_crop": True,
            "audio_mode": "direct",
        },
    }
