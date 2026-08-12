"""AVBench/LatentSync synchronization evaluation helpers.

The third-party AVBench checkout stays unmodified.  This module owns the
project-specific decision policy and the S3FD face-track adapter so a fresh
clone does not depend on local patches inside the submodule.
"""

from __future__ import annotations

import math
from pathlib import Path
import statistics
from typing import Any


SYNCNET_FPS = 25.0
SYNCNET_AUDIO_SAMPLE_RATE = 16_000
MIN_FACE_TRACK_FRAMES = 50
RELIABLE_CONFIDENCE_THRESHOLD = 3.0
MAX_ALIGNED_OFFSET_FRAMES = 6


def classify_sync_result(
    confidence: float,
    offset_frames: int,
    *,
    vshift: int = 15,
) -> dict[str, Any]:
    """Convert raw SyncNet evidence into a conservative decision."""

    boundary_hit = abs(offset_frames) >= vshift
    confidence_status = (
        "reliable"
        if confidence >= RELIABLE_CONFIDENCE_THRESHOLD
        else "uncertain"
    )
    if boundary_hit:
        decision = "uncertain"
    elif abs(offset_frames) > MAX_ALIGNED_OFFSET_FRAMES:
        decision = "desync_candidate"
    else:
        decision = "aligned_or_no_large_offset"
    return {
        "confidence_status": confidence_status,
        "offset_boundary_hit": boundary_hit,
        "sync_decision": decision,
    }


def compute_sync_score(confidence: float, offset_frames: int) -> float:
    """Return the legacy composite score for comparison, not classification."""

    confidence_score = 1.0 / (1.0 + math.exp(-(confidence - 5.0) * 0.5))
    offset_decay = math.exp(-abs(offset_frames) / 3.0)
    return round(confidence_score * offset_decay * 100.0, 2)


def evaluate_lip_sync(
    video_path: str,
    syncnet_eval: Any,
    temp_dir: str,
    *,
    batch_size: int = 20,
    vshift: int = 15,
    syncnet_detector: Any,
) -> dict[str, Any]:
    """Evaluate one video using official S3FD tracks and SyncNet features."""

    try:
        import cv2

        capture = cv2.VideoCapture(video_path)
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if source_fps <= 0:
            source_fps = SYNCNET_FPS

        track_root = Path(temp_dir) / "detect_results"
        syncnet_detector.detect_results_dir = str(track_root)
        syncnet_detector(
            video_path=video_path,
            min_track=MIN_FACE_TRACK_FRAMES,
        )
        track_paths = sorted((track_root / "crop").glob("*.mp4"))
        if not track_paths:
            raise RuntimeError("S3FD found no face track long enough for SyncNet")

        track_results: list[dict[str, Any]] = []
        for track_index, track_path in enumerate(track_paths):
            result = syncnet_eval.evaluate(
                str(track_path),
                temp_dir=str(Path(temp_dir) / f"track_{track_index}"),
                batch_size=batch_size,
                vshift=vshift,
            )
            if isinstance(result, (tuple, list)) and len(result) >= 4:
                offset, min_distance, confidence, returned_fps = result[:4]
                if returned_fps is not None and float(returned_fps) > 0:
                    source_fps = float(returned_fps)
            else:
                offset, min_distance, confidence = result[:3]

            offset = int(offset)
            confidence = float(confidence)
            decision = classify_sync_result(confidence, offset, vshift=vshift)
            track_results.append(
                {
                    "track_id": f"track_{track_index:05d}",
                    "crop_video": str(track_path),
                    "offset_frames": offset,
                    "offset_sec": offset / SYNCNET_FPS,
                    "confidence": confidence,
                    "min_dist": float(min_distance),
                    **decision,
                }
            )

        offset = int(
            round(statistics.fmean(item["offset_frames"] for item in track_results))
        )
        confidence = float(
            statistics.fmean(item["confidence"] for item in track_results)
        )
        min_distance = float(
            statistics.fmean(item["min_dist"] for item in track_results)
        )
        decision = classify_sync_result(confidence, offset, vshift=vshift)
        if confidence >= 7.0:
            quality = "Excellent"
        elif confidence >= 5.0:
            quality = "Good"
        elif confidence >= 3.0:
            quality = "Fair"
        else:
            quality = "Uncertain"

        return {
            "video_path": video_path,
            "fps": source_fps,
            "offset_frames": offset,
            "offset_sec": offset / SYNCNET_FPS,
            "confidence": confidence,
            "min_dist": min_distance,
            "sync_quality": quality,
            "sync_score": compute_sync_score(confidence, offset),
            "sync_score_calibrated": False,
            **decision,
            "face_detection": "S3FD",
            "face_track_count": len(track_results),
            "processed_fps": SYNCNET_FPS,
            "processed_audio_sample_rate": SYNCNET_AUDIO_SAMPLE_RATE,
            "track_results": track_results,
            "success": True,
            "error": None,
        }
    except Exception as exc:
        return {
            "video_path": video_path,
            "fps": None,
            "offset_frames": None,
            "offset_sec": None,
            "confidence": None,
            "min_dist": None,
            "sync_quality": None,
            "sync_score": None,
            "sync_score_calibrated": False,
            "confidence_status": "not_evaluable",
            "offset_boundary_hit": None,
            "sync_decision": "not_evaluable",
            "face_detection": "S3FD",
            "face_track_count": 0,
            "processed_fps": SYNCNET_FPS,
            "processed_audio_sample_rate": SYNCNET_AUDIO_SAMPLE_RATE,
            "track_results": [],
            "success": False,
            "error": str(exc),
        }
