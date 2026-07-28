"""Local ONNX-based RapidOCR backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..schemas import SubtitleObservation


class RapidOCRBackend:
    name = "rapidocr-onnxruntime"

    def __init__(self) -> None:
        self._engine: Any = None

    def _load_engine(self):
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "缺少 rapidocr；请安装 requirements-audio-agent.txt"
            ) from exc
        self._engine = RapidOCR()
        return self._engine

    @staticmethod
    def _entries(result: Any) -> Iterable[Any]:
        if hasattr(result, "boxes") and hasattr(result, "txts"):
            boxes = getattr(result, "boxes", None)
            texts = getattr(result, "txts", None)
            scores = getattr(result, "scores", None)
            return zip(
                () if boxes is None else boxes,
                () if texts is None else texts,
                () if scores is None else scores,
            )
        if isinstance(result, tuple):
            result = result[0]
        return result or ()

    def recognize(
        self,
        timestamp_sec: float,
        image_path: Path,
    ) -> Iterable[SubtitleObservation]:
        result = self._load_engine()(str(image_path))
        entries = list(self._entries(result))
        if not entries:
            return []
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("RapidOCR 结果归一化需要 Pillow") from exc
        with Image.open(image_path) as image:
            width, height = image.size
        observations = []
        for entry in entries:
            if len(entry) < 3:
                continue
            box, text, score = entry[:3]
            points = list(box)
            if not points:
                continue
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            observations.append(
                SubtitleObservation(
                    timestamp_sec=timestamp_sec,
                    text=str(text).strip(),
                    bbox=(
                        min(xs) / width,
                        min(ys) / height,
                        max(xs) / width,
                        max(ys) / height,
                    ),
                    confidence=float(score),
                )
            )
        return observations
