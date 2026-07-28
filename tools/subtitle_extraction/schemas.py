"""Subtitle evidence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class SubtitleObservation:
    timestamp_sec: float
    text: str
    bbox: BBox
    confidence: float


@dataclass(frozen=True)
class SubtitleSegment:
    start_sec: float
    end_sec: float
    text: str
    bbox: BBox
    confidence: float
    source: str = "burned_in"


@dataclass(frozen=True)
class SubtitleTrack:
    segments: Tuple[SubtitleSegment, ...]
    backend: str
