"""Speech/subtitle alignment contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AlignmentIssue:
    issue_type: str
    speech_text: str
    subtitle_text: str
    difference: str
    start_sec: float
    end_sec: float
    confidence: str


@dataclass(frozen=True)
class AlignmentResult:
    issues: Tuple[AlignmentIssue, ...]
