"""Stable, JSON-serializable contracts for classic issue checks.

The contracts in this module intentionally separate three kinds of state:

* :class:`EvaluationSample` contains only inference-safe sample fields.
* :class:`ToolResult` records evidence returned by one reusable tool.
* :class:`ClassicCheckResult` records one classic check's decision.

No contract has a field for a gold answer, reference answer, or reasoning label.
This makes it difficult for later orchestration code to accidentally pass those
values into an inference tool.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence


TOOL_STATUSES = frozenset({"ok", "not_applicable", "failed"})
CHECK_EXECUTION_STATUSES = TOOL_STATUSES
CHECK_DECISIONS = frozenset({"detected", "not_detected", "not_evaluable"})
EVIDENCE_LEVELS = frozenset({"none", "candidate", "supported", "deterministic"})
CACHE_STATUSES = frozenset({"hit", "miss"})
TRACE_SOURCES = frozenset({"executed", "preloaded", "cache"})
EXPECTED_CLASSIC_CHECK_COUNT = 10


def _normalized_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = str(key)
        if normalized_key in normalized:
            raise ValueError(
                "mapping contains keys that collide after string normalization: "
                f"{normalized_key!r}"
            )
        normalized[normalized_key] = to_jsonable(item)
    return {key: normalized[key] for key in sorted(normalized)}


def to_jsonable(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation of ``value``.

    Lists and tuples intentionally share the same representation so callers can
    use either form without changing cache keys. ``Path`` and string paths also
    converge to the same JSON string. Sets are sorted by their canonical JSON
    encoding. Unsupported objects fail early instead of leaking an unstable
    ``repr`` into an artifact or cache key.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not supported in stable contracts")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, StableSerializable):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return _normalized_mapping(value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [to_jsonable(item) for item in value]
        return sorted(items, key=stable_json_dumps)
    raise TypeError(
        f"unsupported value for stable JSON serialization: {type(value).__name__}"
    )


def stable_json_dumps(value: Any) -> str:
    """Serialize a value with stable key ordering and no non-JSON constants."""

    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class StableSerializable:
    """Mixin providing deterministic dictionary and JSON representations."""

    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):  # pragma: no cover - programming error guard.
            raise TypeError("StableSerializable must be mixed into a dataclass")
        return {
            field.name: to_jsonable(getattr(self, field.name))
            for field in fields(self)
        }

    def to_json(self) -> str:
        return stable_json_dumps(self.to_dict())


def _mapping_field(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return _normalized_mapping(value)


def _issue_tuple(
    values: Sequence[Mapping[str, Any]],
    field_name: str,
) -> tuple[dict[str, Any], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of mappings")
    normalized = []
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name}[{index}] must be a mapping")
        normalized.append(_normalized_mapping(item))
    return tuple(normalized)


def _string_tuple(
    values: Sequence[str],
    field_name: str,
    *,
    deduplicate: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    normalized: list[str] = []
    for index, item in enumerate(values):
        if not isinstance(item, str):
            raise TypeError(f"{field_name}[{index}] must be a string")
        text = item.strip()
        if not text:
            raise ValueError(f"{field_name}[{index}] must not be empty")
        if not deduplicate or text not in normalized:
            normalized.append(text)
    return tuple(normalized)


@dataclass(frozen=True)
class EvaluationSample(StableSerializable):
    """Inference-safe input for all classic checks.

    These are the complete allowed fields. In particular, there is no field for
    ``思考过程及标准答案``, gold issues, or any derived human-review label.
    """

    sample_id: str
    prompt: str
    reference_images: tuple[str, ...]
    video_path: Path
    feedback: str

    ALLOWED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "sample_id",
            "prompt",
            "reference_images",
            "video_path",
            "feedback",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", str(self.sample_id))
        object.__setattr__(self, "prompt", str(self.prompt))
        object.__setattr__(self, "feedback", str(self.feedback))
        object.__setattr__(self, "video_path", Path(self.video_path))
        object.__setattr__(
            self,
            "reference_images",
            _string_tuple(
                self.reference_images,
                "reference_images",
                deduplicate=False,
            ),
        )


@dataclass(frozen=True)
class ToolResult(StableSerializable):
    """Evidence and diagnostics returned by one tool invocation."""

    status: str
    evidence: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    error: str
    usage: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in TOOL_STATUSES:
            raise ValueError(
                f"ToolResult.status must be one of {sorted(TOOL_STATUSES)}"
            )
        object.__setattr__(self, "error", str(self.error or ""))
        for name in ("evidence", "artifacts", "diagnostics", "usage"):
            object.__setattr__(self, name, _mapping_field(getattr(self, name), name))

    @classmethod
    def ok(
        cls,
        *,
        evidence: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            status="ok",
            evidence=evidence or {},
            artifacts=artifacts or {},
            diagnostics=diagnostics or {},
            error="",
            usage=usage or {},
        )

    @classmethod
    def not_applicable(
        cls,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            status="not_applicable",
            evidence={},
            artifacts={},
            diagnostics=diagnostics or {},
            error="",
            usage={},
        )

    @classmethod
    def failed(
        cls,
        error: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            status="failed",
            evidence={},
            artifacts={},
            diagnostics=diagnostics or {},
            error=error,
            usage=usage or {},
        )


@dataclass(frozen=True)
class ClassicCheckResult(StableSerializable):
    """Decision and evidence references for one named classic check."""

    check_name: str
    execution_status: str
    decision: str
    evidence_level: str
    issues: tuple[Mapping[str, Any], ...]
    tool_refs: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        check_name = str(self.check_name).strip()
        if not check_name:
            raise ValueError("ClassicCheckResult.check_name must not be empty")
        if self.execution_status not in CHECK_EXECUTION_STATUSES:
            raise ValueError(
                "ClassicCheckResult.execution_status must be one of "
                f"{sorted(CHECK_EXECUTION_STATUSES)}"
            )
        if self.decision not in CHECK_DECISIONS:
            raise ValueError(
                f"ClassicCheckResult.decision must be one of {sorted(CHECK_DECISIONS)}"
            )
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise ValueError(
                "ClassicCheckResult.evidence_level must be one of "
                f"{sorted(EVIDENCE_LEVELS)}"
            )
        object.__setattr__(self, "check_name", check_name)
        object.__setattr__(self, "issues", _issue_tuple(self.issues, "issues"))
        object.__setattr__(self, "tool_refs", _string_tuple(self.tool_refs, "tool_refs"))
        object.__setattr__(
            self,
            "limitations",
            _string_tuple(self.limitations, "limitations"),
        )


@dataclass(frozen=True)
class ToolTraceEntry(StableSerializable):
    """One auditable cache lookup made through :class:`EvaluationContext`."""

    sequence: int
    tool_name: str
    cache_key: str
    cache_status: str
    parameters: Mapping[str, Any]
    result_status: str
    source: str

    def __post_init__(self) -> None:
        if int(self.sequence) < 1:
            raise ValueError("ToolTraceEntry.sequence must be positive")
        tool_name = str(self.tool_name).strip()
        if not tool_name:
            raise ValueError("ToolTraceEntry.tool_name must not be empty")
        if self.cache_status not in CACHE_STATUSES:
            raise ValueError(
                f"ToolTraceEntry.cache_status must be one of {sorted(CACHE_STATUSES)}"
            )
        if self.result_status not in TOOL_STATUSES:
            raise ValueError(
                f"ToolTraceEntry.result_status must be one of {sorted(TOOL_STATUSES)}"
            )
        if self.source not in TRACE_SOURCES:
            raise ValueError(
                f"ToolTraceEntry.source must be one of {sorted(TRACE_SOURCES)}"
            )
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "cache_key", str(self.cache_key))
        object.__setattr__(
            self,
            "parameters",
            _mapping_field(self.parameters, "parameters"),
        )


@dataclass(frozen=True)
class EvaluationResult(StableSerializable):
    """Unified result containing exactly ten classic check decisions."""

    checks: tuple[ClassicCheckResult, ...]
    final_issues: tuple[Mapping[str, Any], ...]
    tool_trace: tuple[ToolTraceEntry, ...]
    compatibility_log: Mapping[str, Any]

    def __post_init__(self) -> None:
        checks = tuple(self.checks)
        if len(checks) != EXPECTED_CLASSIC_CHECK_COUNT:
            raise ValueError(
                "EvaluationResult.checks must contain exactly "
                f"{EXPECTED_CLASSIC_CHECK_COUNT} entries"
            )
        if not all(isinstance(item, ClassicCheckResult) for item in checks):
            raise TypeError("EvaluationResult.checks must contain ClassicCheckResult")
        check_names = [item.check_name for item in checks]
        if len(set(check_names)) != len(check_names):
            raise ValueError("EvaluationResult.check names must be unique")
        traces = tuple(self.tool_trace)
        if not all(isinstance(item, ToolTraceEntry) for item in traces):
            raise TypeError("EvaluationResult.tool_trace must contain ToolTraceEntry")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(
            self,
            "final_issues",
            _issue_tuple(self.final_issues, "final_issues"),
        )
        object.__setattr__(self, "tool_trace", traces)
        object.__setattr__(
            self,
            "compatibility_log",
            _mapping_field(self.compatibility_log, "compatibility_log"),
        )
