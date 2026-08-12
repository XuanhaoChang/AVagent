"""Lazy, auditable tool execution context for classic issue checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from threading import RLock
from typing import Any

from .contracts import (
    EvaluationSample,
    ToolResult,
    ToolTraceEntry,
    stable_json_dumps,
    to_jsonable,
)


ToolCallable = Callable[..., ToolResult]


@dataclass(frozen=True)
class _CacheEntry:
    result: ToolResult
    source: str


class EvaluationContext:
    """Run registered tools once per normalized invocation.

    Cache identity is the tool name plus the canonical JSON representation of
    keyword parameters. Parameter mapping order, list-versus-tuple choices, and
    ``Path``-versus-string path values therefore do not cause duplicate work.

    Tool callables receive only the explicit keyword parameters supplied to
    :meth:`run_tool`; the sample remains available as :attr:`sample`. This keeps
    dependencies explicit and avoids silently widening a tool's input contract.
    """

    def __init__(
        self,
        sample: EvaluationSample,
        tools: Mapping[str, ToolCallable] | None = None,
    ) -> None:
        if not isinstance(sample, EvaluationSample):
            raise TypeError("sample must be an EvaluationSample")
        self.sample = sample
        self._tools: dict[str, ToolCallable] = {}
        self._cache: dict[str, _CacheEntry] = {}
        self._trace: list[ToolTraceEntry] = []
        self._stats: dict[str, dict[str, int]] = {}
        self._lock = RLock()
        for tool_name, tool in (tools or {}).items():
            self.register_tool(tool_name, tool)

    @staticmethod
    def _tool_name(value: str) -> str:
        name = str(value).strip()
        if not name:
            raise ValueError("tool_name must not be empty")
        return name

    @staticmethod
    def normalized_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        normalized = to_jsonable(parameters)
        assert isinstance(normalized, dict)
        return normalized

    @classmethod
    def make_cache_key(
        cls,
        tool_name: str,
        parameters: Mapping[str, Any],
    ) -> str:
        name = cls._tool_name(tool_name)
        normalized = cls.normalized_parameters(parameters)
        digest = hashlib.sha256(
            stable_json_dumps(normalized).encode("utf-8")
        ).hexdigest()
        return f"{name}:{digest}"

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {
            "requests": 0,
            "executions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "preloaded": 0,
            "ok_returns": 0,
            "not_applicable_returns": 0,
            "failed_returns": 0,
        }

    def _stats_for(self, tool_name: str) -> dict[str, int]:
        return self._stats.setdefault(tool_name, self._empty_stats())

    def register_tool(
        self,
        tool_name: str,
        tool: ToolCallable,
        *,
        replace: bool = False,
    ) -> None:
        """Register a keyword-callable tool under a stable name."""

        name = self._tool_name(tool_name)
        if not callable(tool):
            raise TypeError(f"tool {name!r} must be callable")
        with self._lock:
            if name in self._tools and not replace:
                raise ValueError(f"tool already registered: {name}")
            self._tools[name] = tool

    def preload_tool_result(
        self,
        tool_name: str,
        result: ToolResult,
        **parameters: Any,
    ) -> str:
        """Seed the cache with evidence produced by the existing pipeline.

        The returned cache key can be persisted in a compatibility log. A later
        matching :meth:`run_tool` is recorded as a cache hit and does not require
        the corresponding tool to be registered.
        """

        name = self._tool_name(tool_name)
        if not isinstance(result, ToolResult):
            raise TypeError("result must be a ToolResult")
        key = self.make_cache_key(name, parameters)
        with self._lock:
            self._cache[key] = _CacheEntry(result=result, source="preloaded")
            self._stats_for(name)["preloaded"] += 1
        return key

    def _record_trace(
        self,
        *,
        tool_name: str,
        cache_key: str,
        cache_status: str,
        parameters: Mapping[str, Any],
        result: ToolResult,
        source: str,
    ) -> None:
        self._trace.append(
            ToolTraceEntry(
                sequence=len(self._trace) + 1,
                tool_name=tool_name,
                cache_key=cache_key,
                cache_status=cache_status,
                parameters=parameters,
                result_status=result.status,
                source=source,
            )
        )

    def _record_return(self, stats: dict[str, int], result: ToolResult) -> None:
        stats[f"{result.status}_returns"] += 1

    def run_tool(self, tool_name: str, **parameters: Any) -> ToolResult:
        """Return a cached result or lazily invoke ``tool(**parameters)`` once.

        Tool exceptions and contract violations are converted into a cached
        ``failed`` result. This lets every classic check report ``not_evaluable``
        without losing the underlying failure in the tool trace.
        """

        name = self._tool_name(tool_name)
        normalized = self.normalized_parameters(parameters)
        key = self.make_cache_key(name, normalized)
        with self._lock:
            stats = self._stats_for(name)
            stats["requests"] += 1
            cached = self._cache.get(key)
            if cached is not None:
                stats["cache_hits"] += 1
                self._record_return(stats, cached.result)
                self._record_trace(
                    tool_name=name,
                    cache_key=key,
                    cache_status="hit",
                    parameters=normalized,
                    result=cached.result,
                    source=(
                        "preloaded" if cached.source == "preloaded" else "cache"
                    ),
                )
                return cached.result

            stats["cache_misses"] += 1
            tool = self._tools.get(name)
            if tool is None:
                result = ToolResult.failed(
                    f"tool is not registered: {name}",
                    diagnostics={"reason": "tool_not_registered"},
                )
            else:
                stats["executions"] += 1
                try:
                    raw_result = tool(**parameters)
                    if not isinstance(raw_result, ToolResult):
                        raise TypeError(
                            f"tool {name!r} returned {type(raw_result).__name__}; "
                            "expected ToolResult"
                        )
                    result = raw_result
                except Exception as exc:
                    result = ToolResult.failed(
                        f"{type(exc).__name__}: {exc}",
                        diagnostics={
                            "reason": "tool_execution_exception",
                            "exception_type": type(exc).__name__,
                        },
                    )
            self._cache[key] = _CacheEntry(result=result, source="executed")
            self._record_return(stats, result)
            self._record_trace(
                tool_name=name,
                cache_key=key,
                cache_status="miss",
                parameters=normalized,
                result=result,
                source="executed",
            )
            return result

    @property
    def tool_trace(self) -> tuple[ToolTraceEntry, ...]:
        with self._lock:
            return tuple(self._trace)

    @property
    def call_stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {
                name: dict(self._stats[name])
                for name in sorted(self._stats)
            }

    @property
    def cached_result_count(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def registered_tools(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._tools))
