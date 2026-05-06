"""Tracing hooks.

Two providers are wired here:
- A lightweight in-process JSON span recorder (always on).
- LangSmith via the @traceable decorator (auto-enabled when LANGSMITH_API_KEY is set).

Agents call `trace_span(name, attributes)` as a context manager; the span is captured
in the active recorder so it can be exported to JSON or rendered in a CLI summary.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, time
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    attributes: dict[str, Any]
    started_at: float
    duration_seconds: float | None = None
    status: str = "ok"
    error: str | None = None


@dataclass
class TraceRecorder:
    spans: list[Span] = field(default_factory=list)
    _stack: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def open(self, name: str, attributes: dict[str, Any]) -> Span:
        with self._lock:
            parent_id = self._stack[-1] if self._stack else None
            span = Span(
                span_id=uuid.uuid4().hex,
                parent_id=parent_id,
                name=name,
                attributes=dict(attributes),
                started_at=time(),
            )
            self.spans.append(span)
            self._stack.append(span.span_id)
            return span

    def close(self, span: Span, duration: float, error: str | None) -> None:
        with self._lock:
            span.duration_seconds = duration
            if error:
                span.status = "error"
                span.error = error
            if self._stack and self._stack[-1] == span.span_id:
                self._stack.pop()

    def to_dict(self) -> list[dict[str, Any]]:
        return [span.__dict__ for span in self.spans]


_active: ContextVar[TraceRecorder | None] = ContextVar("trace_recorder", default=None)


@contextmanager
def trace_recorder() -> Iterator[TraceRecorder]:
    recorder = TraceRecorder()
    token = _active.set(recorder)
    try:
        yield recorder
    finally:
        _active.reset(token)


def current_recorder() -> TraceRecorder | None:
    return _active.get()


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Span]:
    recorder = _active.get()
    attrs = attributes or {}
    if recorder is None:
        # No active recorder (e.g. ad-hoc test). Yield a transient span.
        span = Span(
            span_id=uuid.uuid4().hex,
            parent_id=None,
            name=name,
            attributes=attrs,
            started_at=time(),
        )
        started = perf_counter()
        try:
            yield span
        finally:
            span.duration_seconds = perf_counter() - started
        return

    span = recorder.open(name, attrs)
    started = perf_counter()
    err: str | None = None
    try:
        yield span
    except Exception as exc:
        err = repr(exc)
        raise
    finally:
        recorder.close(span, perf_counter() - started, err)


def write_trace_json(recorder: TraceRecorder, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"spans": recorder.to_dict()}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def configure_langsmith() -> bool:
    """Enable LangSmith tracing if a key is configured. Returns True when active."""

    settings = get_settings()
    if not settings.langsmith_api_key or not settings.langsmith_tracing:
        return False
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    logger.info("LangSmith tracing enabled for project=%s", settings.langsmith_project)
    return True
