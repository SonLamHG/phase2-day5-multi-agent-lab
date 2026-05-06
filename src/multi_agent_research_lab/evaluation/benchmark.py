"""Benchmark runner: latency, cost, citation coverage, and error rate."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute a runner once and capture metrics."""

    started = perf_counter()
    state: ResearchState | None = None
    error_msg: str | None = None
    try:
        state = runner(query)
    except LabError as exc:
        logger.warning("benchmark run failed name=%s err=%s", run_name, exc)
        error_msg = str(exc)
    latency = perf_counter() - started

    if state is None:
        return _empty_state(query), BenchmarkMetrics(
            run_name=run_name,
            query=query,
            latency_seconds=latency,
            error_count=1,
            notes=f"failed: {error_msg}" if error_msg else "failed",
        )

    in_tokens, out_tokens = state.total_tokens()
    metrics = BenchmarkMetrics(
        run_name=run_name,
        query=query,
        latency_seconds=latency,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        estimated_cost_usd=state.total_cost_usd() or None,
        citation_coverage=_citation_coverage(state),
        error_count=len(state.errors),
        iterations=state.iteration,
        notes=" | ".join(state.errors) if state.errors else "",
    )
    return state, metrics


def _citation_coverage(state: ResearchState) -> float | None:
    if not state.final_answer or not state.sources:
        return None
    cited = {int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", state.final_answer)}
    cited = {n for n in cited if 1 <= n <= len(state.sources)}
    return round(len(cited) / len(state.sources), 3)


def _empty_state(query: str) -> ResearchState:
    from multi_agent_research_lab.core.schemas import ResearchQuery

    return ResearchState(request=ResearchQuery(query=query))
