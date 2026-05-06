"""Benchmark + report rendering tests."""

from __future__ import annotations

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report


def _runner_factory(answer: str):
    def runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        state.final_answer = answer
        return state

    return runner


def test_run_benchmark_captures_latency_and_query() -> None:
    state, metrics = run_benchmark("baseline", "Explain X", _runner_factory("answer"))
    assert metrics.run_name == "baseline"
    assert metrics.query == "Explain X"
    assert metrics.latency_seconds >= 0
    assert state.final_answer == "answer"


def test_report_includes_aggregate_section() -> None:
    metrics = [
        BenchmarkMetrics(run_name="baseline", query="q1", latency_seconds=1.0, input_tokens=10, output_tokens=20),
        BenchmarkMetrics(run_name="multi-agent", query="q1", latency_seconds=3.5, input_tokens=80, output_tokens=120),
    ]
    report = render_markdown_report(metrics)
    assert "Benchmark Report" in report
    assert "Aggregate by run_name" in report
    assert "baseline" in report
    assert "multi-agent" in report
