"""Command-line entrypoint for the lab."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.baseline import SingleAgentBaseline
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import (
    TraceRecorder,
    configure_langsmith,
    trace_recorder,
    write_trace_json,
)
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _force_utf8_stdout() -> None:
    # Windows console defaults to cp1252 and chokes on the Unicode arrows /
    # em dashes emitted by Rich and the LLM. Force UTF-8 before any Console
    # call. No-op on systems whose streams are already UTF-8.
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(AttributeError, OSError):
                stream.reconfigure(encoding="utf-8")


def _init() -> None:
    _force_utf8_stdout()
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_langsmith()


def _persist_trace(state: ResearchState, recorder: TraceRecorder, run_label: str) -> Path:
    store = LocalArtifactStore()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    rel = f"traces/{run_label}-{timestamp}.json"
    path = store.root / rel
    write_trace_json(recorder, path)
    state.add_trace_event("trace.exported", {"path": str(path)})
    return path


def _summary_table(state: ResearchState, label: str) -> Table:
    in_tok, out_tok = state.total_tokens()
    table = Table(title=f"Run summary — {label}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Iterations", str(state.iteration))
    table.add_row("Route history", " -> ".join(state.route_history) or "(none)")
    table.add_row("Sources", str(len(state.sources)))
    table.add_row("Input tokens", str(in_tok))
    table.add_row("Output tokens", str(out_tok))
    table.add_row("Estimated cost USD", f"{state.total_cost_usd():.6f}")
    table.add_row("Errors", str(len(state.errors)))
    return table


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    audience: Annotated[
        str, typer.Option("--audience", help="Target audience")
    ] = "technical learners",
    max_sources: Annotated[int, typer.Option("--max-sources", help="Top-K sources to fetch")] = 5,
) -> None:
    """Run the single-agent baseline (search + 1 LLM call)."""

    _init()
    request = ResearchQuery(query=query, audience=audience, max_sources=max_sources)
    state = ResearchState(request=request)
    runner = SingleAgentBaseline()
    with trace_recorder() as recorder:
        try:
            state = runner.run(state)
        except LabError as exc:
            console.print(Panel.fit(str(exc), title="Baseline failed", style="red"))
            raise typer.Exit(code=1) from exc
    trace_path = _persist_trace(state, recorder, "baseline")
    console.print(Panel.fit(state.final_answer or "(empty)", title="Single-Agent Baseline"))
    console.print(_summary_table(state, "baseline"))
    console.print(f"[dim]Trace saved to {trace_path}[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    audience: Annotated[
        str, typer.Option("--audience", help="Target audience")
    ] = "technical learners",
    max_sources: Annotated[int, typer.Option("--max-sources", help="Top-K sources to fetch")] = 5,
    enable_critic: Annotated[
        bool, typer.Option("--critic/--no-critic", help="Enable critic agent")
    ] = True,
) -> None:
    """Run the multi-agent workflow (Supervisor + Researcher + Analyst + Writer + Critic)."""

    _init()
    request = ResearchQuery(query=query, audience=audience, max_sources=max_sources)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow(enable_critic=enable_critic)
    with trace_recorder() as recorder:
        try:
            state = workflow.run(state)
        except LabError as exc:
            console.print(Panel.fit(str(exc), title="Multi-agent failed", style="red"))
            raise typer.Exit(code=1) from exc
    trace_path = _persist_trace(state, recorder, "multi-agent")
    console.print(Panel.fit(state.final_answer or "(empty)", title="Multi-Agent Answer"))
    if state.critic_report:
        console.print(Panel.fit(state.critic_report, title="Critic Report", style="cyan"))
    console.print(_summary_table(state, "multi-agent"))
    console.print(f"[dim]Trace saved to {trace_path}[/dim]")


@app.command()
def benchmark(
    queries: Annotated[
        list[str] | None,
        typer.Option("--query", "-q", help="Repeatable; overrides default query set"),
    ] = None,
    audience: Annotated[str, typer.Option("--audience")] = "technical learners",
    max_sources: Annotated[int, typer.Option("--max-sources")] = 5,
    no_critic: Annotated[
        bool, typer.Option("--no-critic", help="Disable critic in multi-agent run")
    ] = False,
    output: Annotated[Path, typer.Option("--output", help="Markdown report path")] = Path(
        "reports/benchmark_report.md"
    ),
) -> None:
    """Run baseline and multi-agent across queries and emit a markdown report."""

    _init()
    default_queries = [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]
    queries = queries or default_queries

    metrics_all = []
    for q in queries:
        for label, runner in (
            ("baseline", _make_baseline_runner(audience, max_sources)),
            ("multi-agent", _make_multi_runner(audience, max_sources, not no_critic)),
        ):
            console.print(f"[bold]Running {label}[/bold]: {q}")
            with trace_recorder() as recorder:
                try:
                    state, metrics = run_benchmark(label, q, runner)
                except LabError as exc:
                    console.print(f"[red]{label} failed:[/red] {exc}")
                    continue
            metrics_all.append(metrics)
            _persist_trace(state, recorder, f"benchmark-{label}")

    report = render_markdown_report(metrics_all)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    console.print(Panel.fit(report, title=f"Benchmark Report — {output}"))


def _make_baseline_runner(
    audience: str, max_sources: int
) -> Callable[[str], ResearchState]:
    runner = SingleAgentBaseline()

    def call(query: str) -> ResearchState:
        state = ResearchState(
            request=ResearchQuery(query=query, audience=audience, max_sources=max_sources)
        )
        return runner.run(state)

    return call


def _make_multi_runner(
    audience: str, max_sources: int, enable_critic: bool
) -> Callable[[str], ResearchState]:
    workflow = MultiAgentWorkflow(enable_critic=enable_critic)

    def call(query: str) -> ResearchState:
        state = ResearchState(
            request=ResearchQuery(query=query, audience=audience, max_sources=max_sources)
        )
        return workflow.run(state)

    return call


if __name__ == "__main__":
    app()
