"""Benchmark report rendering."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Benchmark Report",
        "",
        f"_Generated: {timestamp}_",
        "",
        "## Per-run results",
        "",
        "| Run | Query | Latency (s) | Tokens (in/out) | Cost (USD) | Citation cov. | Iters | Errors | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in metrics:
        cost = "" if m.estimated_cost_usd is None else f"{m.estimated_cost_usd:.6f}"
        cov = "" if m.citation_coverage is None else f"{m.citation_coverage:.2f}"
        lines.append(
            f"| {m.run_name} | {_truncate(m.query, 60)} | {m.latency_seconds:.2f} | "
            f"{m.input_tokens}/{m.output_tokens} | {cost} | {cov} | {m.iterations} | "
            f"{m.error_count} | {_truncate(m.notes, 60)} |"
        )

    lines += ["", "## Aggregate by run_name", "", _aggregate_table(metrics)]
    lines += ["", "## How to read this", "", _reading_guide()]
    return "\n".join(lines) + "\n"


def _aggregate_table(metrics: list[BenchmarkMetrics]) -> str:
    grouped: dict[str, list[BenchmarkMetrics]] = defaultdict(list)
    for m in metrics:
        grouped[m.run_name].append(m)
    out = [
        "| Run | Avg latency (s) | Total tokens | Total cost (USD) | Avg citation cov. | Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, items in grouped.items():
        n = max(1, len(items))
        avg_latency = sum(m.latency_seconds for m in items) / n
        total_tokens = sum(m.input_tokens + m.output_tokens for m in items)
        total_cost = sum((m.estimated_cost_usd or 0.0) for m in items)
        coverages = [m.citation_coverage for m in items if m.citation_coverage is not None]
        avg_cov = sum(coverages) / len(coverages) if coverages else None
        cov_str = "" if avg_cov is None else f"{avg_cov:.2f}"
        errors = sum(m.error_count for m in items)
        out.append(
            f"| {name} | {avg_latency:.2f} | {total_tokens} | {total_cost:.6f} | {cov_str} | {errors} |"
        )
    return "\n".join(out)


def _reading_guide() -> str:
    return (
        "- **Latency** is wall-clock time per query. Lower is better, but compare alongside quality.\n"
        "- **Tokens / Cost** capture LLM expenditure. Multi-agent typically spends more.\n"
        "- **Citation cov.** = fraction of provided sources that the final answer cites.\n"
        "- **Iters** counts supervisor decisions; check it stays under the configured `max_iterations`.\n"
        "- Add a `quality_score` (0–10) from peer review before finalising deliverables."
    )


def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"
