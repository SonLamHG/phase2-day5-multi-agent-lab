"""Shared state for the multi-agent workflow.

Students should extend this file when adding new agents, outputs, or evaluation metrics.
"""

from typing import Any

from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
)


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    request: ResearchQuery
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)

    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None
    critic_report: str | None = None

    agent_results: list[AgentResult] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def record_route(self, route: str) -> None:
        self.route_history.append(route)
        self.iteration += 1

    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        self.trace.append({"name": name, "payload": payload})

    def add_agent_result(
        self,
        agent: AgentName,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        result = AgentResult(agent=agent, content=content, metadata=metadata or {})
        self.agent_results.append(result)
        self.add_trace_event(f"agent.{agent.value}", {"metadata": result.metadata})
        return result

    def total_cost_usd(self) -> float:
        total = 0.0
        for item in self.agent_results:
            cost = item.metadata.get("cost_usd")
            if isinstance(cost, (int, float)):
                total += float(cost)
        return total

    def total_tokens(self) -> tuple[int, int]:
        in_total = 0
        out_total = 0
        for item in self.agent_results:
            in_t = item.metadata.get("input_tokens")
            out_t = item.metadata.get("output_tokens")
            if isinstance(in_t, (int, float)):
                in_total += int(in_t)
            if isinstance(out_t, (int, float)):
                out_total += int(out_t)
        return in_total, out_total
