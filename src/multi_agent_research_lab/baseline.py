"""Single-agent baseline runner used for benchmarking against multi-agent."""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import (
    LLMClient,
    build_default_llm_client,
)
from multi_agent_research_lab.services.search_client import (
    SearchClient,
    build_default_search_client,
)

logger = logging.getLogger(__name__)

_BASELINE_SYSTEM_PROMPT = """You are an expert research assistant.
Given a query and a numbered list of source snippets, write a complete, well-cited answer.
Cite sources inline with [n] and end with a `## References` section listing each cited source.
Stay grounded in the provided sources; do not invent facts."""


class SingleAgentBaseline:
    """All-in-one baseline: search → 1 LLM call → final answer."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self._llm = llm or build_default_llm_client()
        self._search = search or build_default_search_client(allow_mock=True)

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("baseline.single_agent", {"query": state.request.query}):
            sources = self._search.search(
                state.request.query, max_results=state.request.max_sources
            )
            if not sources:
                raise AgentExecutionError("Baseline could not retrieve any sources.")
            state.sources = sources

            user_prompt = self._render_prompt(state.request.query, sources)
            response = self._llm.complete(
                system_prompt=_BASELINE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=900,
            )
            state.final_answer = response.content
            state.add_agent_result(
                AgentName.WRITER,  # baseline uses writer slot for accounting
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "model": response.model,
                    "mode": "baseline",
                    "source_count": len(sources),
                },
            )
        return state

    @staticmethod
    def _render_prompt(query: str, sources: list[SourceDocument]) -> str:
        lines = [f"Query: {query}", "", "Sources:"]
        for idx, doc in enumerate(sources, start=1):
            lines.append(f"[{idx}] {doc.title}")
            if doc.url:
                lines.append(f"    URL: {doc.url}")
            lines.append(f"    Snippet: {doc.snippet}")
        lines.append("")
        lines.append("Now write the final answer with inline citations and a References section.")
        return "\n".join(lines)
