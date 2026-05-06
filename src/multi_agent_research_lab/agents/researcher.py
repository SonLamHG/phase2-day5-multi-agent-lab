"""Researcher agent — collects sources and turns them into citation-rich notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a meticulous research analyst.
Given a user query and numbered source snippets, write concise research notes that:
- Stay under 250 words.
- Capture only facts grounded in the snippets.
- Cite sources inline as [1], [2], ... matching the numbering provided.
- Flag any topic the sources do not cover with the prefix `GAP:`.
Do not invent URLs or facts. If the sources are insufficient, say so explicitly."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm: LLMClient, search: SearchClient) -> None:
        self._llm = llm
        self._search = search

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("agent.researcher", {"query": state.request.query}):
            sources = self._collect_sources(state)
            if not sources:
                raise AgentExecutionError("Researcher could not retrieve any sources.")
            state.sources = sources

            user_prompt = self._render_prompt(state.request.query, sources)
            response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=600,
            )
            state.research_notes = response.content
            state.add_agent_result(
                AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "model": response.model,
                    "source_count": len(sources),
                },
            )
        return state

    def _collect_sources(self, state: ResearchState) -> list[SourceDocument]:
        if state.sources:
            return state.sources
        try:
            return self._search.search(state.request.query, max_results=state.request.max_sources)
        except AgentExecutionError as exc:
            state.errors.append(f"researcher.search: {exc}")
            raise

    @staticmethod
    def _render_prompt(query: str, sources: list[SourceDocument]) -> str:
        lines = [f"Query: {query}", "", "Sources:"]
        for idx, doc in enumerate(sources, start=1):
            lines.append(f"[{idx}] {doc.title}")
            if doc.url:
                lines.append(f"    URL: {doc.url}")
            lines.append(f"    Snippet: {doc.snippet}")
        lines.append("")
        lines.append("Write the research notes now.")
        return "\n".join(lines)
