"""Writer agent — synthesises a final answer with citations."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise technical writer.
Use the research notes and analysis to draft a final answer for the requested audience.
Rules:
- Mirror the user's requested length (default ~500 words; respect explicit length cues).
- Cite sources inline using the same [n] markers from the notes.
- Add a final `## References` section listing each cited source as `[n] Title — URL`.
- Do not invent claims, URLs, or numbers not present in the notes.
- If the notes flag a `GAP:` or `WEAK:` item that is material, surface it in the answer."""


class WriterAgent(BaseAgent):
    """Produces the final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            raise AgentExecutionError("Writer requires research_notes.")
        with trace_span("agent.writer", {"audience": state.request.audience}):
            user_prompt = self._render_prompt(state)
            response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.4,
                max_tokens=900,
            )
            state.final_answer = response.content
            state.add_agent_result(
                AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "model": response.model,
                },
            )
        return state

    @staticmethod
    def _render_prompt(state: ResearchState) -> str:
        sources_block = _format_sources(state.sources)
        analysis_block = state.analysis_notes or "(no analysis available)"
        return (
            f"User query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis:\n{analysis_block}\n\n"
            f"Source list (for the references section):\n{sources_block}\n\n"
            "Write the final answer now."
        )


def _format_sources(sources: list[SourceDocument]) -> str:
    if not sources:
        return "(none)"
    lines: list[str] = []
    for idx, doc in enumerate(sources, start=1):
        url = doc.url or "(no url)"
        lines.append(f"[{idx}] {doc.title} — {url}")
    return "\n".join(lines)
