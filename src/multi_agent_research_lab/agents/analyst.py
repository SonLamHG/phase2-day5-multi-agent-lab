"""Analyst agent — turns research notes into structured insights."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a senior analyst.
Given research notes, produce a short structured analysis with three sections:
1. Key claims — bullet points, each ending with the citation marker(s) from the notes.
2. Tensions or contradictions — bullet points; write `None observed.` if absent.
3. Evidence gaps — bullet points; tag weak evidence with `WEAK:` and missing topics with `GAP:`.
Stay under 200 words. Do not introduce new facts beyond the notes."""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            raise AgentExecutionError("Analyst requires research_notes to be present.")
        with trace_span("agent.analyst", {"notes_chars": len(state.research_notes)}):
            user_prompt = (
                f"User query: {state.request.query}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                "Write the structured analysis now."
            )
            response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=500,
            )
            state.analysis_notes = response.content
            state.add_agent_result(
                AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "model": response.model,
                },
            )
        return state
