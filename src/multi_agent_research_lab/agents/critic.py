"""Critic agent — fact-checks the final answer against sources."""

from __future__ import annotations

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a strict fact-check critic.
Given a final answer and the source list it cites, output exactly two lines:
1. `verdict: pass` if every claim is supported by a cited source, otherwise `verdict: fail`.
2. `issues: <comma-separated short issues>` or `issues: none`.
Then on subsequent lines, list at most 3 specific bullet points starting with `- `."""


class CriticAgent(BaseAgent):
    """Optional fact-checking and citation-coverage agent."""

    name = "critic"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, state: ResearchState) -> ResearchState:
        if not state.final_answer:
            raise AgentExecutionError("Critic requires final_answer to be present.")
        with trace_span("agent.critic", {"answer_chars": len(state.final_answer)}):
            user_prompt = self._render_prompt(state)
            response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=300,
            )
            state.critic_report = response.content
            verdict = _parse_verdict(response.content)
            citation_coverage = _citation_coverage(state.final_answer, len(state.sources))
            state.add_agent_result(
                AgentName.CRITIC,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "model": response.model,
                    "verdict": verdict,
                    "citation_coverage": citation_coverage,
                },
            )
        return state

    @staticmethod
    def _render_prompt(state: ResearchState) -> str:
        sources_block = "\n".join(
            f"[{idx}] {doc.title} — {doc.url or '(no url)'}"
            for idx, doc in enumerate(state.sources, start=1)
        )
        return (
            f"User query: {state.request.query}\n\n"
            f"Final answer:\n{state.final_answer}\n\n"
            f"Sources:\n{sources_block or '(no sources)'}\n\n"
            "Now produce the critique."
        )


def _parse_verdict(content: str) -> str:
    for line in content.splitlines():
        line = line.strip().lower()
        if line.startswith("verdict:"):
            return line.split(":", 1)[1].strip().split()[0] if ":" in line else "unknown"
    return "unknown"


def _citation_coverage(answer: str, source_count: int) -> float:
    """Fraction of cited source indices [n] vs available sources. 0.0 if no sources."""

    if source_count <= 0:
        return 0.0
    cited = {int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer)}
    cited = {n for n in cited if 1 <= n <= source_count}
    return round(len(cited) / source_count, 3)
