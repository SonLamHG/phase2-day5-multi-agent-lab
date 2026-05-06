"""LangGraph workflow wiring supervisor + worker agents."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import Route, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, LabError
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


class MultiAgentWorkflow:
    """Builds and runs the supervisor + workers graph."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
        *,
        enable_critic: bool = True,
        max_iterations: int | None = None,
    ) -> None:
        settings = get_settings()
        self._max_iterations = max_iterations or settings.max_iterations
        self._llm = llm or build_default_llm_client()
        self._search = search or build_default_search_client(allow_mock=True)

        self._supervisor = SupervisorAgent(
            max_iterations=self._max_iterations,
            enable_critic=enable_critic,
        )
        self._researcher = ResearcherAgent(self._llm, self._search)
        self._analyst = AnalystAgent(self._llm)
        self._writer = WriterAgent(self._llm)
        self._critic = CriticAgent(self._llm)
        self._graph = self.build()

    def build(self) -> Any:
        graph: StateGraph = StateGraph(ResearchState)

        graph.add_node("supervisor", self._wrap(self._supervisor.run))
        graph.add_node("researcher", self._wrap(self._researcher.run))
        graph.add_node("analyst", self._wrap(self._analyst.run))
        graph.add_node("writer", self._wrap(self._writer.run))
        graph.add_node("critic", self._wrap(self._critic.run))

        graph.set_entry_point("supervisor")

        graph.add_conditional_edges(
            "supervisor",
            self._next_route,
            {
                Route.RESEARCHER.value: "researcher",
                Route.ANALYST.value: "analyst",
                Route.WRITER.value: "writer",
                Route.CRITIC.value: "critic",
                Route.DONE.value: END,
            },
        )
        for worker in ("researcher", "analyst", "writer", "critic"):
            graph.add_edge(worker, "supervisor")

        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("workflow.run", {"query": state.request.query}):
            try:
                raw = self._graph.invoke(
                    state,
                    config={"recursion_limit": self._max_iterations * 2 + 5},
                )
            except LabError:
                raise
            except Exception as exc:
                logger.exception("Graph invocation failed")
                state.errors.append(f"workflow: {exc}")
                if not state.final_answer:
                    raise AgentExecutionError(f"workflow failed: {exc}") from exc
                return state
        if isinstance(raw, ResearchState):
            return raw
        if isinstance(raw, dict):
            return ResearchState.model_validate(raw)
        raise AgentExecutionError(f"unexpected workflow output type: {type(raw)!r}")

    @staticmethod
    def _wrap(func):
        def runner(state: ResearchState) -> ResearchState:
            return func(state)

        return runner

    @staticmethod
    def _next_route(state: ResearchState) -> str:
        if not state.route_history:
            return Route.RESEARCHER.value
        return state.route_history[-1]
