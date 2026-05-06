"""Supervisor / router that decides which worker runs next."""

from __future__ import annotations

import logging
from enum import StrEnum

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class Route(StrEnum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"
    DONE = "done"


class SupervisorAgent(BaseAgent):
    """Rule-based router.

    Decision policy:
    - No sources or research_notes → researcher
    - Research notes present, no analysis_notes → analyst
    - Analysis present, no final_answer → writer
    - Final answer present, no critic_report and critic enabled → critic
    - Otherwise → done
    - On exceeding max_iterations → done with fallback note in errors
    """

    name = "supervisor"

    def __init__(self, max_iterations: int = 6, enable_critic: bool = True) -> None:
        self._max_iterations = max_iterations
        self._enable_critic = enable_critic

    def run(self, state: ResearchState) -> ResearchState:
        decision = self.decide(state)
        state.record_route(decision.value)
        state.add_trace_event("supervisor.route", {"next": decision.value})
        logger.info(
            "supervisor decision iter=%s next=%s history=%s",
            state.iteration,
            decision.value,
            state.route_history,
        )
        return state

    def decide(self, state: ResearchState) -> Route:
        if state.iteration >= self._max_iterations:
            if not state.final_answer and (state.research_notes or state.sources):
                # Force a writer fallback to produce *something* before stopping.
                return Route.WRITER
            state.errors.append(
                f"supervisor: hit max_iterations={self._max_iterations}, stopping."
            )
            return Route.DONE

        if not state.sources or not state.research_notes:
            return Route.RESEARCHER
        if not state.analysis_notes:
            return Route.ANALYST
        if not state.final_answer:
            return Route.WRITER
        if self._enable_critic and not state.critic_report:
            return Route.CRITIC
        return Route.DONE
