"""Routing tests for the supervisor agent."""

from __future__ import annotations

from multi_agent_research_lab.agents.supervisor import Route, SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_to_researcher_first() -> None:
    sup = SupervisorAgent(max_iterations=6)
    assert sup.decide(_state()) == Route.RESEARCHER


def test_supervisor_routes_to_analyst_after_research() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    sup = SupervisorAgent(max_iterations=6)
    assert sup.decide(state) == Route.ANALYST


def test_supervisor_routes_to_writer_after_analysis() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    assert SupervisorAgent().decide(state) == Route.WRITER


def test_supervisor_routes_to_critic_after_writer() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    assert SupervisorAgent(enable_critic=True).decide(state) == Route.CRITIC


def test_supervisor_done_when_critic_disabled_and_answer_present() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    assert SupervisorAgent(enable_critic=False).decide(state) == Route.DONE


def test_supervisor_done_after_critic_runs() -> None:
    state = _state()
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "answer"
    state.critic_report = "verdict: pass"
    assert SupervisorAgent(enable_critic=True).decide(state) == Route.DONE


def test_supervisor_fallbacks_to_writer_at_max_iterations() -> None:
    state = _state()
    state.iteration = 6  # at limit
    state.sources = [SourceDocument(title="t", snippet="s")]
    state.research_notes = "notes"
    sup = SupervisorAgent(max_iterations=6)
    assert sup.decide(state) == Route.WRITER  # produce *something* before stopping


def test_supervisor_done_at_limit_with_nothing() -> None:
    state = _state()
    state.iteration = 6
    sup = SupervisorAgent(max_iterations=6)
    assert sup.decide(state) == Route.DONE
    assert any("max_iterations" in err for err in state.errors)
