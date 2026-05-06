"""Baseline single-agent runner test using mocks."""

from __future__ import annotations

from multi_agent_research_lab.baseline import SingleAgentBaseline
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient


def test_baseline_writes_final_answer() -> None:
    llm = MockLLMClient(default="Final answer with [1] citation.\n\n## References\n[1] X")
    runner = SingleAgentBaseline(llm=llm, search=MockSearchClient())
    state = ResearchState(request=ResearchQuery(query="What is GraphRAG and why care"))
    result = runner.run(state)
    assert result.final_answer
    assert "[1]" in result.final_answer
    assert result.sources
    assert any(item.metadata.get("mode") == "baseline" for item in result.agent_results)
