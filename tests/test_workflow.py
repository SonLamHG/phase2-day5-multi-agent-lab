"""End-to-end workflow test using mock LLM and mock search."""

from __future__ import annotations

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient


def _mock_llm() -> MockLLMClient:
    return MockLLMClient(
        responses={
            "Write the research notes": (
                "Research notes: GraphRAG augments retrieval with knowledge graphs [1]. "
                "Multi-agent systems require justification [2]."
            ),
            "Write the structured analysis": (
                "1. Key claims\n- Knowledge graphs improve retrieval [1].\n"
                "2. Tensions or contradictions\n- None observed.\n"
                "3. Evidence gaps\n- GAP: empirical numbers."
            ),
            "Write the final answer": (
                "Multi-agent systems combine specialised roles to handle research workflows. "
                "GraphRAG [1] uses graphs for retrieval; agent design [2] favours simple workflows.\n\n"
                "## References\n[1] GraphRAG — https://example.com/graphrag-overview\n"
                "[2] Building effective agents — https://example.com/anthropic-agents"
            ),
            "produce the critique": "verdict: pass\nissues: none\n- Citations look consistent.",
        },
        default="(mock fallback)",
    )


def test_multi_agent_workflow_happy_path() -> None:
    workflow = MultiAgentWorkflow(
        llm=_mock_llm(),
        search=MockSearchClient(),
        enable_critic=True,
        max_iterations=6,
    )
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems for research"))
    result = workflow.run(state)
    assert result.final_answer
    assert result.research_notes
    assert result.analysis_notes
    assert result.critic_report
    assert result.sources, "sources should be populated by researcher"
    assert "researcher" in result.route_history
    assert "writer" in result.route_history
    assert any(item.agent == "writer" for item in result.agent_results)


def test_multi_agent_without_critic() -> None:
    workflow = MultiAgentWorkflow(
        llm=_mock_llm(),
        search=MockSearchClient(),
        enable_critic=False,
        max_iterations=6,
    )
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = workflow.run(state)
    assert result.final_answer
    assert result.critic_report is None
    assert "critic" not in result.route_history
