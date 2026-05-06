"""Search client abstraction for ResearcherAgent."""

from __future__ import annotations

import logging
from typing import Protocol

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient(Protocol):
    """Provider-agnostic search client interface."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]: ...


class TavilySearchClient:
    """Tavily-backed search returning a list of SourceDocument."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        if not self._settings.tavily_api_key:
            raise AgentExecutionError(
                "TAVILY_API_KEY is not set. Add it to .env to use TavilySearchClient."
            )
        from tavily import TavilyClient  # type: ignore[import-untyped]

        self._client = TavilyClient(api_key=self._settings.tavily_api_key)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        max_results = max(1, min(max_results, 10))
        try:
            payload = self._client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=False,
            )
        except Exception as exc:  # tavily-python raises ad-hoc errors
            logger.warning("Tavily search failed: %s", exc)
            raise AgentExecutionError(f"Tavily search failed: {exc}") from exc

        results = payload.get("results", []) if isinstance(payload, dict) else []
        documents: list[SourceDocument] = []
        for item in results:
            documents.append(
                SourceDocument(
                    title=str(item.get("title") or "Untitled"),
                    url=item.get("url"),
                    snippet=str(item.get("content") or "")[:1200],
                    metadata={"score": item.get("score")},
                )
            )
        logger.info("tavily_search query=%r results=%d", query, len(documents))
        return documents


class MockSearchClient:
    """Static corpus search used in tests and offline development."""

    _CORPUS: list[SourceDocument] = [
        SourceDocument(
            title="GraphRAG: Unlocking LLM discovery on narrative private data",
            url="https://example.com/graphrag-overview",
            snippet=(
                "GraphRAG combines knowledge graphs with retrieval-augmented generation to "
                "improve question answering over private corpora. It builds entity graphs and "
                "uses community detection for hierarchical summarization."
            ),
        ),
        SourceDocument(
            title="Building effective agents",
            url="https://example.com/anthropic-agents",
            snippet=(
                "Anthropic recommends starting with workflows and only adopting agentic loops "
                "when tasks are open-ended. Multi-agent systems should be justified by clear "
                "specialization, not novelty."
            ),
        ),
        SourceDocument(
            title="LangGraph patterns for multi-agent orchestration",
            url="https://example.com/langgraph-patterns",
            snippet=(
                "Supervisor + workers is a common LangGraph pattern. The supervisor routes by "
                "inspecting shared state. Each worker writes back to a typed schema."
            ),
        ),
        SourceDocument(
            title="Production guardrails for LLM agents",
            url="https://example.com/agent-guardrails",
            snippet=(
                "Effective guardrails include max iterations, timeouts, retries with exponential "
                "backoff, schema validation between agents, and human-in-the-loop for high-risk "
                "actions."
            ),
        ),
        SourceDocument(
            title="Single-agent vs multi-agent for customer support",
            url="https://example.com/single-vs-multi",
            snippet=(
                "Single-agent setups win on latency for simple intents. Multi-agent systems "
                "help when separating retrieval, reasoning, and response writing reduces "
                "cognitive load per call and improves citation discipline."
            ),
        ),
    ]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        terms = {token.lower() for token in query.split() if len(token) > 3}
        scored: list[tuple[int, SourceDocument]] = []
        for doc in self._CORPUS:
            text = (doc.title + " " + doc.snippet).lower()
            score = sum(1 for term in terms if term in text)
            scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [doc for score, doc in scored if score > 0][:max_results]
        return top or self._CORPUS[:max_results]


def build_default_search_client(*, allow_mock: bool = False) -> SearchClient:
    """Return a Tavily client; fall back to mock when explicitly allowed."""

    settings = get_settings()
    if settings.tavily_api_key:
        return TavilySearchClient(settings)
    if allow_mock:
        logger.warning("TAVILY_API_KEY missing — falling back to MockSearchClient.")
        return MockSearchClient()
    raise AgentExecutionError("TAVILY_API_KEY missing. Set it in .env or pass allow_mock=True.")
