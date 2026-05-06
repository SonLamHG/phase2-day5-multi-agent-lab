# Lab 20: Multi-Agent Research System Starter

Starter repo cho bài lab **Multi-Agent Systems**: xây dựng hệ thống nghiên cứu gồm **Supervisor + Researcher + Analyst + Writer** và benchmark với single-agent baseline.

> Mục tiêu của repo này là cung cấp **production-grade skeleton** để học viên phát triển code cá nhân. Các phần logic quan trọng được để ở dạng `TODO` để học viên tự triển khai.

## Learning outcomes

Sau 2 giờ lab, học viên cần có thể:

1. Thiết kế role rõ ràng cho nhiều agent.
2. Xây dựng shared state đủ thông tin cho handoff.
3. Thêm guardrail tối thiểu: max iterations, timeout, retry/fallback, validation.
4. Trace được luồng chạy và giải thích agent nào làm gì.
5. Benchmark single-agent vs multi-agent theo quality, latency, cost.

## Architecture mục tiêu

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |
   v
Trace + Benchmark Report
```

## Cấu trúc repo

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/              # Agent interfaces + skeletons
│   ├── core/                # Config, state, schemas, errors
│   ├── graph/               # LangGraph workflow skeleton
│   ├── services/            # LLM, search, storage clients
│   ├── evaluation/          # Benchmark/evaluation skeleton
│   ├── observability/       # Logging/tracing hooks
│   └── cli.py               # CLI entrypoint
├── configs/                 # YAML configs for lab variants
├── docs/                    # Lab guide, rubric, design notes
├── tests/                   # Unit tests for skeleton behavior
├── notebooks/               # Optional notebook entrypoint
├── scripts/                 # Helper scripts
├── .env.example             # Environment variables template
├── pyproject.toml           # Python project config
├── Dockerfile               # Containerized dev/runtime
└── Makefile                 # Common commands
```

## Quickstart

### 1. Tạo môi trường

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -e ".[dev,llm]"
cp .env.example .env   # Windows: copy .env.example .env
```

### 2. Cấu hình API keys

Mở `.env` và điền:

```text
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5-nano
TAVILY_API_KEY=tvly-...
LANGSMITH_API_KEY=lsv2_...   # optional but recommended for trace screenshots
LANGSMITH_TRACING=true
```

### 3. Chạy smoke test

```bash
pytest
python -m multi_agent_research_lab.cli --help
```

### 4. Chạy baseline (single-agent)

```bash
python -m multi_agent_research_lab.cli baseline \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Baseline = `Tavily search` → 1 LLM call → answer + references. In ra summary table với
latency, tokens, cost, plus đường dẫn JSON trace.

### 5. Chạy multi-agent

```bash
python -m multi_agent_research_lab.cli multi-agent \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Workflow LangGraph: Supervisor → Researcher → Analyst → Writer → Critic → Done.

Tắt critic nếu muốn nhanh hơn:

```bash
python -m multi_agent_research_lab.cli multi-agent --no-critic --query "..."
```

### 6. Benchmark (single vs multi)

```bash
python -m multi_agent_research_lab.cli benchmark
```

Ghi `reports/benchmark_report.md` và `reports/traces/*.json`.

## Milestones trong 2 giờ lab

| Thời lượng | Milestone | File gợi ý |
|---:|---|---|
| 0-15' | Setup, chạy baseline skeleton | `cli.py`, `services/llm_client.py` |
| 15-45' | Build Supervisor / router | `agents/supervisor.py`, `graph/workflow.py` |
| 45-75' | Thêm Researcher, Analyst, Writer | `agents/*.py`, `core/state.py` |
| 75-95' | Trace + benchmark single vs multi | `observability/tracing.py`, `evaluation/benchmark.py` |
| 95-115' | Peer review theo rubric | `docs/peer_review_rubric.md` |
| 115-120' | Exit ticket | `docs/lab_guide.md` |

## Quy ước production trong repo

- Tách rõ `agents`, `services`, `core`, `graph`, `evaluation`, `observability`.
- Không hard-code API key trong code.
- Tất cả input/output chính dùng Pydantic schema.
- Có type hints, linting, formatting, unit test tối thiểu.
- Có logging/tracing hook ngay từ đầu.
- Không để agent chạy vô hạn: dùng `max_iterations`, `timeout_seconds`.
- Có benchmark report thay vì chỉ demo output đẹp.

## Trạng thái triển khai

| Thành phần | File | Trạng thái |
|---|---|---|
| LLM client (OpenAI gpt-5-nano + retry/cost) | `services/llm_client.py` | ✅ |
| Search client (Tavily + mock) | `services/search_client.py` | ✅ |
| Supervisor routing rule | `agents/supervisor.py` | ✅ |
| Researcher / Analyst / Writer | `agents/*.py` | ✅ |
| Critic (bonus) | `agents/critic.py` | ✅ |
| LangGraph workflow | `graph/workflow.py` | ✅ |
| JSON trace + LangSmith hook | `observability/tracing.py` | ✅ |
| Benchmark + markdown report | `evaluation/{benchmark,report}.py` | ✅ |

## Deliverables

Học viên nộp:

1. GitHub repo cá nhân.
2. Screenshot trace hoặc link trace.
3. `reports/benchmark_report.md` so sánh single vs multi-agent.
4. Một đoạn giải thích failure mode và cách fix.

## References

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK orchestration/handoffs — https://developers.openai.com/api/docs/guides/agents/orchestration
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs
