# Exit ticket

Both answers are grounded in the live benchmark numbers in
[`benchmark_report.md`](benchmark_report.md): 3 queries × {baseline, multi-agent}
on gpt-5-nano + Tavily.

## 1. Case nào NÊN dùng multi-agent?

**Khi tasks có thể tách thành các sub-task có ranh giới rõ và chất lượng /
độ tin cậy quan trọng hơn latency.**

Bằng chứng từ benchmark này:

- **Citation coverage 1.00 vs 0.93** — multi-agent cite hết source, baseline
  drop 1 source ở câu hỏi GraphRAG. Workflow nào cần truy vết nguồn (research
  report, compliance doc, due-diligence) thì 7 điểm % này là khoản đáng trả.
- **Audit trail** — `state.agent_results` ghi rõ từng agent đã làm gì, tốn bao
  nhiêu token. Khi output sai, dễ định vị (researcher fail vs writer ảo) hơn
  là debug 1 LLM call all-in-one.
- **Critic verdict** — multi-agent chạy thêm fact-check, baseline thì không.
  Cùng 1 query nhưng có "verdict: pass" để post-process / gate.

Do đó ưu tiên multi-agent cho: long-form research với citations, regulatory /
compliance Q&A, code-review với rationale tách khỏi summary, hoặc bất cứ workflow
nào sau này muốn HITL từng giai đoạn.

## 2. Case nào KHÔNG nên dùng multi-agent?

**Khi task đơn giản, hoặc latency / chi phí là constraint cứng.**

Bằng chứng từ benchmark này:

- **Latency 1.46×** chậm hơn (avg 18.29s vs 12.49s) — 5 LLM call thay vì 1.
  Hot path serving real-time user (chatbot QA, autocomplete) không chịu nổi.
- **Cost 1.83×** đắt hơn ($0.00252 vs $0.00138 cho 3 query) — input tokens
  tăng 2.4× vì state đi kèm mỗi handoff. Khi quy mô lên triệu query/ngày,
  khoảng cách này nhân tuyến tính.
- **Iter overhead** — supervisor + 4 worker = 5 hop, thêm latency mạng /
  scheduling. Với câu hỏi 1-shot kiểu intent classification hay simple FAQ,
  một LLM call là đủ.

Do đó tránh multi-agent cho: customer-support chatbot intent routing,
realtime autocomplete, batch classification job tiết kiệm token, prototypes
khi chưa rõ task có cần specialization hay không. Theo nguyên tắc Anthropic
"Building effective agents": bắt đầu bằng workflow đơn giản, chỉ chia agent
khi có lý do cụ thể.
