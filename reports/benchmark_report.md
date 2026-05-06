# Benchmark Report

_Generated: 2026-05-06 04:44 UTC_

## Per-run results

| Run | Query | Latency (s) | Tokens (in/out) | Cost (USD) | Citation cov. | Iters | Errors | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | Research GraphRAG state-of-the-art and write a 500-word sum… | 8.25 | 1309/914 | 0.000431 | 0.80 | 0 | 0 |  |
| multi-agent | Research GraphRAG state-of-the-art and write a 500-word sum… | 18.56 | 3995/1524 | 0.000809 | 1.00 | 5 | 0 |  |
| baseline | Compare single-agent and multi-agent workflows for customer… | 19.18 | 1258/1073 | 0.000492 | 1.00 | 0 | 0 |  |
| multi-agent | Compare single-agent and multi-agent workflows for customer… | 18.51 | 4079/1608 | 0.000847 | 1.00 | 5 | 0 |  |
| baseline | Summarize production guardrails for LLM agents | 10.04 | 1508/949 | 0.000455 | 1.00 | 0 | 0 |  |
| multi-agent | Summarize production guardrails for LLM agents | 17.80 | 4246/1636 | 0.000867 | 1.00 | 5 | 0 |  |

## Aggregate by run_name

| Run | Avg latency (s) | Total tokens | Total cost (USD) | Avg citation cov. | Errors |
|---|---:|---:|---:|---:|---:|
| baseline | 12.49 | 7011 | 0.001378 | 0.93 | 0 |
| multi-agent | 18.29 | 17088 | 0.002523 | 1.00 | 0 |

## How to read this

- **Latency** is wall-clock time per query. Lower is better, but compare alongside quality.
- **Tokens / Cost** capture LLM expenditure. Multi-agent typically spends more.
- **Citation cov.** = fraction of provided sources that the final answer cites.
- **Iters** counts supervisor decisions; check it stays under the configured `max_iterations`.
- Add a `quality_score` (0–10) from peer review before finalising deliverables.
