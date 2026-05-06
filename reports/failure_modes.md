# Failure modes encountered during the lab

Two failure modes hit during the live benchmark run on 2026-05-06 (gpt-5-nano +
Tavily on Windows). Documenting them here per lab deliverable #4.

## 1. Empty completion from a reasoning model

**Symptom.** Baseline first run reported `output_tokens=932` with `content=""`.
Rich rendered an empty `(empty)` panel; the answer never reached the user.

**Root cause.** `gpt-5-nano` is a reasoning model. The full output budget
(`max_completion_tokens`) is shared between hidden reasoning tokens and the
visible answer. With `max_tokens=900` and the default `reasoning_effort`,
the model spent all 900 on internal reasoning and had zero budget left for
the answer string.

**Fix** (`services/llm_client.py`):

- Detect gpt-5 family (`_is_gpt5`) and force `reasoning_effort="minimal"`.
- Inflate the budget: `max_completion_tokens = max(2048, caller_max_tokens * 3)`
  so reasoning + answer both fit.
- Treat empty `content` as a hard failure: raise `AgentExecutionError` with
  `finish_reason` and `reasoning_tokens` in the message instead of silently
  returning `""`. This surfaces the regression next time it happens.
- Tighten retry to only retry on `APIConnectionError`, `APITimeoutError`,
  `RateLimitError`, `InternalServerError` — empty content is NOT a transient
  error, so retrying would burn tokens.

## 2. Windows cp1252 console crash

**Symptom.** Multi-agent run died inside Rich with
`UnicodeEncodeError: 'charmap' codec can't encode character '→'`. The
workflow had finished correctly; only the CLI panel rendering crashed.

**Root cause.** Windows PowerShell defaults stdout to `cp1252`, which cannot
encode the Unicode arrow `→` used in the route history table or the em-dashes
that gpt-5-nano emits in its references list.

**Fix** (`cli.py::_force_utf8_stdout`):

- At CLI init, call `sys.stdout.reconfigure(encoding="utf-8")` and the same
  for stderr. Wrapped in `contextlib.suppress(AttributeError, OSError)` for
  non-reconfigurable streams.
- Replaced the `→` separator in the route history with `->` so the value is
  ASCII-clean even before reconfigure runs.

## 3. Theoretical: supervisor infinite loop

**Mitigations already in code:**
- `SupervisorAgent.decide` returns `Route.DONE` (or a writer fallback) once
  `state.iteration >= max_iterations`.
- LangGraph compile passes `recursion_limit = max_iterations * 2 + 5`.
- The fallback path forces a writer pass with whatever data exists, so the
  user always sees an answer + an entry in `state.errors`.

## 4. Theoretical: Tavily 429 / network failure

**Mitigations already in code:**
- `TavilySearchClient.search` wraps the SDK call and re-raises as
  `AgentExecutionError`.
- `build_default_search_client(allow_mock=True)` falls back to
  `MockSearchClient` for offline / quota-exhausted scenarios.
- LLM-side retry uses tenacity with exponential backoff scoped to transient
  OpenAI errors only.
