# Implementation Flow

How Finnie was built, in order, and how a request flows through the finished system. Milestone numbers refer to `AI_Finance_Assistant_Project_Milestones.md`.

## Part 1 - Build order

| Step | Milestone | What was built | Key files | Verified by |
|---|---|---|---|---|
| 1 | 1. Research & architecture | Project skeleton from the prescribed structure; config (`config.yaml` + env overrides); logging, TTL cache, retry/backoff, rate limiter, formatting + disclaimer | `config.yaml`, `src/core/config.py`, `src/utils/*` | `tests/test_utils.py` |
| 2 | 2. Knowledge base | 81 articles in 11 category files (`## Title` sections with `category:` / `reference:` headers), 48-term glossary, asset metadata, sample portfolios, mock prices, sample news | `src/data/knowledge_base/*.md`, `glossary.json`, `asset_metadata.json`, `sample_portfolios.json`, `mock_market.json`, `sample_news.json` | `tests/test_rag.py` (size ≥ 50, categories) |
| 3 | 6. RAG system | Loader + chunker, TF-IDF embeddings, FAISS index (NumPy fallback), retriever with relevance threshold, category filter, one-chunk-per-article dedupe, source attribution | `src/rag/documents.py`, `vector_store.py`, `retriever.py` | `tests/test_rag.py` |
| 4 | 8. Real-time data | Provider chain (yfinance, Alpha Vantage, mock), cache, retries, rate-limit handling, stale/mock fallback with freshness flags, trend metrics, news | `src/data/market_data.py` | `tests/test_market_data.py` (fault injection, no network) |
| 5 | 3. Core agents (foundations) | Pure calculation modules, written before the agents that use them: finance math (future value, required saving, Monte Carlo, diversification), portfolio analysis, goal planner | `src/core/finance_math.py`, `portfolio_analysis.py`, `goal_planner.py` | `tests/test_finance_math.py`, `tests/test_portfolio_analysis.py` |
| 6 | 3. Core agents | `BaseAgent` (error containment, LLM polish), provider-agnostic LLM clients with offline fallback, shared models, text parsing (tickers, holdings, money, years, risk) | `src/core/base_agent.py`, `llm.py`, `models.py`, `src/utils/parsing.py` | `tests/test_llm.py`, `tests/test_parsing.py` |
| 7 | 3-4. Agents | Finance Q&A, Portfolio, Market, Goal Planning, News Synthesizer, Tax Education | `src/agents/*.py` | `tests/test_agents.py` |
| 8 | 5. Workflow | Router (LLM + rules), LangGraph `StateGraph`, checkpointer memory, `FinanceAssistant` facade | `src/workflow/router.py`, `graph.py`, `assistant.py` | `tests/test_router.py`, `tests/test_workflow.py` |
| 9 | 7. UI | Streamlit tabs and Plotly charts (colorblind-safe palette, single y-axis) | `src/web_app/app.py`, `charts.py` | `tests/test_web_app.py` (`streamlit.testing`) |
| 10 | 11. MCP (stretch) | Six tools over stdio; supports `mcp` 1.x and 2.x | `src/mcp_server/server.py` | `tests/test_mcp_server.py`, stdio `initialize` handshake |
| 11 | 9-10. QA & docs | Coverage config, benchmark script, Dockerfile/compose, README, design doc | `.coveragerc`, `scripts/benchmark.py`, `Dockerfile`, `README.md`, `docs/DESIGN.md` | full suite: 172 tests, ~96% coverage |

Principle used throughout: **compute facts deterministically first, let the LLM only rephrase them.** That is why the calculations and retrieval (steps 3-5) were built and tested before any LLM or UI code.

## Part 2 - Runtime flow of one chat turn

```
Streamlit chat_input / MCP tool call
        │  FinanceAssistant.chat(query, session_id, profile?, portfolio?)
        ▼
 ┌─ LangGraph (state checkpointed per session_id) ───────────────────────────┐
 │ 1 understand                                                              │
 │     reset per-turn fields · extract tickers → last_tickers (memory)       │
 │     update profile (risk tolerance, age) from phrases in the message      │
 │ 2 route                                                                   │
 │     greeting? → intro   |   no finance words & not a follow-up? → off-topic│
 │     else: LLM JSON classification (validated) → fallback to keyword rules  │
 │     flags "should I buy…?" as advice request (adds market_analysis)       │
 │     result: ordered queue of 1-3 agents                                   │
 │ 3 agent nodes (loop via conditional edge on queue[0])                     │
 │     market → news → portfolio → goal → tax → Q&A (pipeline order)         │
 │     each: build AgentContext → agent.handle() → append AgentResponse      │
 │           → pop queue → write `shared` data for later agents              │
 │ 4 compose                                                                 │
 │     section headings if >1 agent · consolidate/renumber sources           │
 │     advice note if flagged · disclaimer · store holdings · trim history   │
 └───────────────────────────────────────────────────────────────────────────┘
        ▼
 ChatResult(answer, agents, sources, trace, route_method)
```

### Inside a typical agent

```
Finance Q&A:    query ─► (expand short follow-up with previous question) ─► Retriever.search
                  ─► draft = top article + related snippets + [n] Sources ─► polish() ─► response
Portfolio:      holdings (from message or saved portfolio) ─► MarketDataService.get_quotes
                  ─► analyze_portfolio() ─► draft table + metrics + observations ─► polish()
Market:         tickers (message or remembered) ─► quote + trend_metrics ─► freshness label ─► polish()
                  no ticker ─► index overview table
Goal Planning:  parse_goal() (target, years, current, monthly, risk; profile fallback)
                  missing target/years ─► ask the user
                  else plan_goal() (scenarios + Monte Carlo) ─► draft ─► polish()
News:           get_news (provider chain, sample fallback) ─► keyword sentiment ─► theme → RAG background
Tax:            Q&A restricted to tax/retirement/glossary + account comparison table + "limits change yearly"
```

`polish()` sends the draft to the LLM only if one is configured; any LLM error returns the draft unchanged.

## Part 3 - Failure handling flow

| Failure | What happens |
|---|---|
| yfinance error / timeout | retry with exponential backoff (3 tries) → Alpha Vantage (if keyed) → stale cache (`is_stale`) → mock (`is_mock`); UI and answers show the freshness label |
| Alpha Vantage quota ("Note") | raised as `RateLimitError`, **not retried**, falls through the chain |
| Invalid ticker text | `ValueError` caught per symbol; other symbols still processed |
| Nothing relevant in the knowledge base | Q&A says it couldn't find a verified article rather than improvising |
| LLM unavailable / errors / returns garbage routing | deterministic draft / keyword routing is used |
| An agent raises | `BaseAgent.handle` returns a fallback message and trace entry; remaining agents still run |
| Graph raises | `chat()` returns a friendly fallback with `ok=False` |

## Part 4 - Where to change things

| To… | Edit |
|---|---|
| Add knowledge | append a `## Title` section to a file in `src/data/knowledge_base/` (or add a new category file); the index rebuilds at startup |
| Add a glossary term | `src/data/glossary.json` |
| Teach the app a new ticker's class/sector/fees | `src/data/asset_metadata.json` |
| Add an agent | subclass `BaseAgent`, register it in `FinanceAssistant.__init__`, add routing rules and a slot in `AGENT_ORDER` in `router.py`, add a title in `SECTION_TITLES` in `graph.py` |
| Change return/volatility assumptions | `RISK_ASSUMPTIONS` in `src/core/finance_math.py` |
| Swap the LLM or market provider | `config.yaml` (`llm.provider`, `market.provider`) |
| Swap embeddings | `Retriever._build` in `src/rag/retriever.py` |

## Part 5 - Commands

```bash
pip install -r requirements.txt
streamlit run src/web_app/app.py              # app
python -m pytest --cov                        # 172 tests, ~96% coverage
python scripts/benchmark.py                   # latency numbers
python -m src.mcp_server.server               # MCP server (stdio)
```
