# Finnie - AI Finance Assistant

A multi-agent conversational assistant that teaches beginners about investing. Six specialised agents are orchestrated by a **LangGraph** workflow, grounded by a **FAISS** knowledge base (81 articles + 48 glossary terms, with source attribution), and connected to **live market data** (yfinance / Alpha Vantage) with caching, retries and graceful fallbacks. A **Streamlit** app provides chat, portfolio, market, goal and knowledge tabs, and an **MCP server** exposes the same tools to Claude Desktop.

> Educational tool only. It does not give personalised financial, tax or legal advice, and every answer carries a disclaimer.

**It runs with zero API keys** - without an LLM key the agents return their deterministic, fully grounded answers; with a key (Anthropic, Gemini or OpenAI) the same facts are rewritten conversationally.

## Quick start

```bash
pip install -r requirements.txt
streamlit run src/web_app/app.py          # http://localhost:8501
```

Optional: `cp .env.example .env` and set `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` or `OPENAI_API_KEY` (and `ALPHA_VANTAGE_API_KEY`). Install the matching SDK (`pip install google-genai` / `openai`; `anthropic` is in requirements).

Docker: `docker compose up --build` (not exercised in the development environment - see *Known limitations*).

## Architecture

```
                         ┌────────────── LangGraph StateGraph (checkpointed per session) ──────────────┐
 User ─► Streamlit ─►    │ understand ─► route ─► [agent ─► agent ─► ...] ─► compose ─► final answer  │
        / MCP client     └───────────────────────────────┬──────────────────────────────────────────────┘
                                                         │ (1-3 agents, run in pipeline order, sharing data)
   ┌─────────────────┬───────────────────┬───────────────┼───────────────┬───────────────────┬──────────────┐
   │ Market Analysis │ News Synthesizer  │ Portfolio     │ Goal Planning │ Tax Education     │ Finance Q&A  │
   │ quotes, trends  │ headlines + tone  │ metrics, risk │ projections,  │ accounts, rules   │ RAG answers  │
   │                 │ + RAG background  │ stress test   │ Monte Carlo   │ (RAG, tax filter) │ + citations  │
   └───────┬─────────┴─────────┬─────────┴───────┬───────┴───────┬───────┴─────────┬─────────┴──────┬───────┘
           ▼                   ▼                 ▼               ▼                 ▼                ▼
     MarketDataService    Retriever (FAISS + TF-IDF)     finance_math / portfolio_analysis / goal_planner (pure functions)
     cache → yfinance → Alpha Vantage → stale cache → mock
```

| Graph node | Responsibility |
|---|---|
| `understand` | Resets per-turn fields; extracts tickers (remembered for follow-ups) and risk/age into the user profile |
| `route` | Chooses 1-3 agents: LLM classification (validated JSON) when available, otherwise weighted keyword rules; detects greetings, off-topic and "should I buy…?" advice requests |
| agent nodes | Each agent pops itself off the `queue`, writes its `AgentResponse`, and may leave data in `shared` for later agents (e.g. portfolio analysis → goal plan) |
| `compose` | Merges sections, consolidates citations, adds the advice note and disclaimer, trims and stores history |

Details and design rationale: [docs/DESIGN.md](docs/DESIGN.md).

### Project layout

```
src/
  agents/      six agents (finance_qa, portfolio_agent, market_agent, goal_agent, news_agent, tax_education)
  core/        BaseAgent, LLM clients, models, config, finance_math, portfolio_analysis, goal_planner
  data/        market_data.py (providers + cache + fallbacks), assets.py, knowledge_base/*.md, glossary.json, samples
  rag/         documents (load/chunk), retriever (TF-IDF embeddings), vector_store (FAISS / NumPy)
  workflow/    router.py, graph.py (LangGraph), assistant.py (facade)
  web_app/     app.py (Streamlit), charts.py (Plotly)
  mcp_server/  server.py (Claude Desktop integration)
  utils/       cache, retry/backoff, rate limiter, parsing, formatting, logging
tests/         172 tests    scripts/benchmark.py    config.yaml    Dockerfile
```

## What each agent does

| Agent | Example | Output |
|---|---|---|
| Finance Q&A | "What is an ETF?" | Cited answer from the knowledge base; short follow-ups ("why is that?") reuse the previous question for retrieval |
| Portfolio Analysis | "I own 10 VTI, 20 AAPL and $2,000 in BND" | Value, allocation (class / sector / holding), diversification score, risk rating, bear-market stress test, expense ratio, plain-English observations |
| Market Analysis | "How is NVDA doing?" / "market today" | Quote, 1m/6m return, SMA20/50 trend, volatility, drawdown, data-freshness label; index snapshot when no ticker |
| Goal Planning | "$50,000 in 8 years, have $5,000, save $300/month, moderate" | Expected value, required monthly saving across scenarios, Monte Carlo odds, illustrative stock/bond/cash mix tied to risk and horizon |
| News Synthesizer | "Latest news on Tesla" | Headline digest, tone, and linked educational background (e.g. rates, inflation) with citations |
| Tax Education | "Roth vs 401(k)" | Account comparison table + cited explanation; reminds that limits change yearly |

Multi-intent messages run several agents in order, e.g. *"Analyze my portfolio and plan $100,000 in 10 years"* produces a portfolio section, then a goal section that references the portfolio's equity weight.

## Configuration

`config.yaml` (override the path with `FINANCE_CONFIG`):

| Key | Meaning |
|---|---|
| `llm.provider` | `auto` (first provider with an API key, else offline), `anthropic`, `gemini`, `openai`, `offline` |
| `llm.*_model`, `temperature`, `max_tokens`, `requests_per_minute` | Model choice and client-side rate limiting |
| `market.provider` | `yfinance`, `alpha_vantage`, `mock` |
| `market.cache_ttl_seconds` / `retries` / `backoff_base_seconds` | 30-minute TTL cache, exponential backoff |
| `market.indices` | Tickers shown in the market overview |
| `rag.*` | Knowledge-base paths, chunk size/overlap, `top_k`, `min_score` |

Environment overrides: `FINANCE_LLM_PROVIDER`, `FINANCE_MARKET_PROVIDER`, `FINANCE_CONFIG`.

## Using it

**Python API** (what the UI and MCP server use):

```python
from src.workflow.assistant import FinanceAssistant

bot = FinanceAssistant()
r = bot.chat("I own 10 VTI and 5 BND. Analyze my portfolio", session_id="alice",
             profile={"risk_tolerance": "moderate", "age": 30})
print(r.answer)        # markdown with disclaimer
print(r.agents)        # ['portfolio_analysis']
print(r.sources)       # citations, if any
print(r.trace)         # routing / agent trace
bot.get_state("alice") # profile, portfolio, history, remembered tickers
```

`chat()` never raises: workflow failures return a friendly fallback with `ok=False`; a failing agent is contained and reported in the answer while the other agents still respond.

**Pure-function building blocks** (no I/O, easy to reuse): `src.core.portfolio_analysis.analyze_portfolio`, `src.core.goal_planner.plan_goal`, `src.core.finance_math.*`, `Retriever.search(query, k, category, min_score)`, `MarketDataService.get_quote / get_history / analyze_trend / get_news / market_overview`.

### Streamlit tabs

Chat (example prompts, "How I answered" trace) · Portfolio (editable table, sample portfolios, donut/bar charts, observations) · Market (index overview, ticker lookup with moving averages, news) · Goals (sliders, projected-balance fan chart, yearly table) · Knowledge (category-filtered search with sources).

### MCP server for Claude Desktop (optional)

Tools: `ask_finance_assistant`, `search_knowledge_base`, `get_stock_quote`, `analyze_portfolio_holdings`, `plan_financial_goal`, `explain_term`. Add to `claude_desktop_config.json` (use absolute paths):

```json
{
  "mcpServers": {
    "finnie": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "C:\\path\\to\\ik-multi-agent-capston"
    }
  }
}
```

Works with `mcp` 1.x (`FastMCP`) and 2.x (`MCPServer`). The server is stdio-based; tools return text/JSON and include the disclaimer where numbers are produced. The tool registration and tool functions are tested, but I have not run it inside Claude Desktop itself.

## Testing

```bash
python -m pytest --cov            # 172 tests, ~96% line coverage (offline: mock market data, no API keys)
python scripts/benchmark.py       # latency benchmarks
```

Tests cover each agent, routing (rules, LLM path, follow-ups, off-topic, advice), the graph (multi-agent hand-off, session isolation, history trimming, failure containment), RAG relevance and filters, market-data fallbacks (retry/backoff, rate limits, stale cache, mock), finance math, parsing edge cases, the Streamlit app (`streamlit.testing`), and the MCP tools. The Anthropic / Gemini / OpenAI client classes themselves are not exercised (they need keys); the shared retry/error wrapper is.

### Benchmarks (this machine: offline LLM, mock market data)

| Measure | Result |
|---|---|
| Knowledge-base build (129 entries, FAISS) | ~30 ms |
| Retrieval | ~0.5 ms median |
| Goal plan + 2,000-path Monte Carlo | ~1 ms |
| End-to-end turn (route → agent → compose) | 5-10 ms median |
| Live yfinance quote (first call) | ~7 s here; repeat calls served from cache in <1 ms |

With an LLM enabled, each answer adds one provider round-trip (and one more for LLM routing).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Sample data (live feed unavailable)" banners | yfinance failed or is rate-limited; the app is using clearly labelled sample prices. Retry later, or set `ALPHA_VANTAGE_API_KEY` as a secondary source |
| "Cached data (may be outdated)" | Live fetch failed; showing the last good value past its 30-min TTL |
| Sidebar says "AI model: offline" | No API key found. Set one in `.env`/environment and install its SDK |
| Headlines are labelled "sample" | yfinance returned no news (it sometimes does); the fallback is demo headlines |
| `No module named 'mcp.server.fastmcp'` | Use the updated server (handles mcp 2.x) or `pip install "mcp<2"` |
| Ticker not recognised in chat | Chat recognises tickers in `src/data/asset_metadata.json`, `$CASHTAGS`, and common company names; add your own to that file, or use `$XYZ` |
| Alpha Vantage returns a "Note" | Free-tier quota reached; rate-limit errors are not retried and the system falls through to the next source |

## Known limitations

- **Embeddings are TF-IDF**, not neural. They are fast, key-free and good on this vocabulary, but they miss paraphrases with no word overlap. The `Retriever` is the only place to swap in a neural embedder.
- **Knowledge-base content was written for this project**, not scraped from the cited sites; `reference` links point to the relevant authoritative site section for further reading. Tax limits quoted are labelled as tax-year 2025 and should be verified.
- **Projections** use fixed illustrative return/volatility assumptions by risk level (4.5-8% / 6-16%); they are not forecasts. The stress-test shocks and asset-class risk scores are rough teaching heuristics.
- **Asset classification** covers ~40 common tickers; unknown tickers are treated as individual stocks in an "Unclassified" sector (flagged).
- **Docker** files are provided but were not built in the development environment. Streamlit session memory is in-process (no persistence across restarts, no authentication).
- The optional LLM layer was not run against a live provider during development (no keys available); it is covered through a fake client.
