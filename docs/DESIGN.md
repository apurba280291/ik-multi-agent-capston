# Technical Design

## 1. Goals and constraints

Build an educational finance assistant that is **grounded** (answers come from curated content or computed facts), **transparent** (citations, freshness labels), **safe** (no personalised advice, disclaimers), and **resilient** (every external dependency has a fallback). It must be runnable by an evaluator without any API key.

## 2. System architecture decisions

| Decision | Choice | Why |
|---|---|---|
| Orchestration | LangGraph `StateGraph` + `MemorySaver` checkpointer | Explicit routing and state, per-session memory keyed by `thread_id`, matches the project brief |
| Agent roles | 6 agents with one `BaseAgent` contract (`run`, `handle`, `polish`) | Separation of concerns; `handle` converts any exception into a contained fallback response |
| LLM usage | *Facts first, LLM second*: agents compute a deterministic draft; the LLM only rewrites it ("use ONLY facts in the draft") | Prevents invented numbers/sources, makes tests deterministic, and lets the product run offline. Failure of the LLM never fails the turn |
| LLM providers | Anthropic / Gemini / OpenAI behind one interface; `auto` picks by available key; client-side rate limiter + retry | Brief allows any; avoids vendor lock-in |
| Vector DB | FAISS `IndexFlatIP` (NumPy fallback if FAISS is missing) | Exact search is instantaneous at ~130 entries; simplest correct choice |
| Embeddings | TF-IDF (1-2-grams, sublinear tf, title boosted) | No key or model download; title boosting + glossary entries make short definitional queries work. Trade-off documented in README |
| UI | Streamlit multi-tab | Per brief; session state keeps the conversation and portfolio |
| Calculations | Pure functions in `core/` | The agents, UI and MCP server share exactly one implementation |

## 3. Workflow and agent communication protocol

State (`FinanceState`, a `TypedDict`): `query`, `profile`, `portfolio`, `history`, `last_tickers` (persisted across turns by the checkpointer) and per-turn `intents`, `queue`, `responses`, `shared`, `flags`, `trace`, `final_answer`, `sources`.

Per turn:

1. **understand** - resets per-turn fields, extracts tickers (kept in `last_tickers` so "what about its news?" works) and updates `profile` from phrases like "I'm an aggressive investor, 28".
2. **route** - `Router.route()`:
   * greeting → intro; no finance vocabulary and not a short follow-up → polite off-topic reply;
   * otherwise weighted regex rules score each agent (≥3 selects it; at most 3 agents; "explain/what is" adds Q&A to a specific intent);
   * with an LLM, a JSON list is requested and **validated against the known agent names**; any parse/validation/API failure falls back to the rules, which are always the safety net;
   * short follow-ups ("why is that?") inherit the previous turn's route;
   * "should I buy X?" is flagged as an advice request and `market_analysis` is added so the user gets facts instead of a recommendation.
3. **agents** - a conditional edge sends control to `queue[0]`; each agent node builds an `AgentContext` (profile, portfolio, history, tickers, `shared`), calls `agent.handle()`, appends its `AgentResponse`, pops the queue, and writes `shared` back. The fixed order `market → news → portfolio → goal → tax → Q&A` guarantees producers run before consumers.
4. **compose** - merges sections (headed when >1 agent), de-duplicates and renumbers sources into one block, appends the advice note if flagged, always appends the disclaimer, persists any holdings the user mentioned, and trims history to `app.max_history_turns`.

**Inter-agent protocol.** `AgentResponse{agent, content, sources[], data{}, ok, error}` flows to the composer; `AgentContext.shared` is a per-turn blackboard. Current producers/consumers: `portfolio_analysis` writes `portfolio_analysis`, which `goal_planning` reads to compare the user's equity weight with the suggested mix; `market_analysis` writes `tickers`/`market_data`. Agents never call each other directly, so any agent can fail or be skipped without breaking the others.

**Error handling layers.** Provider retry/backoff → provider fallback → stale cache → mock data (market); LLM failure → deterministic draft; agent exception → contained fallback message + trace entry; graph exception → `FALLBACK` message from `chat()`; invalid tickers → `ValueError` caught per symbol.

## 4. RAG implementation

* **Corpus**: 81 articles in 11 category files (`basics, stocks, bonds, funds, portfolio, risk, retirement, tax, planning, markets, behavior`) + 48 glossary terms (category `glossary`) = 129 entries. Files use a small header (`category:`, `reference:`) and `## Title` sections.
* **Chunking**: word-based windows (140 words, 30 overlap); nearly all entries fit in one chunk, so each chunk is a whole idea with its title.
* **Indexing**: `TfidfVectorizer(stop_words="english", ngram_range=(1,2), sublinear_tf=True)` over `"{title}. {title}. {text}"`; dense L2-normalised vectors in a FAISS inner-product index (cosine).
* **Retrieval**: rank all chunks, drop those below `min_score`, de-duplicate to one chunk per article, apply optional category filter (single or list), return `RetrievedChunk` with score. The tax agent restricts to `tax/retirement/glossary`.
* **Attribution**: every hit becomes a `Source(title, category, reference, score)`; answers carry numbered `[n]` markers and a Sources block; multi-agent answers get one consolidated block. If nothing clears the threshold the agent says so instead of improvising.

## 5. Market data

`MarketDataService._fetch`: fresh cache (30 min TTL) → primary provider with exponential backoff (rate-limit errors are *not* retried) → secondary provider → stale cache (`is_stale=True`) → mock (`is_mock=True`). Every `Quote`/`PriceHistory` has `source`, `as_of` and flags, rendered as freshness captions/warnings in the UI and answers, so fallback data is never presented as live. Ticker input is validated against a strict pattern. News uses the same provider chain, with labelled sample headlines as the last resort.

## 6. Financial methods (all illustrative)

* **Future value / required saving**: monthly compounding annuity formulas.
* **Monte Carlo**: yearly returns ~ Normal(μ, σ) per risk level (conservative 4.5%/6%, moderate 6.5%/11%, aggressive 8%/16%), 2,000 paths, 10th/50th/90th percentiles, probability of reaching the target. Seeded for reproducibility.
* **Suggested mix**: stock share by risk level, capped at 20% (<3 yrs) / 40% (<5 yrs) and raised for ≥20 yrs.
* **Portfolio**: value & weights; allocation by class and sector; weighted expense ratio and annual fee estimate; diversification score = 70% effective-number-of-holdings (1/HHI, full marks at 10) + 30% asset-class breadth (full at 4); risk score = weight-averaged class risk (0-10); stress test = weight-averaged bear-market shocks; observations for concentration, sector tilt, missing bonds/international, crypto share, fees, and gap versus the illustrative mix.

## 7. Compliance and trust

Disclaimer on every analytic answer; advice-request guardrail; conservative language ("many educators suggest…"); citations; freshness labels; tax-limit caveats; no recommendations to buy/sell specific securities.

## 8. Performance and scalability

At this scale latency is dominated by network calls (market data, optional LLM), not by our code: see the benchmark table in the README. The retriever and cache are in-process and thread-safe; the Streamlit app caches the assistant with `st.cache_resource`. To scale out: move the checkpointer to a persistent store (Redis/Postgres), the cache to Redis, serve the graph behind an API, and swap the TF-IDF index for a hosted vector DB.

## 9. Testing strategy

Unit tests for pure logic (math, parsing, portfolio, retrieval); behavioural tests for routing and agents with a mock market provider and an injectable `FakeLLM`; fault-injection tests for the market chain (fail N times, rate-limit, stale cache, total outage); end-to-end graph tests (multi-agent hand-off, memory, isolation, malformed input); UI tests via `streamlit.testing`; MCP tool tests. All offline and deterministic.
