# Finnie - Command Cheat Sheet

All commands are for **PowerShell**, run from the project root (`ik-multi-agent-capston`).

## 1. Install

```powershell
python -m venv .venv                      # optional but recommended
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# If Activate.ps1 is blocked ("running scripts is disabled"), allow it for this window only:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

pip install pypdf                         # read PDFs placed under src/data/pdfs (already in requirements.txt)
pip install openai                        # only if you use OpenAI   (anthropic is already in requirements)
pip install google-genai                  # only if you use Gemini
```

## 2. Set an API key (optional - the app runs offline without one)

The app does **not** read `.env`; set variables in the terminal you launch from.

```powershell
$env:OPENAI_API_KEY    = "sk-..."         # or:
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:GOOGLE_API_KEY    = "..."
$env:ALPHA_VANTAGE_API_KEY = "..."        # optional secondary market-data source

$env:FINANCE_LLM_PROVIDER = "openai"      # force: anthropic | gemini | openai | offline  (default: auto)
$env:FINANCE_MARKET_PROVIDER = "mock"     # yfinance | alpha_vantage | mock (offline sample prices)

echo $env:OPENAI_API_KEY                  # check it is visible to this terminal
setx OPENAI_API_KEY "sk-..."              # make permanent (open a new terminal afterwards)
```

## 3. Run

```powershell
python -m streamlit run src/web_app/app.py     # http://localhost:8501 ; sidebar shows "AI model: <provider>" or "offline"
python -m src.mcp_server.server                # MCP server for Claude Desktop (stdio)
docker compose up --build                      # optional; not verified in development
```

Quick one-off question without the UI:

```powershell
python -c "from src.workflow.assistant import FinanceAssistant; r=FinanceAssistant().chat('What is an ETF?', session_id='demo'); print(r.agents); print(r.answer); print(r.trace)"
```

## 4. Test and benchmark

```powershell
python -m pytest -q                        # 236 tests, offline (mock data, no keys needed)
python -m pytest --cov=src -q              # with coverage (~95%)
python -m pytest tests/test_guardrails.py -v
python scripts/benchmark.py                # latency benchmarks
```

## 4b. RAG and routing evals

```powershell
python scripts/eval_retrieval.py                       # hit@1, hit@3, MRR, precision@3, no-answer rate, ms/query
python scripts/eval_retrieval.py --misses              # also list the queries that failed
python scripts/eval_retrieval.py --compare             # tfidf vs sentence_transformers vs openai, flat vs hnsw
python scripts/eval_retrieval.py --set pdf_eval.json --misses   # questions answered by the indexed PDF (page-level hits)
python scripts/eval_retrieval.py --no-pdf              # curated articles only (ignore PDFs)
python scripts/eval_retrieval.py --embedder openai --index hnsw --chunking fixed -k 5
python scripts/eval_retrieval.py --min-hit 0.8         # exit 1 below this hit@k (CI gate)
python scripts/eval_routing.py --misses                # router exact match / agent recall / agent precision
```

```powershell
pip install sentence-transformers                      # enables hybrid / neural search (installs PyTorch; model downloads on first use)
python scripts/eval_retrieval.py --embedder hybrid --misses            # default mode: meaning + exact words
python scripts/eval_retrieval.py --embedder hybrid --lexical-weight 0.3 # try another blend (0 = neural only, 1 = TF-IDF only)
python scripts/eval_retrieval.py --embedder hybrid --calibrate         # suggests a score floor (rag.hybrid_min_score)
python scripts/eval_retrieval.py --embedder tfidf                      # keyword-only, no install needed
```

Switch the retrieval back end in `config.yaml` (`rag.embedder`, `rag.index_type`, `rag.chunk_strategy`) or per run with the flags above. Neural embeddings: `pip install sentence-transformers` and set `rag.embedder: sentence_transformers` (or `openai` with `OPENAI_API_KEY`; its token use is logged under caller `embeddings`).

### Add your own PDF

```powershell
mkdir src\data\pdfs\tax                               # sub-folder name = category (tax, bonds, ...); loose files get 'documents'
copy C:\path\to\guide.pdf src\data\pdfs\tax\          # optional: add title/reference in src\data\pdfs\sources.json
python -m streamlit run src/web_app/app.py              # restart; log shows: pdf guide.pdf: N pages -> N documents
```

Extracted text is cached in `.cache/pdf_text/` and neural embeddings in `.cache/embeddings/` (delete `.cache/` to force a rebuild).

## 5. Check logging

Logs are written to `logs/` (git-ignored). Change the folder with `$env:FINANCE_LOG_DIR`; disable the call/token logs with `$env:FINANCE_CALL_LOG = "off"`.

```powershell
Get-Content logs\app.log -Tail 50                    # latest application log lines
Get-Content logs\app.log -Wait                       # follow live
Select-String "failed|rejected by guardrail|skipped" logs\app.log     # problems only
Get-Content logs\agent_calls.jsonl -Tail 10          # latest agent calls
```

| File | What is in it |
|---|---|
| `logs/app.log` | Rotating app log (1 MB x 3), including `agent X failed` stack traces |
| `logs/agent_calls.jsonl` | One row per agent call: agent, ok, duration, error, truncated query |
| `logs/llm_usage.jsonl` | One row per LLM call: caller, provider, model, input/output tokens |

## 6. Check token usage

Only populated when an LLM key is set and used (offline mode uses no tokens).

```powershell
python scripts/token_usage.py                       # totals + averages per caller (agent or "router")
python scripts/token_usage.py --calls               # every call, one per line ("~" = estimated)
python scripts/token_usage.py --since 2026-10-25    # from a date
```

Token-saver settings live under `llm:` in `config.yaml` (`routing_confident_score`, `polish_max_chars`, `polish_cache_ttl_seconds`).

## 7. Find agent failures (including intermittent ones)

```powershell
python scripts/agent_stats.py                       # per agent: calls, failures, fail %, p50/max ms, top error
python scripts/agent_stats.py --failures            # plus each failed call with its error and query
python scripts/agent_stats.py --since 2026-10-25
```

How to read it: a failure rate between 0% and 100% means intermittent. If the failed queries share something (a ticker, a format like "$50k"), it is an input bug. If not, look at the error type - `TimeoutError` / rate-limit errors point to external services.

**Smoke-test all six agents once:**

```powershell
python -c "from src.workflow.assistant import FinanceAssistant as F; a=F(); p={'finance_qa':'What is an ETF?','market_analysis':'How is NVDA doing?','portfolio_analysis':'I own 10 VTI and 5 AAPL','goal_planning':'I want `$50,000 in 8 years, have `$5,000, save `$300/month','news_synthesizer':'Latest news on Tesla','tax_education':'Roth vs traditional IRA'}; [print(n.ljust(20), 'ok' if n in (r:=a.chat(q, session_id=n)).agents and not any('FAILED' in t for t in r.trace) else ('PROBLEM', r.trace)) for n,q in p.items()]"
```

**Repeat one query 20 times to see if a failure is random or input-dependent:**

```powershell
python -c "from src.workflow.assistant import FinanceAssistant as F; a=F(); [print(i, a.chat('How is NVDA doing?', session_id=str(i)).trace[-2]) for i in range(20)]"
python scripts/agent_stats.py --failures
```

**Rule out external services** (if failures stop, the cause is the network or the LLM provider):

```powershell
$env:FINANCE_MARKET_PROVIDER = "mock"; $env:FINANCE_LLM_PROVIDER = "offline"
```

## 8. Regenerate the architecture diagrams

```powershell
npx -y @mermaid-js/mermaid-cli -i docs/diagrams/architecture.mmd -o docs/diagrams/architecture.svg -b white
npx -y @mermaid-js/mermaid-cli -i docs/diagrams/flow.mmd -o docs/diagrams/flow.png -b white -s 2
```

## 9. Git

```powershell
git status
git add -A; git commit -m "message"
git push origin develop
```
