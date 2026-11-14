# Architecture and flow

Diagram sources (Mermaid) and rendered images live in [diagrams/](diagrams/). Regenerate with
`npx -y @mermaid-js/mermaid-cli -i diagrams/<name>.mmd -o diagrams/<name>.svg`.

## System architecture

![Architecture](diagrams/architecture.svg)

```mermaid
flowchart TB
    subgraph Clients
        UI["Streamlit UI<br/>login · chat · portfolio · market · goals · knowledge"]
        MCP["MCP client<br/>Claude Desktop"]
    end

    UI --> AUTH["Auth<br/>scrypt passwords · lockout · session timeout"] --> FA
    MCP --> MS["mcp_server/server.py"] --> FA

    FA["FinanceAssistant facade<br/>workflow/assistant.py"]

    subgraph LG["LangGraph StateGraph · MemorySaver checkpoint per session"]
        direction LR
        U["understand<br/>PII redaction · injection block"] --> R["route<br/>rules · off-topic · advice"] --> O["orchestrate<br/>LLM plans agents + order"] --> AG["agent nodes<br/>run the plan"] --> C[compose]
        AG -.->|"failed / needs input"| RV["review<br/>LLM keeps, changes or ends plan"]
        RV -.-> AG
    end
    FA --> LG
    ORCH["Orchestrator<br/>validated JSON · caps · dependency order<br/>falls back to the rules plan"] -.-> O
    ORCH -.-> RV

    subgraph Agents
        A1["Market Analysis"]
        A2["News Synthesizer"]
        A3["Portfolio Analysis"]
        A4["Goal Planning"]
        A5["Tax Education"]
        A6["Finance Q&A"]
    end
    AG --- Agents

    subgraph Services
        MD["MarketDataService<br/>cache → yfinance → Alpha Vantage<br/>→ stale cache → mock"]
        RAG["Hybrid retriever<br/>neural meaning + TF-IDF exact words<br/>FAISS flat / HNSW"]
        FM["Pure functions<br/>finance_math · portfolio_analysis · goal_planner"]
    end
    A1 --> MD
    A2 --> MD
    A2 --> RAG
    A3 --> FM
    A3 --> MD
    A4 --> FM
    A5 --> RAG
    A6 --> RAG

    KB[("Knowledge base<br/>81 articles · 48 glossary terms · PDFs by page")] --> RAG
    LLM["LLM client (optional)<br/>Anthropic · Gemini · OpenAI<br/>fallback: OfflineLLM"]
    Agents -.->|"rewrite grounded drafts"| LLM
    ORCH -.->|"plan / review"| LLM
    CFG["config.yaml + env vars"] -.-> LG
```

## Request flow (one chat turn)

![Flow](diagrams/flow.svg)

```mermaid
flowchart TD
    Q([User message]) --> UN["understand<br/>redact PII · flag prompt injection<br/>extract tickers · update profile"]
    UN --> RO["route<br/>deterministic: greeting · off-topic · advice<br/>keyword rules propose agents"]
    RO --> D0{"agents proposed?"}
    D0 -- "no: greeting / off-topic / blocked<br/>(LLM never consulted)" --> CO
    D0 -- yes --> OR["orchestrate<br/>LLM chooses agents + order<br/>(validated, capped, dependency order)"]
    OR -. "LLM off / invalid reply" .-> RULES["use the keyword-rule plan"]
    OR --> NX
    RULES --> NX{"next in queue?"}
    NX -- yes --> AGN["run agent<br/>pop itself · write AgentResponse<br/>leave data in shared"]
    AGN --> RQ{"failed, or asked<br/>the user for input?"}
    RQ -- "no" --> NX
    RQ -- "yes (within re-plan budget)" --> RVW["review<br/>LLM: continue, add an agent, or finish"]
    RVW --> NX
    NX -- "queue empty" --> CO

    CO["compose<br/>merge sections · dedupe citations<br/>advice note · disclaimer · save history"]
    CO --> ANS([Final answer + sources])
```

### The orchestrator

`orchestrate` and `review` are the LLM-driven part ([src/workflow/orchestrator.py](../src/workflow/orchestrator.py)). The LLM controls
**control flow only** - which agents run, in what order, and whether to carry on after a problem. It never writes the answer: agents
still compute grounded facts and the LLM only optionally rewrites them.

| Step | Trigger | LLM call | If the LLM is off or its reply is unusable |
|---|---|---|---|
| `orchestrate` | every message the deterministic checks let through | 1 (plan) | the keyword-rule plan runs unchanged |
| `review` | an agent failed, or asked the user for input (or every agent with `review_every_step`) | 1 per review, at most `max_replans` (2) | the remaining plan continues unchanged |

Safeguards: replies must be JSON naming known agents; repeats and unknown names are dropped; at most `max_agents` (3) agents are
planned, plus one substitute after a failure; `portfolio_analysis` always runs before `goal_planning`; the user's message is passed
as quoted JSON data; and greetings, off-topic messages, advice flags, PII redaction and prompt-injection blocking happen
deterministically *before* the LLM is consulted. Set `orchestrator.mode: rules` for the previous keyword-rules-only behaviour.

## Inside one agent

![Agent](diagrams/agent.svg)

```mermaid
flowchart LR
    IN["query + AgentContext<br/>profile · portfolio · history · shared"] --> G["Gather facts<br/>RAG / market data / math"]
    G --> T["Deterministic grounded answer"]
    T --> L{LLM key set?}
    L -- yes --> RW["LLM rewrites conversationally"]
    L -- no --> OUT
    RW --> OUT["AgentResponse<br/>content · sources · data"]
```
