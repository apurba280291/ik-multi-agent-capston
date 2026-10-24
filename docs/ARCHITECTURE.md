# Architecture and flow

Diagram sources (Mermaid) and rendered images live in [diagrams/](diagrams/). Regenerate with
`npx -y @mermaid-js/mermaid-cli -i diagrams/<name>.mmd -o diagrams/<name>.svg`.

## System architecture

![Architecture](diagrams/architecture.svg)

```mermaid
flowchart TB
    subgraph Clients
        UI["Streamlit UI<br/>chat · portfolio · market · goals · knowledge"]
        MCP["MCP client<br/>Claude Desktop"]
    end

    UI --> FA
    MCP --> MS["mcp_server/server.py"] --> FA

    FA["FinanceAssistant facade<br/>workflow/assistant.py"]

    subgraph LG["LangGraph StateGraph · MemorySaver checkpoint per session"]
        direction LR
        U[understand] --> R[route] --> AG["agent nodes<br/>1–3 in pipeline order"] --> C[compose]
    end
    FA --> LG
    RT["Router<br/>LLM JSON classify or keyword rules"] -.-> R

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
        RAG["Retriever<br/>FAISS + TF-IDF"]
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

    KB[("Knowledge base<br/>81 articles · 48 glossary terms")] --> RAG
    LLM["LLM client (optional)<br/>Anthropic · Gemini · OpenAI<br/>fallback: OfflineLLM"]
    Agents -.-> LLM
    RT -.-> LLM
    CFG["config.yaml + env vars"] -.-> LG
```

## Request flow (one chat turn)

![Flow](diagrams/flow.svg)

```mermaid
flowchart TD
    Q([User message]) --> UN["understand<br/>extract tickers · update profile<br/>reset per-turn fields"]
    UN --> RO["route<br/>choose agents + flags"]
    RO --> D{queue empty?}

    D -- "greeting / off-topic" --> CO
    D -- "no" --> AGN["run first agent in queue<br/>pop itself · write AgentResponse<br/>leave data in shared"]
    AGN --> D2{more agents<br/>in queue?}
    D2 -- yes --> AGN
    D2 -- no --> CO

    CO["compose<br/>merge sections · dedupe citations<br/>advice note · disclaimer · save history"]
    CO --> ANS([Final answer + sources])
```

Multi-intent example: "Analyze my portfolio and plan $100,000 in 10 years" runs the portfolio agent first, which leaves its
equity weight in `shared`; the goal agent reads it, and `compose` merges both sections.

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
