"""Streamlit UI: Chat, Portfolio, Market, Goals and Knowledge tabs.

Run:  streamlit run src/web_app/app.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.core.goal_planner import plan_goal  # noqa: E402
from src.core.portfolio_analysis import analyze_portfolio  # noqa: E402
from src.data.assets import load_sample_portfolios  # noqa: E402
from src.data.market_data import MarketDataService  # noqa: E402
from src.utils.formatting import DISCLAIMER, money  # noqa: E402
from src.web_app import charts  # noqa: E402
from src.workflow.assistant import FinanceAssistant  # noqa: E402

st.set_page_config(page_title="Finnie - AI Finance Assistant", page_icon="💬", layout="wide")

EXAMPLES = [
    "What is an ETF and why do people like index funds?",
    "How is NVDA doing today?",
    "I own 10 VTI, 20 AAPL and $2,000 in BND. Analyze my portfolio",
    "I want $50,000 in 8 years. I have $5,000 and can save $300 a month. I'm a moderate investor.",
    "Roth vs traditional IRA - which is better?",
    "Latest news on Tesla and what it means",
]


@st.cache_resource(show_spinner="Loading knowledge base and agents...")
def get_assistant() -> FinanceAssistant:
    return FinanceAssistant()


def init_state() -> None:
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("portfolio", [])
    st.session_state.setdefault("risk", "moderate")
    st.session_state.setdefault("age", 30)


def freshness_caption(item) -> None:
    text = MarketDataService.freshness(item)
    (st.warning if item.is_mock or item.is_stale else st.caption)(text)


def sidebar(bot: FinanceAssistant) -> None:
    with st.sidebar:
        st.title("Finnie")
        st.caption("Your beginner-friendly finance guide")
        st.selectbox("Your risk tolerance", ["conservative", "moderate", "aggressive"], key="risk")
        st.number_input("Your age", 16, 90, key="age")
        st.divider()
        llm = bot.llm
        st.markdown(f"**AI model:** {'`' + llm.name + '`' if llm.available else 'offline (grounded templates)'}")
        st.markdown(f"**Market data:** `{bot.market.primary.name if bot.market.primary else 'sample data'}`")
        st.markdown(f"**Knowledge base:** {bot.retriever.document_count} entries, `{bot.retriever.backend}` index")
        if st.button("New conversation"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption(DISCLAIMER)


def chat_tab(bot: FinanceAssistant) -> None:
    st.subheader("Ask me anything about investing")
    if not st.session_state.messages:
        st.write("Try one of these:")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex{i}", width="stretch"):
                st.session_state.pending = ex
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                with st.expander("How I answered"):
                    st.write(f"**Agents:** {', '.join(m['meta']['agents']) or 'none'}  \n**Routing:** {m['meta']['route']}")
                    st.code("\n".join(m["meta"]["trace"]), language="text")

    prompt = st.chat_input("Ask a question, describe your portfolio, or share a goal...") or st.session_state.pop("pending", None)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = bot.chat(
                    prompt,
                    st.session_state.session_id,
                    profile={"risk_tolerance": st.session_state.risk, "age": st.session_state.age},
                    portfolio=st.session_state.portfolio or None,
                )
            st.markdown(res.answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": res.answer, "meta": {"agents": res.agents, "route": res.route_method, "trace": res.trace}}
        )
        # Portfolio mentioned in chat becomes available to the Portfolio tab.
        held = bot.get_state(st.session_state.session_id).get("portfolio")
        if held:
            st.session_state.portfolio = held
        st.rerun()


def portfolio_tab(bot: FinanceAssistant) -> None:
    st.subheader("Portfolio analysis")
    samples = load_sample_portfolios()
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Start from a sample portfolio", ["(keep mine)"] + list(samples))
    if c2.button("Load sample", disabled=choice == "(keep mine)"):
        st.session_state.portfolio = samples[choice]
        st.rerun()

    df = pd.DataFrame(
        [{"symbol": h["symbol"], "shares": h.get("shares"), "value ($)": h.get("value")} for h in st.session_state.portfolio]
        or [{"symbol": "", "shares": None, "value ($)": None}]
    )
    edited = st.data_editor(
        df, num_rows="dynamic", width="stretch", key="editor",
        column_config={
            "symbol": st.column_config.TextColumn("Ticker", help="e.g. VTI, AAPL, BND"),
            "shares": st.column_config.NumberColumn("Shares", min_value=0.0),
            "value ($)": st.column_config.NumberColumn("Or dollar value", min_value=0.0, help="Use instead of shares"),
        },
    )
    holdings = []
    for _, r in edited.iterrows():
        sym = str(r["symbol"] or "").strip().upper()
        if not sym:
            continue
        h = {"symbol": sym}
        if pd.notna(r["shares"]):
            h["shares"] = float(r["shares"])
        if pd.notna(r["value ($)"]):
            h["value"] = float(r["value ($)"])
        if "shares" in h or "value" in h:
            holdings.append(h)
    st.session_state.portfolio = holdings
    if not holdings:
        st.info("Add holdings above (or load a sample) to see allocation, risk and diversification.")
        return

    quotes = bot.market.get_quotes([h["symbol"] for h in holdings if "shares" in h])
    a = analyze_portfolio(holdings, {s: q.price for s, q in quotes.items()}, st.session_state.risk)
    if a["empty"]:
        st.error("None of those holdings could be priced - check the ticker symbols.")
        return
    if any(q.is_mock or q.is_stale for q in quotes.values()):
        st.warning("Some prices are sample or cached data, not live quotes.")
    if a["unpriced"]:
        st.warning(f"Could not price: {', '.join(a['unpriced'])} (excluded).")

    m = st.columns(4)
    m[0].metric("Total value", money(a["total_value"]))
    m[1].metric("Diversification", f"{a['diversification_score']:.0f}/100")
    m[2].metric("Risk rating", a["risk_label"], f"{a['risk_score']}/10", delta_color="off")
    m[3].metric("Severe-bear-market loss", f"-{abs(a['stress_loss_pct']):.0f}%", help="Illustrative stress test, not a forecast")
    g1, g2 = st.columns(2)
    g1.plotly_chart(charts.allocation_donut(a["allocation_by_class"], "Allocation by asset class"), width="stretch")
    top = a["positions"][:10]
    g2.plotly_chart(charts.weight_bars([p["symbol"] for p in top], [p["weight"] for p in top], "Largest holdings"), width="stretch")
    st.plotly_chart(charts.weight_bars(list(a["allocation_by_sector"]), list(a["allocation_by_sector"].values()), "Sector exposure"), width="stretch")

    st.markdown("**Things to think about**")
    for o in a["observations"]:
        st.markdown(f"- {o}")
    st.caption(f"Weighted expense ratio {a['weighted_expense_ratio']:.2f}% (about {money(a['annual_fees'])}/year in fund fees).")
    table = pd.DataFrame(
        [{"Ticker": p["symbol"], "Name": p["name"], "Class": p["asset_class"], "Sector": p["sector"], "Value": p["value"], "Weight": p["weight"]} for p in a["positions"]]
    )
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={"Value": st.column_config.NumberColumn(format="$%.0f"), "Weight": st.column_config.NumberColumn(format="percent")})
    st.caption(DISCLAIMER)


def market_tab(bot: FinanceAssistant) -> None:
    st.subheader("Market overview")
    rows = bot.market.market_overview()
    if rows:
        c1, c2 = st.columns([1, 1])
        c1.dataframe(
            pd.DataFrame([{"Market": r["name"], "Symbol": r["symbol"], "Price": r["price"], "Change %": r["change_pct"]} for r in rows]),
            hide_index=True, width="stretch",
            column_config={"Price": st.column_config.NumberColumn(format="$%.2f"), "Change %": st.column_config.NumberColumn(format="%+.2f%%")},
        )
        c2.plotly_chart(charts.day_change_bars(rows), width="stretch")
        first = bot.market.get_quote(rows[0]["symbol"])
        freshness_caption(first)
    else:
        st.error("Market data is unavailable right now.")

    st.divider()
    st.subheader("Look up a stock or ETF")
    a, b = st.columns([2, 1])
    symbol = a.text_input("Ticker", "VTI").strip().upper()
    period = b.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    if symbol:
        try:
            q = bot.market.get_quote(symbol)
            hist = bot.market.get_history(symbol, period)
            trend = bot.market.analyze_trend(symbol, period)
        except Exception as exc:
            st.error(f"Couldn't load {symbol}: {exc}")
            return
        m = st.columns(4)
        m[0].metric(symbol, money(q.price, 2), f"{q.change_pct:+.2f}%")
        m[1].metric("Trend", trend["trend"].title())
        m[2].metric("Volatility (annual)", f"{trend.get('volatility_annual_pct', 0):.1f}%")
        m[3].metric("Worst drawdown", f"{trend.get('max_drawdown_pct', 0):.1f}%")
        st.plotly_chart(charts.price_chart(hist.dates, hist.closes, symbol), width="stretch")
        freshness_caption(hist)
        with st.expander("Related headlines"):
            for n in bot.market.get_news(symbol, 5):
                st.markdown(f"- [{n.title}]({n.url})" if n.url else f"- {n.title}")
                if n.is_sample:
                    st.caption("Sample headline")


def goals_tab() -> None:
    st.subheader("Goal planner")
    c = st.columns(3)
    target = c[0].number_input("Target amount ($)", 1000, 50_000_000, 50_000, step=1000)
    years = c[1].slider("Years to goal", 1, 40, 8)
    risk = c[2].selectbox("Risk appetite", ["conservative", "moderate", "aggressive"],
                          index=["conservative", "moderate", "aggressive"].index(st.session_state.risk))
    c = st.columns(2)
    current = c[0].number_input("Already saved ($)", 0, 50_000_000, 5000, step=500)
    monthly = c[1].number_input("Monthly contribution ($)", 0, 100_000, 300, step=25)
    p = plan_goal(float(target), years, float(current), float(monthly), risk)

    m = st.columns(4)
    m[0].metric("Expected value", money(p["projected_value"]["expected"]))
    m[1].metric("Chance of reaching goal", f"{p['success_probability']:.0%}", help="Share of 2,000 simulated market paths that reach the target")
    m[2].metric("Monthly needed", money(p["required_monthly"]["expected"]))
    m[3].metric("Status", p["status"])
    st.plotly_chart(charts.goal_fan(p), width="stretch")
    a = p["allocation"]
    st.markdown(
        f"**Illustrative mix** for a {risk} investor over {years} years: **{a['stocks']}% stocks / {a['bonds']}% bonds / {a['cash']}% cash**. "
        f"Assumed return {p['expected_return']:.1%}/yr (cautious {p['scenarios']['cautious']:.1%}, optimistic {p['scenarios']['optimistic']:.1%}), "
        f"volatility {p['volatility']:.0%}."
    )
    with st.expander("Show yearly numbers"):
        mc = p["monte_carlo"]
        st.dataframe(pd.DataFrame({"Year": mc["years"], "Low (10th pct)": mc["p10"], "Median": mc["p50"], "High (90th pct)": mc["p90"]}),
                     hide_index=True, width="stretch")
    st.caption("Projections are illustrative, based on simple assumptions, and not a promise of results. " + DISCLAIMER)


def knowledge_tab(bot: FinanceAssistant) -> None:
    st.subheader("Knowledge base")
    st.caption(f"{bot.retriever.document_count} entries in {len(bot.retriever.categories)} categories. Answers in chat cite these sources.")
    c = st.columns([3, 1])
    q = c[0].text_input("Search", placeholder="e.g. wash sale, expense ratio, rebalancing")
    cat = c[1].selectbox("Category", ["all"] + bot.retriever.categories)
    if q:
        hits = bot.retriever.search(q, k=6, category=None if cat == "all" else cat)
        if not hits:
            st.info("No matching articles.")
        for h in hits:
            with st.expander(f"{h.title}  ·  {h.category}  ·  relevance {h.score:.2f}"):
                st.write(h.text)
                if h.reference:
                    st.caption(f"Reference: {h.reference}")


def main() -> None:
    init_state()
    bot = get_assistant()
    sidebar(bot)
    st.warning("Educational tool only - not financial advice.", icon="⚠️")
    tabs = st.tabs(["💬 Chat", "📊 Portfolio", "📈 Market", "🎯 Goals", "📚 Knowledge"])
    with tabs[0]:
        chat_tab(bot)
    with tabs[1]:
        portfolio_tab(bot)
    with tabs[2]:
        market_tab(bot)
    with tabs[3]:
        goals_tab()
    with tabs[4]:
        knowledge_tab(bot)


main()
