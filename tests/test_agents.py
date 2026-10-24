import pytest

from src.agents.finance_qa import FinanceQAAgent
from src.agents.goal_agent import GoalPlanningAgent, parse_goal
from src.agents.market_agent import MarketAnalysisAgent
from src.agents.news_agent import NewsSynthesizerAgent, sentiment
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.agents.tax_education import TaxEducationAgent
from src.core.models import AgentContext


@pytest.fixture
def kw(retriever, market):
    return dict(retriever=retriever, market=market)


# ---------------- Finance Q&A
def test_qa_answers_with_citations(kw):
    r = FinanceQAAgent(**kw).handle("What is an ETF?", AgentContext())
    assert r.ok and r.sources and "[1]" in r.content and "Sources" in r.content
    assert r.data["hits"] >= 1


def test_qa_no_hits_graceful(kw):
    r = FinanceQAAgent(**kw).handle("zxqv wlkj", AgentContext())
    assert r.ok and r.data["hits"] == 0 and "couldn't find" in r.content


def test_qa_without_retriever():
    r = FinanceQAAgent().handle("what is a bond", AgentContext())
    assert "couldn't find" in r.content


def test_qa_follow_up_uses_previous_question(kw):
    ctx = AgentContext(history=[("user", "what is dividend yield"), ("assistant", "...")])
    r = FinanceQAAgent(**kw).handle("why is that?", ctx)
    assert any("ividend" in s.title for s in r.sources)


def test_qa_llm_polish_keeps_sources(kw, fake_llm):
    r = FinanceQAAgent(llm=fake_llm, **kw).handle("What is an ETF?", AgentContext())
    assert r.content.startswith("POLISHED") and r.sources


def test_qa_category_hint(kw):
    ctx = AgentContext(shared={"category_hint": "bonds"})
    r = FinanceQAAgent(**kw).handle("what is a coupon", ctx)
    assert all(s.category == "bonds" for s in r.sources)


# ---------------- Tax
def test_tax_agent_table_and_disclaimer(kw):
    r = TaxEducationAgent(**kw).handle("Roth IRA vs 401k", AgentContext())
    assert "| Account |" in r.content and "Roth IRA" in r.content and "IRS.gov" in r.content
    assert set(r.data["accounts"]) >= {"roth", "401k"}
    assert all(s.category in {"tax", "retirement", "glossary"} for s in r.sources)


def test_tax_agent_compare_without_names_lists_all(kw):
    r = TaxEducationAgent(**kw).handle("compare account types", AgentContext())
    assert len(r.data["accounts"]) == 5


def test_tax_agent_single_concept_has_no_table(kw):
    r = TaxEducationAgent(**kw).handle("explain capital gains tax", AgentContext())
    assert "| Account |" not in r.content and "IRS.gov" in r.content


# ---------------- Portfolio
def test_portfolio_from_query(kw):
    ctx = AgentContext()
    r = PortfolioAnalysisAgent(**kw).handle("I own 10 shares of VTI and $1,000 in BND", ctx)
    assert r.ok and "Portfolio value" in r.content and "Diversification score" in r.content
    assert r.data["holdings"] and "portfolio_analysis" in ctx.shared  # shared for downstream agents
    assert "sample/cached" in r.content  # mock prices are labelled


def test_portfolio_from_context_state(kw):
    ctx = AgentContext(portfolio=[{"symbol": "VOO", "shares": 2}])
    assert "VOO" in PortfolioAnalysisAgent(**kw).handle("review my portfolio", ctx).content


def test_portfolio_missing_input_prompts_user(kw):
    r = PortfolioAnalysisAgent(**kw).handle("analyze my portfolio", AgentContext())
    assert r.data["needs_input"] and "Portfolio** tab" in r.content


def test_portfolio_unpriceable(kw):
    ctx = AgentContext(portfolio=[{"symbol": "NOSUCH", "shares": 5}])
    r = PortfolioAnalysisAgent(**kw).handle("analyze my portfolio", ctx)
    assert not r.ok and "couldn't price" in r.content


# ---------------- Market
def test_market_single_ticker(kw):
    ctx = AgentContext()
    r = MarketAnalysisAgent(**kw).handle("How is NVDA doing?", ctx)
    assert r.ok and "NVDA: $140.00" in r.content and "Sample data" in r.content
    assert ctx.shared["tickers"] == ["NVDA"]


def test_market_uses_remembered_tickers(kw):
    r = MarketAnalysisAgent(**kw).handle("what's the price now?", AgentContext(last_tickers=["AAPL"]))
    assert "AAPL" in r.content


def test_market_overview_when_no_ticker(kw):
    r = MarketAnalysisAgent(**kw).handle("how is the market today", AgentContext())
    assert "Market snapshot" in r.content and "S&P 500" in r.content and r.data["overview"]


def test_market_requires_service():
    r = MarketAnalysisAgent().handle("market today", AgentContext())
    assert not r.ok and "market data service" in r.error


def test_market_handles_ticker_without_data(kw):
    r = MarketAnalysisAgent(**kw).handle("price of $ZZZZ", AgentContext())
    assert "no data available" in r.content


# ---------------- Goals
def test_parse_goal_full_sentence():
    g = parse_goal("I want $50,000 in 8 years. I have $5,000 and can save $300 a month. I'm an aggressive investor", AgentContext())
    assert (g["target"], g["years"], g["current"], g["monthly"], g["risk"]) == (50000, 8, 5000, 300, "aggressive")


def test_parse_goal_variants():
    g = parse_goal("save 1.5 million for retirement in 25 years, $2k/month", AgentContext())
    assert g["target"] == 1_500_000 and g["monthly"] == 2000 and g["goal_type"] == "retirement"
    g = parse_goal("down payment of $80k in 5 years, already saved $10,000", AgentContext(profile={"risk_tolerance": "low"}))
    assert g["current"] == 10000 and g["goal_type"] == "home down payment" and g["target"] == 80000


def test_parse_goal_uses_profile_age_for_retirement():
    g = parse_goal("I want to retire with $2 million", AgentContext(profile={"age": 40}))
    assert g["years"] == 25 and g["target"] == 2_000_000


def test_goal_agent_plan(kw):
    ctx = AgentContext(profile={"risk_tolerance": "moderate"})
    r = GoalPlanningAgent(**kw).handle("I want $50,000 in 8 years, I have $5,000 and can save $300 a month", ctx)
    assert r.ok and "Monte Carlo odds" in r.content and r.data["plan"]["target"] == 50000
    assert "goal_plan" in ctx.shared


def test_goal_agent_asks_for_missing_info(kw):
    r = GoalPlanningAgent(**kw).handle("help me plan my savings goal", AgentContext())
    assert r.data["needs_input"] and "target amount" in r.content


def test_goal_agent_short_horizon_and_portfolio_handoff(kw):
    ctx = AgentContext(shared={"portfolio_analysis": {"empty": False, "equity_weight": 0.9}})
    r = GoalPlanningAgent(**kw).handle("I need $20,000 in 2 years, aggressive, saving $100 monthly", ctx)
    assert "under 5 years" in r.content and "current portfolio holds 90%" in r.content and "close the" in r.content


# ---------------- News
def test_sentiment():
    assert sentiment("Stocks rally on strong earnings") > 0
    assert sentiment("Shares plunge on weak outlook, recession worries") < 0
    assert sentiment("Company announces meeting") == 0


def test_news_digest_with_background(kw):
    r = NewsSynthesizerAgent(**kw).handle("latest news on Tesla", AgentContext())
    assert r.ok and "News digest for TSLA" in r.content and "sample headlines" in r.content
    assert r.sources and "Why it matters" in r.content


def test_news_defaults_to_market_and_requires_service(kw):
    assert "SPY" in NewsSynthesizerAgent(**kw).handle("any headlines?", AgentContext()).content
    assert not NewsSynthesizerAgent().handle("news", AgentContext()).ok


def test_news_empty(kw, monkeypatch):
    agent = NewsSynthesizerAgent(**kw)
    monkeypatch.setattr(agent.market, "get_news", lambda *a, **k: [])
    r = agent.handle("news", AgentContext())
    assert not r.ok and "couldn't retrieve" in r.content
