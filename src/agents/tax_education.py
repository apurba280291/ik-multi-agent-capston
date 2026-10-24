"""Tax Education agent: account types and tax concepts, grounded in the tax knowledge base."""
from __future__ import annotations

import re

from src.agents.finance_qa import FinanceQAAgent
from src.core.models import AgentContext, AgentResponse
from src.utils.formatting import citation_block

ACCOUNT_TABLE = {
    "401k": ("401(k)", "Pre-tax (or Roth) contributions via employer; employer match possible", "Taxed as income at withdrawal (traditional); 10% penalty before ~59.5 with exceptions"),
    "ira": ("Traditional IRA", "Contributions may be deductible; tax-deferred growth", "Taxed as income at withdrawal; RMDs start in your 70s"),
    "roth": ("Roth IRA", "After-tax contributions; tax-free growth and qualified withdrawals", "Income limits apply; contributions (not earnings) withdrawable anytime"),
    "hsa": ("HSA", "Deductible contributions, tax-free growth and tax-free medical withdrawals", "Requires a high-deductible health plan"),
    "529": ("529 Plan", "Tax-free growth and withdrawals for qualified education costs", "Some states give a deduction; non-qualified use is penalised"),
}
_KEYS = {"401k": ("401k", "401(k)", "403b"), "ira": ("ira", "traditional"), "roth": ("roth",), "hsa": ("hsa", "health savings"), "529": ("529", "college savings")}
_COMPARE = re.compile(r"\b(compare|difference|versus|vs\.?|which|types? of accounts?|account types?)\b", re.I)


class TaxEducationAgent(FinanceQAAgent):
    name = "tax_education"
    description = "tax education"
    categories = ["tax", "retirement", "glossary"]

    def run(self, query: str, ctx: AgentContext) -> AgentResponse:
        base = super().run(query, ctx)
        q = query.lower()
        mentioned = [k for k, words in _KEYS.items() if any(w in q for w in words)]
        if _COMPARE.search(query) and len(mentioned) < 2:
            mentioned = list(ACCOUNT_TABLE)
        table = ""
        if len(mentioned) >= 2:
            rows = ["| Account | Tax treatment | Key notes |", "|---|---|---|"]
            rows += ["| " + " | ".join(ACCOUNT_TABLE[k]) + " |" for k in mentioned]
            table = "\n".join(rows)
        content = base.content
        if table:
            content = f"{table}\n\n{content}"
        content += (
            "\n\n*Contribution limits, income phase-outs and tax rules change every year and depend on your "
            "situation - confirm current figures at IRS.gov or with a qualified tax professional.*"
        )
        return AgentResponse(
            agent=self.name, content=content, sources=base.sources, data={**base.data, "accounts": mentioned}, ok=base.ok
        )
