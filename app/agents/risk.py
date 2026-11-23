"""Risk agent.

Flags protection gaps (insurance, thin emergency fund with dependents) that should be closed
before investing. Uses the savings agent's shared emergency-fund finding when present.
The life-cover figure is an ILLUSTRATIVE rule of thumb, not a quote or advice.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentContext, AgentResult, Recommendation, money_json

COVER_YEARS_OF_INCOME = 10  # illustrative rule of thumb per household with dependents


class RiskAgent(Agent):
    name = "risk"
    description = "Insurance and protection gaps"

    async def run(self, ctx: AgentContext) -> AgentResult:
        p = ctx.profile
        covered = bool(p.insurance_covered)
        dependents = ctx.dependents()
        income = p.monthly_gross_income
        kids = [h.doc.id for h in await self.cite(ctx, "insurance coverage protection", limit=1)]
        fund_hit = ctx.latest_finding("emergency_fund_status")
        underfunded = fund_hit is not None and fund_hit[1].data.get("funded") is False
        memory_ids = [fund_hit[0].record.id] if fund_hit else []

        gaps: list[str] = []
        recs: list[Recommendation] = []
        suggested_cover = income * 12 * COVER_YEARS_OF_INCOME if income is not None and dependents > 0 else None

        if not covered:
            gaps.append("insurance")
            who = f" You have {dependents} dependent(s), so the stakes are higher." if dependents else ""
            cover = (f" A common rule of thumb is about {COVER_YEARS_OF_INCOME} years of income in life cover ({suggested_cover})."
                     if suggested_cover is not None else "")
            recs.append(await self.recommend(
                ctx, title="Close your insurance gap before investing",
                summary=f"You don't have health and life coverage in place.{who}{cover} Treat this as a step before extra investing.",
                reasoning=f"insurance_covered is False; dependents: {dependents}.",
                metrics={"insurance_covered": False, "dependents": dependents,
                         "suggested_cover": str(suggested_cover) if suggested_cover is not None else None},
                knowledge_ids=kids, inputs={"dependents": dependents}, priority=1 if dependents else 2))
        if underfunded and dependents > 0:
            gaps.append("thin_buffer_with_dependents")
            recs.append(await self.recommend(
                ctx, title="Your safety net is thin for a household with dependents",
                summary=f"With {dependents} dependent(s) and an underfunded emergency fund, an income shock would hit hard. Prioritise the buffer.",
                reasoning="Savings agent reports the emergency fund is underfunded; dependents > 0.",
                metrics={"dependents": dependents}, memory_ids=memory_ids, knowledge_ids=kids, priority=2))
        if p.single_income and covered is True and dependents > 0 and not underfunded:
            recs.append(await self.recommend(
                ctx, title="Single-income household: review cover levels",
                summary="With one earner and dependents, check that cover would replace that income.",
                reasoning="single_income is True; dependents > 0.", metrics={"dependents": dependents},
                knowledge_ids=kids, priority=4))
        if not recs:
            recs.append(await self.recommend(
                ctx, title="Protection looks in place",
                summary="Coverage is in place and no protection gaps were found.",
                reasoning="insurance_covered is True and no thin-buffer-with-dependents flag.",
                metrics={"insurance_covered": True}, knowledge_ids=kids, priority=5))

        plan = {"insurance_covered": covered, "dependents": dependents, "gaps": gaps,
                "suggested_life_cover": money_json(suggested_cover)}
        return AgentResult(self.name, "ok", tuple(recs), plan=plan)