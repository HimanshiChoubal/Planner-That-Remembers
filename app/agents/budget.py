"""Budget agent.

Owns cash flow. Plans on a conservative income for irregular earners (the 10th-percentile
forecast, not the average), computes the monthly surplus, and shares what it learns
(income volatility and surplus) so the debt, savings and goals agents don't relearn it.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentContext, AgentResult, Recommendation, money_json
from app.context.schemas import IncomePattern, MoneyValue
from app.engine import forecast as fc
from app.engine import metrics as mt
from app.engine import networth as nw
from app.engine.money import Money, sum_money

HORIZON_MONTHS = 3
P10_Z = 1.28  # normal-approximation z for the 10th percentile, used only without history


class BudgetAgent(Agent):
    name = "budget"
    description = "Cash flow, savings rate and irregular-income planning"

    async def run(self, ctx: AgentContext) -> AgentResult:
        p = ctx.profile
        cur = p.currency
        patterns = ctx.models(IncomePattern)
        income = p.monthly_gross_income or (patterns[0].monthly_mean.to_money() if patterns else None)
        expenses = p.monthly_expenses
        if income is None or expenses is None:
            raise ValueError("BudgetAgent requires income and expenses (gap check should prevent this)")

        history = p.income_history
        kind: str | None = None
        cv: float | None = None
        planning = income
        mean_income = income
        written = []

        if len(history) >= fc.MIN_OBSERVATIONS:
            cv = fc.coefficient_of_variation(history)
            kind = fc.classify_income(history)
            mean_income = sum_money(history, cur) / len(history)
            if kind == "irregular":
                forecast = fc.forecast(fc.fit_series(history), HORIZON_MONTHS, interval=0.8)
                planning = sum_money(forecast.low, cur) / HORIZON_MONTHS
            if not patterns:
                written.append(await ctx.bus.learn_shared(IncomePattern(
                    kind=kind, monthly_mean=MoneyValue.from_money(mean_income), monthly_cv=cv)))
        elif patterns:
            kind, cv = patterns[0].kind, patterns[0].monthly_cv
            if kind != "salaried":  # no history: approximate the 10th percentile from the stored CV
                planning = income * max(0.3, 1.0 - P10_Z * cv)

        surplus = planning - expenses
        rate = nw.savings_rate(planning, expenses)

        if cv is not None:
            written.append(await self.share_finding(
                ctx, "income_volatility",
                f"Income is {kind}; variability (CV) is {cv:.2f}.",
                {"cv": cv, "kind": kind}))
        written.append(await self.share_finding(
            ctx, "monthly_surplus",
            f"Monthly surplus after expenses is {surplus} on a planning income of {planning}.",
            {"surplus_minor": surplus.minor, "currency": cur.code, "planning_income_minor": planning.minor}))

        knowledge = await self.cite(ctx, "budget savings rate irregular income", limit=1)
        kids = [h.doc.id for h in knowledge]
        recs: list[Recommendation] = []

        if surplus.is_negative():
            recs.append(await self.recommend(
                ctx, title="Spending exceeds planning income",
                summary=f"You spend {expenses} a month against a planning income of {planning}: a gap of {-surplus}.",
                reasoning=f"Planning income {planning} minus expenses {expenses} is {surplus}.",
                metrics={"planning_income": str(planning), "expenses": str(expenses), "gap": str(-surplus)},
                knowledge_ids=kids, priority=1))
        if kind == "irregular":
            recs.append(await self.recommend(
                ctx, title="Budget on your lower-income months, not the average",
                summary=(f"Your average income is {mean_income}, but it swings a lot (variability {cv:.0%}). "
                         f"Plan on about {planning} a month and treat anything above it as a bonus."),
                reasoning=(f"The 10th-percentile forecast of the next {HORIZON_MONTHS} months averages {planning}, "
                           f"versus a historical mean of {mean_income}."),
                metrics={"mean_income": str(mean_income), "planning_income": str(planning), "cv": round(cv or 0, 3)},
                memory_ids=[w.id for w in written], knowledge_ids=kids, priority=2))
        if not surplus.is_negative() and rate < mt.SAVINGS_GOOD:
            gap = planning * mt.SAVINGS_GOOD - surplus
            recs.append(await self.recommend(
                ctx, title="Raise your savings rate toward 20%",
                summary=f"You save {rate:.1%} of planning income; saving another {gap} a month would reach 20%.",
                reasoning=f"Savings rate = ({planning} - {expenses}) / {planning} = {rate:.1%}; the 20% rule of thumb needs {gap} more.",
                metrics={"savings_rate": f"{rate:.3f}", "extra_needed": str(gap)},
                knowledge_ids=kids, priority=3))
        if not recs:
            recs.append(await self.recommend(
                ctx, title="Your budget is on track",
                summary=f"You keep {rate:.1%} of planning income ({surplus} a month).",
                reasoning=f"Savings rate {rate:.1%} meets the 20% rule of thumb.",
                metrics={"savings_rate": f"{rate:.3f}", "surplus": str(surplus)},
                knowledge_ids=kids, priority=5))

        plan = {
            "income_kind": kind,
            "mean_income": money_json(mean_income),
            "planning_income": money_json(planning),
            "monthly_expenses": money_json(expenses),
            "monthly_surplus": money_json(surplus),
            "savings_rate": round(float(rate), 4),
        }
        return AgentResult(self.name, "ok", tuple(recs), plan=plan, findings=tuple(written))