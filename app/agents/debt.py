"""Debt agent.

Compares avalanche and snowball payoff on exact arithmetic and recommends the cheapest. The
extra monthly payment comes from the priority waterfall (shared context from the budget
agent), so debt only gets money once higher-priority steps are covered.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentContext, AgentResult, Recommendation, money_json
from app.engine import debt_payoff as dp
from app.engine.money import Money, sum_money
from app.engine.priorities import HIGH_INTEREST_APR, Step


class DebtAgent(Agent):
    name = "debt"
    description = "Debt payoff order, timeline and interest saved"

    async def run(self, ctx: AgentContext) -> AgentResult:
        p = ctx.profile
        cur = p.currency
        debts = tuple(d for d in (p.debts or ()) if d.balance.minor > 0)
        if not debts:
            return AgentResult(self.name, "ok", plan={"status": "debt_free", "debts": []})

        minimums = sum_money([d.min_payment for d in debts], cur)
        extra, notes, source_ids = self._extra_payment(ctx)
        budget = minimums + extra
        recs: list[Recommendation] = []
        kids = [h.doc.id for h in await self.cite(ctx, "debt payoff avalanche snowball", limit=1)]

        for d in debts:  # a minimum that doesn't cover interest never pays the debt off
            monthly_interest = d.balance * (d.apr / 12)
            if d.min_payment <= monthly_interest:
                recs.append(await self.recommend(
                    ctx, title=f"The minimum on {d.name} doesn't cover its interest",
                    summary=(f"{d.name} costs about {monthly_interest} in interest each month but the minimum "
                             f"payment is {d.min_payment}, so the balance will not shrink on minimums alone."),
                    reasoning=f"Monthly interest {monthly_interest} >= minimum payment {d.min_payment}.",
                    metrics={"monthly_interest": str(monthly_interest), "min_payment": str(d.min_payment)},
                    knowledge_ids=kids, priority=1))

        try:
            comparison = dp.compare_strategies(debts, budget)
        except dp.NeverPaysOffError:
            return AgentResult(
                self.name, "ok", tuple(recs),
                plan={"status": "never_pays_off", "monthly_budget": money_json(budget)})

        best = comparison.best()
        first = next(d for d in debts if d.id == best.order[0])
        saved = comparison.interest_saved(best.strategy)
        avalanche, snowball = comparison.plans["avalanche"], comparison.plans["snowball"]
        snowball_extra = snowball.total_interest - avalanche.total_interest

        extra_text = f"Put {extra} extra toward" if extra.minor > 0 else "Focus first on"
        summary = (f"{extra_text} {first.name} ({first.apr:.2%} APR) and pay minimums on the rest. "
                   f"Debt-free in {best.months} months with {best.total_interest} total interest")
        summary += f", saving {saved} versus minimums only." if saved is not None else "."
        if extra.minor == 0 and notes:
            summary += " " + notes[0]
        recs.append(await self.recommend(
            ctx, title=f"Pay {first.name} first ({best.strategy})", summary=summary,
            reasoning=(f"Monthly debt budget {budget} = minimums {minimums} + extra {extra}. "
                       f"{best.strategy.title()} gives {best.total_interest} interest over {best.months} months."),
            metrics={"strategy": best.strategy, "months": best.months, "total_interest": str(best.total_interest),
                     "interest_saved_vs_minimum": str(saved) if saved is not None else None,
                     "monthly_budget": str(budget), "extra": str(extra)},
            memory_ids=source_ids, knowledge_ids=kids,
            inputs={"debts": [d.id for d in debts], "assumptions": notes}, priority=1 if extra.minor else 2))

        if snowball_extra.minor > 0:
            recs.append(await self.recommend(
                ctx, title="Avalanche beats snowball here",
                summary=f"Paying the smallest balance first (snowball) would cost {snowball_extra} more interest.",
                reasoning=f"Avalanche interest {avalanche.total_interest} vs snowball {snowball.total_interest}.",
                metrics={"avalanche_interest": str(avalanche.total_interest),
                         "snowball_interest": str(snowball.total_interest), "extra_cost": str(snowball_extra)},
                knowledge_ids=kids, priority=4))

        finding = await self.share_finding(
            ctx, "debt_payoff_plan",
            f"Debt-free in {best.months} months; costliest debt is {first.name} at {first.apr:.2%}.",
            {"months": best.months, "extra_minor": extra.minor, "currency": cur.code,
             "high_interest_present": any(d.apr >= HIGH_INTEREST_APR for d in debts)})

        plan = {
            "status": "ok",
            "strategy": best.strategy,
            "order": list(best.order),
            "months": best.months,
            "total_interest": money_json(best.total_interest),
            "interest_saved_vs_minimum": money_json(saved),
            "monthly_budget": money_json(budget),
            "extra_payment": money_json(extra),
            "payoff_month": dict(best.payoff_month),
            "assumptions": notes,
        }
        return AgentResult(self.name, "ok", tuple(recs), plan=plan, findings=(finding,))

    @staticmethod
    def _extra_payment(ctx: AgentContext) -> tuple[Money, list[str], list[str]]:
        """Extra monthly payment beyond minimums, with the assumptions behind it."""
        zero = Money.zero(ctx.profile.currency)
        wf = ctx.waterfall()
        if wf is not None:
            ids = [wf.source_memory_id] if wf.source_memory_id else []
            extra = wf.allocated_to(Step.HIGH_INTEREST_DEBT, Step.OTHER_DEBT)
            notes = list(wf.assumptions)
            if extra.is_zero():
                notes.append("Higher-priority steps (such as the emergency buffer) take this month's surplus first.")
            return extra, notes, ids
        surplus, source = ctx.monthly_surplus()
        if surplus is None:
            return zero, ["Your monthly surplus is unknown, so this plan uses minimum payments only."], []
        extra = zero if surplus.is_negative() else surplus
        return extra, ["Savings or essential spending were unknown, so the full surplus is assumed to go to debt."], (
            [source] if source else [])