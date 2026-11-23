"""Savings agent.

Sets the emergency-fund target from income volatility, dependents and household structure,
then schedules funding. Income volatility comes from the budget agent's shared finding when
present: the savings agent never relearns what the budget agent already knows.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentContext, AgentResult, Recommendation, money_json
from app.engine import emergency_fund as ef
from app.engine.priorities import Step

FUNDING_HORIZON_MONTHS = 12


class SavingsAgent(Agent):
    name = "savings"
    description = "Emergency fund target, runway and funding schedule"

    async def run(self, ctx: AgentContext) -> AgentResult:
        p = ctx.profile
        essential, liquid = p.monthly_essential_expenses, p.liquid_savings
        if essential is None or liquid is None:
            raise ValueError("SavingsAgent requires essential expenses and liquid savings")

        cv = ctx.income_cv()
        dependents = ctx.dependents()
        target = ef.build_target(essential, income_cv=cv, dependents=dependents, single_income=p.single_income)
        status = ef.status(liquid, target)

        volatility_hit = ctx.latest_finding("income_volatility")
        memory_ids = [volatility_hit[0].record.id] if volatility_hit else []
        kids = [h.doc.id for h in await self.cite(ctx, "emergency fund months of expenses", limit=1)] + [target.rule_id]

        wf = ctx.waterfall()
        contribution = wf.allocated_to(Step.STARTER_BUFFER, Step.FULL_EMERGENCY_FUND) if wf else None
        months_needed = ef.months_to_fund(status.shortfall, contribution) if contribution is not None else None
        required = ef.required_monthly_contribution(status.shortfall, FUNDING_HORIZON_MONTHS)
        assumptions = list(wf.assumptions) if wf else []

        parts = ", ".join(f"{k.replace('_', ' ')} +{v}" for k, v in target.components.items() if k != "base" and v > 0)
        why_target = f"3 months base{(', ' + parts) if parts else ''} = {target.months} months of {essential}"
        inputs = {"income_cv": round(cv, 3), "dependents": dependents, "single_income": p.single_income,
                  "components": {k: str(v) for k, v in target.components.items()}}
        recs: list[Recommendation] = []

        if status.is_funded:
            recs.append(await self.recommend(
                ctx, title="Emergency fund fully funded",
                summary=f"You hold {liquid} against a target of {target.amount} ({target.months} months): about {status.runway_months:.1f} months of runway.",
                reasoning=f"Target: {why_target}. Liquid savings {liquid} cover it.",
                metrics={"runway_months": f"{status.runway_months:.2f}", "target_months": str(target.months)},
                memory_ids=memory_ids, knowledge_ids=kids, inputs=inputs, priority=5))
        else:
            schedule = (f" At {contribution} a month that takes {months_needed} months."
                        if contribution is not None and months_needed is not None and contribution.minor > 0
                        else f" Saving {required} a month would close it in {FUNDING_HORIZON_MONTHS} months.")
            recs.append(await self.recommend(
                ctx, title=f"Build your emergency fund to {target.months} months",
                summary=(f"Target {target.amount} ({target.months} months of essentials); you hold {liquid} "
                         f"({status.runway_months:.1f} months). Shortfall: {status.shortfall}.{schedule}"),
                reasoning=f"Target: {why_target}. Shortfall = {target.amount} - {liquid} = {status.shortfall}.",
                metrics={"target_months": str(target.months), "target_amount": str(target.amount),
                         "shortfall": str(status.shortfall), "runway_months": f"{status.runway_months:.2f}",
                         "months_to_fund": months_needed, "required_monthly_12": str(required)},
                memory_ids=memory_ids, knowledge_ids=kids, inputs=inputs, priority=2))

        finding = await self.share_finding(
            ctx, "emergency_fund_status",
            f"Emergency fund is {'funded' if status.is_funded else 'underfunded'} ({status.runway_months:.1f} of {target.months} months).",
            {"funded": status.is_funded, "runway_months": float(status.runway_months), "target_months": float(target.months)})

        plan = {
            "target_months": str(target.months),
            "target_amount": money_json(target.amount),
            "liquid_savings": money_json(liquid),
            "shortfall": money_json(status.shortfall),
            "runway_months": round(float(status.runway_months), 2),
            "funded": status.is_funded,
            "monthly_contribution": money_json(contribution),
            "months_to_fund": months_needed,
            "required_monthly_12": money_json(required),
            "components": {k: str(v) for k, v in target.components.items()},
            "assumptions": assumptions,
        }
        return AgentResult(self.name, "ok", tuple(recs), plan=plan, findings=(finding,))