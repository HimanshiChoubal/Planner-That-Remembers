"""Goals agent.

For each saved goal, estimates the monthly contribution needed for an 80% chance of success
(Monte Carlo, so every answer is a probability, not a point estimate), accounts for the
user's income volatility (shared by the budget agent), and checks the total against what the
priority waterfall leaves for goals.
"""

from __future__ import annotations

from app.agents.base import Agent, AgentContext, AgentResult, Recommendation, money_json
from app.context.schemas import Goal
from app.engine import goal_probability as gp
from app.engine.money import Money
from app.engine.priorities import Step

TARGET_PROBABILITY = 0.8
MAX_CONTRIBUTION_CV = 0.5


def months_between(start, end) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def return_assumptions(months: int) -> tuple[float, float]:
    """Illustrative, conservative (annual return, annual volatility) by horizon: short goals
    sit in cash-like assets, long goals can take more risk."""
    if months <= 36:
        return 0.02, 0.03
    if months <= 84:
        return 0.04, 0.08
    return 0.05, 0.12


class GoalsAgent(Agent):
    name = "goals"
    description = "Goal funding: required contributions and success probability"

    async def run(self, ctx: AgentContext) -> AgentResult:
        cur = ctx.profile.currency
        cv = min(ctx.income_cv(), MAX_CONTRIBUTION_CV)
        volatility_hit = ctx.latest_finding("income_volatility")
        shared_ids = [volatility_hit[0].record.id] if volatility_hit else []
        kids = [h.doc.id for h in await self.cite(ctx, "saving for goals probability", limit=1)]

        wf = ctx.waterfall()
        available = wf.allocated_to(Step.GOALS_AND_RETIREMENT) if wf else None
        deferred = bool(wf) and wf.status_of(Step.GOALS_AND_RETIREMENT) == "pending"

        goals = sorted(
            ((m, m.model) for m in ctx.memories if isinstance(m.model, Goal)),
            key=lambda t: (t[1].priority, t[1].deadline or ctx.today.replace(year=ctx.today.year + 100)),
        )
        recs: list[Recommendation] = []
        plan_goals: list[dict] = []
        total_required = Money.zero(cur)

        for memory, goal in goals:
            target = goal.target.to_money()
            if target.currency != cur:
                continue
            saved = goal.saved.to_money() if goal.saved else Money.zero(cur)
            entry: dict = {"name": goal.name, "target": money_json(target), "saved": money_json(saved),
                           "deadline": goal.deadline.isoformat() if goal.deadline else None}
            priority = 4 if deferred else 2 + (goal.priority > 3)

            if goal.deadline is None:
                recs.append(await self.recommend(
                    ctx, title=f"Set a target date for {goal.name}",
                    summary=f"Without a deadline I can't work out what to save each month toward {target}.",
                    reasoning="No deadline stored for this goal.", metrics={"goal": goal.name},
                    memory_ids=[memory.record.id], knowledge_ids=kids, priority=4))
                entry["status"] = "needs_deadline"
                plan_goals.append(entry)
                continue

            months = months_between(ctx.today, goal.deadline)
            if months <= 0:
                entry["status"] = "deadline_passed"
                plan_goals.append(entry)
                recs.append(await self.recommend(
                    ctx, title=f"The deadline for {goal.name} has passed",
                    summary=f"Update the date for {goal.name} so I can plan it.",
                    reasoning=f"Deadline {goal.deadline} is not in the future.", metrics={"goal": goal.name},
                    memory_ids=[memory.record.id], knowledge_ids=kids, priority=3))
                continue

            ret, vol = return_assumptions(months)
            common = dict(current=saved, months=months, target=target, annual_return=ret,
                          annual_volatility=vol, contribution_cv=cv)
            already = gp.goal_success_probability(monthly_contribution=Money.zero(cur), **common)
            if already.probability >= TARGET_PROBABILITY:
                entry.update(status="on_track", months=months, probability=round(already.probability, 3))
                recs.append(await self.recommend(
                    ctx, title=f"{goal.name} is already on track",
                    summary=f"What you've set aside ({saved}) is likely to reach {target} by {goal.deadline} ({already.probability:.0%} chance) without further saving.",
                    reasoning=f"Simulated {already.n_sims} paths over {months} months with no new contributions.",
                    metrics={"probability": round(already.probability, 3), "months": months},
                    memory_ids=[memory.record.id, *shared_ids], knowledge_ids=kids, priority=5))
                plan_goals.append(entry)
                continue

            required = gp.required_monthly_contribution(target_probability=TARGET_PROBABILITY, **common)
            outcome = gp.goal_success_probability(monthly_contribution=required, **common)
            total_required = total_required + required
            entry.update(status="needs_saving", months=months, required_monthly=money_json(required),
                         probability=round(outcome.probability, 3),
                         p10=money_json(outcome.p10), p50=money_json(outcome.p50), p90=money_json(outcome.p90))
            plan_goals.append(entry)
            note = " Higher-priority steps come first, so this starts later." if deferred else ""
            recs.append(await self.recommend(
                ctx, title=f"Save {required} a month for {goal.name}",
                summary=(f"To reach {target} by {goal.deadline} with a {TARGET_PROBABILITY:.0%} chance, "
                         f"save about {required} a month for {months} months. Likely outcome range: "
                         f"{outcome.p10} to {outcome.p90}.{note}"),
                reasoning=(f"{outcome.n_sims} simulated paths, assumed return {ret:.0%} (volatility {vol:.0%}), "
                           f"contribution variability {cv:.0%}; contribution chosen so {TARGET_PROBABILITY:.0%} of paths succeed."),
                metrics={"required_monthly": str(required), "months": months, "probability": round(outcome.probability, 3),
                         "p10": str(outcome.p10), "p50": str(outcome.p50), "p90": str(outcome.p90)},
                memory_ids=[memory.record.id, *shared_ids], knowledge_ids=kids,
                inputs={"annual_return": ret, "annual_volatility": vol, "contribution_cv": cv, "seed": outcome.seed},
                priority=priority))

        if available is not None and total_required.minor > 0 and total_required > available:
            shortfall = total_required - available
            recs.append(await self.recommend(
                ctx, title="Your goals need more than your plan can fund",
                summary=(f"Your goals need about {total_required} a month, but only {available} is left for them "
                         f"after higher-priority steps: {shortfall} short. Consider extending a deadline or lowering a target."),
                reasoning=f"Sum of required contributions {total_required} exceeds goal allocation {available}.",
                metrics={"total_required": str(total_required), "available": str(available), "shortfall": str(shortfall)},
                memory_ids=[wf.source_memory_id] if wf and wf.source_memory_id else [], knowledge_ids=kids, priority=2))

        plan = {
            "goals": plan_goals,
            "total_required_monthly": money_json(total_required),
            "available_for_goals": money_json(available),
            "deferred_by_priorities": deferred,
        }
        return AgentResult(self.name, "ok", tuple(recs), plan=plan)