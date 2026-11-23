"""Orchestrator.

One turn: route the message, log it, store any life event, check proposals against remembered
constraints, run the needed agents in dependency order (so shared findings flow forward), save
the plan as a new versioned artifact, diff it against the previous version, and compose a
templated reply. No LLM is involved; every number comes from the agents' engines.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from app.agents.base import Agent, AgentResult, FinancialProfile
from app.agents.budget import BudgetAgent
from app.agents.debt import DebtAgent
from app.agents.goals import GoalsAgent
from app.agents.planner import PlanDiff, build_plan, diff_plans
from app.agents.retirement import RetirementAgent
from app.agents.risk import RiskAgent
from app.agents.router import Route, resolve_agents, route
from app.agents.savings import SavingsAgent
from app.agents.tax import TaxAgent
from app.context.bus import ContextBus
from app.context.conflicts import Conflict, Proposal, blocking, check_proposal
from app.context.schemas import LifeEvent
from app.engine.money import Money
from app.engine.networth import Asset
from app.meko.interfaces import Visibility
from app.regions.registry import get_pack

MAX_REPLY_RECOMMENDATIONS = 3
MAX_REPLY_CHANGES = 3
MAX_REPLY_QUESTIONS = 5


def default_agents() -> dict[str, Agent]:
    return {a.name: a for a in (BudgetAgent(), TaxAgent(), SavingsAgent(), DebtAgent(),
                                GoalsAgent(), RetirementAgent(), RiskAgent())}


@dataclass(frozen=True)
class TurnResult:
    route: Route
    results: Mapping[str, AgentResult]
    plan: Mapping | None
    version: int | None  # plan artifact version saved this turn (None if unchanged or blocked)
    diff: PlanDiff | None
    conflicts: tuple[Conflict, ...]
    questions: tuple[str, ...]
    reply: str
    blocked: bool = False


def apply_life_event(profile: FinancialProfile, event: LifeEvent) -> FinancialProfile:
    """Update the structured profile for a life event using only what the user stated.

    A job loss without a stated income change is taken as losing all income. New costs from the
    event are treated as essential. A change in income clears the income history, because past
    months no longer describe future income.
    """
    cur = profile.currency
    inc = event.monthly_income_delta.to_money() if event.monthly_income_delta else None
    exp = event.monthly_expense_delta.to_money() if event.monthly_expense_delta else None
    if event.event == "job_loss" and inc is None and profile.monthly_gross_income is not None:
        inc = -profile.monthly_gross_income

    changes: dict = {}
    if inc is not None and profile.monthly_gross_income is not None:
        changes["monthly_gross_income"] = profile.monthly_gross_income + inc
        if inc.minor != 0:
            changes["income_history"] = ()
    if exp is not None:
        if profile.monthly_expenses is not None:
            changes["monthly_expenses"] = profile.monthly_expenses + exp
        if profile.monthly_essential_expenses is not None:
            changes["monthly_essential_expenses"] = profile.monthly_essential_expenses + exp
    if event.event == "baby":
        changes["dependents"] = profile.dependents + 1

    if event.one_off_amount is not None and profile.assets is not None:
        amount = event.one_off_amount.to_money()
        assets = list(profile.assets)
        if event.event == "windfall":
            assets.append(Asset("Windfall", "savings", amount))
        else:  # one-off costs draw on liquid assets first
            remaining = amount
            for i, a in enumerate(assets):
                if remaining.minor <= 0:
                    break
                if a.is_liquid:
                    take = min(a.value, remaining)
                    assets[i] = dataclasses.replace(a, value=a.value - take)
                    remaining = remaining - take
        changes["assets"] = tuple(assets)
    return dataclasses.replace(profile, **changes) if changes else profile


class Orchestrator:
    def __init__(self, agents: Mapping[str, Agent] | None = None) -> None:
        self.agents = dict(agents) if agents else default_agents()

    async def handle(
        self,
        bus: ContextBus,
        profile: FinancialProfile,
        text: str,
        *,
        as_of: date | None = None,
        life_event: LifeEvent | None = None,
        proposal: Proposal | None = None,
    ) -> TurnResult:
        bus = bus.for_agent("planner")
        pack = get_pack(profile.region)
        planning_date = as_of or date.today()
        routed = route(text)
        await bus.say("user", text)

        if life_event is not None:
            await bus.remember(life_event, visibility=Visibility.SHARED)
            profile = apply_life_event(profile, life_event)

        memories = await bus.recall("", limit=200)
        prop = proposal or routed.proposal
        conflicts = tuple(check_proposal(prop, memories)) if prop else ()
        if blocking(conflicts):
            reply = "I can't recommend that as stated:\n" + "\n".join(f"- {c.message}" for c in blocking(conflicts)) \
                + "\nTell me if you want to change that limit, or choose something within it."
            await bus.say("assistant", reply)
            return TurnResult(routed, {}, None, None, None, conflicts, (), reply, blocked=True)

        results: dict[str, AgentResult] = {}
        for name in resolve_agents(routed.agents):
            agent = self.agents[name]
            ctx = await agent.context_for(bus, pack, profile, life_event=life_event, as_of=planning_date)
            results[name] = await agent.execute(ctx)

        previous = await bus.latest_artifact("plan")
        plan = build_plan(results, profile=profile, pack=pack, as_of=planning_date,
                          previous=previous.payload if previous else None)
        diff = diff_plans(previous.payload, plan) if previous else None
        version: int | None = None
        has_content = bool(plan["recommendations"])
        if results and has_content and (previous is None or not (diff and diff.is_empty)):
            note = f"Replan after: {life_event.event}" if life_event else f"Update: {routed.intent}"
            trace_ids = [r["trace_id"] for r in plan["recommendations"]]
            saved = await bus.save_artifact("plan", kind="plan", payload=plan, note=note, trace_ids=trace_ids)
            version = saved.version

        questions = list(dict.fromkeys(q["question"] for q in plan["open_questions"]))
        if routed.life_event_kind and life_event is None:
            questions.append(f"What changes does this bring (monthly income, monthly costs, any one-off amount)? "
                             f"Then I can update your plan for the {routed.life_event_kind.replace('_', ' ')}.")
        reply = compose_reply(plan, diff, conflicts, questions, version, ran=bool(results) and has_content)
        top_traces = [r["trace_id"] for r in plan["recommendations"][:MAX_REPLY_RECOMMENDATIONS]]
        await bus.say("assistant", reply, metadata={"plan_version": version, "trace_ids": top_traces})
        return TurnResult(routed, results, plan, version, diff, conflicts, tuple(questions), reply)


def compose_reply(
    plan: Mapping, diff: PlanDiff | None, conflicts: Sequence[Conflict],
    questions: Sequence[str], version: int | None, *, ran: bool,
) -> str:
    """Templated reply built only from plan content (no generated numbers)."""
    lines: list[str] = []
    if ran:
        lines.append(f"Here is your updated plan (version {version})." if version else "Your plan is unchanged.")
        for rec in plan["recommendations"][:MAX_REPLY_RECOMMENDATIONS]:
            lines.append(f"- {rec['title']}: {rec['summary']}")
    if diff is not None and not diff.is_empty:
        lines.append("What changed since the last version:")
        for rc in diff.recommendations[:MAX_REPLY_CHANGES]:
            lines.append(f"  * {rc.kind}: {rc.title}")
        for ch in diff.changes[:MAX_REPLY_CHANGES]:
            lines.append(f"  * {ch.path}: {ch.before} -> {ch.after}")
    warns = [c for c in conflicts if c.severity != "block"]
    if warns:
        lines.append("Things to confirm:")
        lines.extend(f"  * {c.message}" for c in warns)
    if not ran and questions:
        lines.append("I can't build a plan yet: I'm missing some basics.")
    if questions:
        lines.append("To go further I need to know:")
        lines.extend(f"  * {q}" for q in questions[:MAX_REPLY_QUESTIONS])
        if len(questions) > MAX_REPLY_QUESTIONS:
            lines.append(f"  (and {len(questions) - MAX_REPLY_QUESTIONS} more)")
    if not lines:
        lines.append("I need a bit more context before I can help with that.")
    lines.append(plan["disclaimer"] if "disclaimer" in plan else "")
    return "\n".join(line for line in lines if line)